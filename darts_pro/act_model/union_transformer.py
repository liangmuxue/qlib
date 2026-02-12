import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from .cov_cnn import LinelessLayer

# ===================================== 1. 位置编码（无改造，支持连续历史+未来）=====================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0,:,  0::2] = torch.sin(position * div_term)
        pe[0,:,  1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        x = x + self.pe[:,start_pos:start_pos + x.size(1),:]
        return self.dropout(x)

# ===================================== 2. 全局跨对象注意力（全C维度，捕捉5类关联）=====================================
class CrossObjAttentionWithC(nn.Module):
    def __init__(self, d_model: int, M_total: int, M_cov: int, M_tgt: int, C: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.M_total = M_total
        self.M_cov = M_cov
        self.M_tgt = M_tgt
        self.C = C
        self.d_k = d_model // M_total
        assert d_model % M_total == 0, "d_model必须是总对象数M_total的整数倍"

        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cov_c_fuse_mlp = nn.Sequential(nn.Linear(self.d_k, self.d_k), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_k, self.d_k))
        self.tgt_c_fuse_mlp = nn.Sequential(nn.Linear(self.d_k, self.d_k), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.d_k, self.d_k))
        self.q_proj = nn.Linear(self.d_k, self.d_k)
        self.kv_proj = nn.Linear(d_model, self.d_k * 2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        # 时序自注意力
        temporal_out, _ = self.temporal_attn(x, x, x, attn_mask=mask)
        # 按总对象数拆分
        x_split = x.reshape(B, T, self.M_total, self.d_k)
        x_cov, x_tgt = x_split[:, :, :self.M_cov, :], x_split[:, :, self.M_cov:, :]
        # 分类型C维融合
        x_cov_fused = torch.stack([self.cov_c_fuse_mlp(x_cov[:, :, m, :]) for m in range(self.M_cov)], dim=2)
        x_tgt_fused = torch.stack([self.tgt_c_fuse_mlp(x_tgt[:, :, m, :]) for m in range(self.M_tgt)], dim=2)
        x_obj_fused = torch.cat([x_cov_fused, x_tgt_fused], dim=2)
        # 全局跨对象注意力
        cross_out = []
        for m in range(self.M_total):
            q = self.q_proj(x_obj_fused[:, :, m, :])
            kv = self.kv_proj(x_obj_fused.reshape(B, T, D))
            k, v = torch.chunk(kv, 2, dim=-1)
            attn_w = F.softmax(torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(torch.tensor(self.d_k)), dim=-1)
            cross_out.append(torch.matmul(attn_w, v))
        cross_out = torch.cat(cross_out, dim=-1)
        # 融合+残差
        out = self.alpha * self.dropout(cross_out) + (1 - self.alpha) * temporal_out
        return self.norm(out + x)

# ===================================== 3. 损失函数（全C维度，协变量-目标C维关联正则化）=====================================
class CovaTgtCLossWithC(nn.Module):
    def __init__(self, d_model: int, M_cov: int, M_tgt: int, C: int, beta1: float = 0.05, beta2: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.M_cov = M_cov
        self.M_tgt = M_tgt
        self.C = C
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mse = nn.MSELoss(reduction='mean')
        self.mine = nn.Sequential(nn.Linear(2*d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))

    def cova_tgt_corr(self, cov_feat: torch.Tensor, tgt_feat: torch.Tensor) -> torch.Tensor:
        cov_flat = cov_feat.reshape(cov_feat.shape[0], cov_feat.shape[1], -1)
        tgt_flat = tgt_feat.reshape(tgt_feat.shape[0], tgt_feat.shape[1], -1)
        cov_cent = cov_flat - cov_flat.mean(dim=1, keepdim=True)
        tgt_cent = tgt_flat - tgt_flat.mean(dim=1, keepdim=True)
        cov_mat = torch.matmul(cov_cent.transpose(-2, -1), tgt_cent) / (cov_flat.shape[1]-1)
        cov_std = torch.sqrt(torch.sum(cov_cent**2, dim=1, keepdim=True)/(cov_flat.shape[1]-1)+self.eps)
        tgt_std = torch.sqrt(torch.sum(tgt_cent**2, dim=1, keepdim=True)/(tgt_flat.shape[1]-1)+self.eps)
        corr_mat = cov_mat / (torch.matmul(cov_std, tgt_std.transpose(-2, -1))+self.eps)
        return corr_mat.mean(dim=0)

    def mine_mi(self, h_dec: torch.Tensor, fut_cov_enc: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([h_dec, fut_cov_enc], dim=-1)
        pos = self.mine(concat).mean()
        fut_shuffle = fut_cov_enc[torch.randperm(fut_cov_enc.shape[0])]
        concat_shuffle = torch.cat([h_dec, fut_shuffle], dim=-1)
        neg = torch.exp(self.mine(concat_shuffle)).mean()
        return pos - torch.log(neg + self.eps)

    def forward(self, pred: torch.Tensor, true: torch.Tensor, h_dec: torch.Tensor, fut_cov_enc: torch.Tensor,
                cov_feat_hist: torch.Tensor, tgt_feat_hist: torch.Tensor) -> tuple:
        loss_base = self.mse(pred, true)
        corr_pred = self.cova_tgt_corr(cov_feat_hist, pred)
        corr_true = self.cova_tgt_corr(cov_feat_hist, true)
        loss_corr = F.mse_loss(corr_pred, corr_true)
        loss_fut = -self.mine_mi(h_dec, fut_cov_enc)
        loss_total = loss_base + self.beta1 * loss_corr + self.beta2 * loss_fut
        return loss_total, loss_base, loss_corr, loss_fut

# ===================================== 4. 主模型（全张量含C维度，核心实现）=====================================
class MultiTargetTransformerWithFuture(nn.Module):
    def __init__(
        self,
        M_cov_p: int,          # 历史协变量组数
        M_cov_f: int,          # 未来协变量组数
        M_tgt: int,            # 多目标数
        C: int,                # 全张量统一C维度（核心）
        F_s: int,              # 静态协变量基础维度
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 1000
    ):
        super().__init__()
        # 核心维度定义
        self.M_cov_p = M_cov_p
        self.M_cov_f = M_cov_f
        self.M_cov = M_cov_p + M_cov_f  # 总协变量组数
        self.M_tgt = M_tgt
        self.M_total = self.M_cov + M_tgt  # 总对象数（协变量+目标）
        self.C = C
        self.d_model = d_model
        self.d_k = d_model // self.M_total
        assert d_model % self.M_total == 0, "d_model必须是总对象数M_total的整数倍"

        # 1. 静态协变量投影+广播适配（含C维度）
        self.static_proj = nn.Sequential(
            nn.Linear(F_s, self.M_total),
            nn.LayerNorm(self.M_total),
            nn.GELU()
        )
        # 2. C维融合投影层（所有张量共用，保证C维融合逻辑一致）
        self.c_fuse_proj = nn.Sequential(
            nn.Linear(C, self.d_k),
            nn.LayerNorm(self.d_k),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 3. 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 4. 编码器层（含全局跨对象注意力）
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.encoder_cross_obj = CrossObjAttentionWithC(
            d_model, self.M_total, self.M_cov, self.M_tgt, C, nhead, dropout
        )

        # 5. 解码器层（含全局跨对象注意力）
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.decoder_cross_obj = CrossObjAttentionWithC(
            d_model, self.M_total, self.M_cov, self.M_tgt, C, nhead, dropout
        )

        # 6. 输出门控解码层（全C维度适配）
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gate_proj = nn.Linear(d_model, M_tgt)
        self.out_proj = nn.Linear(d_model, M_tgt * C)
        
    def forward(
        self,
        x_cov_p: torch.Tensor,        # 历史协变量 [B, T_past, M_cov_p, C]
        x_cov_f_past: torch.Tensor,   # 未来协变量历史段 [B, T_past, M_cov_f, C]
        y_past: torch.Tensor,         # 历史多目标 [B, T_past, M_tgt, C]
        x_cov_f_fut: torch.Tensor,    # 未来协变量未来段 [B, T_fut, M_cov_f, C]
        x_static: torch.Tensor,       # 静态协变量 [B, F_s]
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_past, _, _ = x_cov_p.shape
        _, T_fut, _, _ = x_cov_f_fut.shape

        # ===================================== 步骤1：编码器输入全C维融合（核心）=====================================
        # 协变量拼接（历史+未来协变量历史段）[B,T_past,M_cov,C]
        x_cov_total_past = torch.cat([x_cov_p, x_cov_f_past], dim=2)
        # 协变量-目标拼接（总对象数）[B,T_past,M_total,C]
        X_enc_raw = torch.cat([x_cov_total_past, y_past], dim=2)
        # C维融合+投影 [B,T_past,M_total,d_k] → [B,T_past,d_model]
        X_c_fused = self.c_fuse_proj(X_enc_raw)
        X_enc_in = X_c_fused.reshape(B, T_past, self.d_model)

        # ===================================== 步骤2：静态协变量广播适配（含C维度）=====================================
        x_static_proj = self.static_proj(x_static.permute(0,2,1)).reshape(B, 1, self.M_total, self.C)  # [B,1,M_total,C]
        x_static_c_fused = self.c_fuse_proj(x_static_proj).reshape(B, 1, self.d_model)  # [B,1,d_model]
        x_static_past = x_static_c_fused.repeat(1, T_past, 1)  # [B,T_past,d_model]
        x_static_fut = x_static_c_fused.repeat(1, T_fut, 1)    # [B,T_fut,d_model]

        # ===================================== 步骤3：编码器编码（全C维度关联捕捉）=====================================
        x_past_enc = self.pos_enc(X_enc_in, start_pos=0) + x_static_past
        enc_out = self.encoder(x_past_enc, mask=src_mask)
        enc_out = self.encoder_cross_obj(enc_out, mask=src_mask)

        # ===================================== 步骤4：解码器输入全C维适配（未来协变量）=====================================
        # 未来协变量C维融合+投影+补0 [B,T_fut,M_cov_f,d_k] → [B,T_fut,d_model]
        x_cov_f_fused = self.c_fuse_proj(x_cov_f_fut)
        x_cov_f_proj = x_cov_f_fused.reshape(B, T_fut, -1)
        pad_dim = self.d_model - x_cov_f_proj.shape[-1]
        x_cov_f_proj_pad = F.pad(x_cov_f_proj, (0, pad_dim))
        # 位置编码+静态特征融合（连续位置）
        x_fut_enc = self.pos_enc(x_cov_f_proj_pad, start_pos=T_past) + x_static_fut

        # ===================================== 步骤5：解码器解码（历史-未来C维联动）=====================================
        dec_out = self.decoder(
            tgt=x_fut_enc, memory=enc_out,
            tgt_mask=tgt_mask, memory_mask=None
        )
        dec_out = self.decoder_cross_obj(dec_out, mask=tgt_mask)

        # ===================================== 步骤6：输出解码（恢复全C维度多目标）=====================================
        h_fuse = torch.cat([dec_out, x_fut_enc], dim=-1)
        h_fuse_proj = self.fuse_proj(h_fuse)
        # 门控权重适配C维
        gate = torch.sigmoid(self.gate_proj(h_fuse_proj)).unsqueeze(-1)
        # 基础预测恢复[B,T_fut,M_tgt,C]
        y_base_flat = self.out_proj(h_fuse_proj)
        y_base = y_base_flat.reshape(B, T_fut, self.M_tgt, self.C)
        # 门控融合
        y_pred = gate * y_base + (1 - gate) * y_base.mean(dim=2, keepdim=True)

        # 返回预测值+编码器输出+解码器输出+历史协变量（用于损失计算）
        return y_pred, enc_out, dec_out, x_cov_total_past


class UnionTransCombine(nn.Module):
    """整合后的完整模型"""

    def __init__(
        self,
        M_cov_p: int,          # 历史协变量组数
        M_cov_f: int,          # 未来协变量组数
        M_tgt: int,            # 多目标数
        C: int,                # 全张量统一C维度（核心）
        F_s: int,              # 静态协变量基础维度
        pred_len: int = 5,     # 预测长度
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = 100,
        hidden_size=16,
    ):
        super().__init__()
        self.trans_model = MultiTargetTransformerWithFuture(
            M_cov_p=M_cov_p, M_cov_f=M_cov_f, F_s=F_s,M_tgt=M_tgt,C=C,
            d_model=d_model,nhead=nhead,num_layers=num_layers,dropout=dropout,max_len=max_len
        )    
        self.pred_len = pred_len           
        # 整合输出网络
        self.ins_layer = LinelessLayer(C*pred_len,C,
                            hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)
        self.ins_att_layer = LinelessLayer(C*pred_len,C,
                        hidden_size=hidden_size,layer_norm=True,batch_norm=False)    
        # 指数整合输出网络       
        self.index_combine_layer = LinelessLayer(C*pred_len,pred_len)     
            
    def forward(
        self,
        x_cov_p: torch.Tensor,        # 历史协变量 [B, T_past, M_cov_p, C]
        x_cov_f_past: torch.Tensor,   # 未来协变量历史段 [B, T_past, M_cov_f, C]
        y_past: torch.Tensor,         # 历史多目标 [B, T_past, M_tgt, C]
        x_cov_f_fut: torch.Tensor,    # 未来协变量未来段 [B, T_fut, M_cov_f, C]
        x_static: torch.Tensor,       # 静态协变量 [B, F_s]
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:    
        
        # 基础模型的向前传播
        y_pred, enc_out, dec_out, cov_feat_hist = self.trans_model(
            x_cov_p, x_cov_f_past, y_past, x_cov_f_fut, x_static,src_mask=src_mask,tgt_mask=tgt_mask
        )   
        cls_out_combine = []
        index_data_combine = []
        dec_out_combine = []        
        # 品种间比较目标的网络输出
        y_pred = y_pred.permute(0,3,1,2)
        pred_reshape = y_pred.reshape(y_pred.shape[0],-1)
        dec_out_combine.append(y_pred)   
        dec_out_combine = torch.cat(dec_out_combine,dim=1).squeeze(-1)
        # 主要比较目标输出
        cls_out_ins = self.ins_layer(pred_reshape)
        # 添加辅助品种比较目标输出
        cls_out_ins_att = self.ins_att_layer(pred_reshape)    
        cls_out_combine.append(cls_out_ins)
        cls_out_combine.append(cls_out_ins_att)  
        # 整体指数预测的网络输出
        sw_index_data = self.index_combine_layer(pred_reshape)     
        index_data_combine = sw_index_data
        
        return dec_out_combine,cls_out_combine,index_data_combine 
        
# ===================================== 5. 测试代码（验证全C维度维度正确性）=====================================
if __name__ == "__main__":
    # 超参数设置（可根据业务自由调整）
    B, T_past, T_fut = 4, 10, 5  # 批次、历史长度、预测长度
    M_cov_p, M_cov_f, M_tgt = 2, 1, 3  # 历史协变量组=2，未来协变量组=1，多目标=3
    C = 3  # 全张量统一C维度（核心，所有协变量+目标都是3维）
    F_s = 4  # 静态协变量基础维度
    d_model, nhead, num_layers = 96, 8, 2  # d_model=96 = M_total(2+1+3)*d_k(16)

    # 生成测试数据（严格匹配全C维度）
    x_cov_p = torch.randn(B, T_past, M_cov_p, C)        # [4,10,2,3] 历史协变量（含C）
    x_cov_f_past = torch.randn(B, T_past, M_cov_f, C)   # [4,10,1,3] 未来协变量历史段（含C）
    y_past = torch.randn(B, T_past, M_tgt, C)           # [4,10,3,3] 历史多目标（含C）
    x_cov_f_fut = torch.randn(B, T_fut, M_cov_f, C)     # [4,5,1,3] 未来协变量未来段（含C）
    x_static = torch.randn(B, F_s)                      # [4,4] 静态协变量
    y_true = torch.randn(B, T_fut, M_tgt, C)            # [4,5,3,3] 未来真实多目标（含C）

    # 初始化模型、损失函数、优化器
    model = MultiTargetTransformerWithFuture(
        M_cov_p=M_cov_p, M_cov_f=M_cov_f, M_tgt=M_tgt, C=C, F_s=F_s,
        d_model=d_model, nhead=nhead, num_layers=num_layers
    )
    loss_fn = CovaTgtCLossWithC(
        d_model=d_model, M_cov=M_cov_p+M_cov_f, M_tgt=M_tgt, C=C
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 前向传播
    y_pred, enc_out, dec_out, cov_feat_hist = model(
        x_cov_p, x_cov_f_past, y_past, x_cov_f_fut, x_static
    )
    # 计算损失（需传入历史协变量特征用于相关性正则化）
    loss_total, loss_base, loss_corr, loss_fut = loss_fn(
        y_pred, y_true, dec_out, model.fut_proj(x_cov_f_fut),
        cov_feat_hist, y_past
    )
    # 反向传播+参数更新
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止爆炸
    optimizer.step()

    # 打印维度和损失（验证全C维度正确性）
    print(f"预测值形状: {y_pred.shape} | 预期形状: ({B}, {T_fut}, {M_tgt}, {C})")
    print(f"总损失: {loss_total.item():.4f} | 基础MSE损失: {loss_base.item():.4f}")
    print(f"协变量-目标C维相关性损失: {loss_corr.item():.4f} | 未来协变量C维MI损失: {loss_fut.item():.4f}")