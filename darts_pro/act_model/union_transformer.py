import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from cus_utils.common_compute import normalization_standard
import cus_utils.global_var as global_var
from darts_pro.tft_futures_dataset import concat_scale_arr,emb_scale_arr

from .cov_cnn import LinelessLayer

PRINT_STD_FLAG = False

def set_global(value):
    global PRINT_STD_FLAG
    PRINT_STD_FLAG = value
    
# ---------------------- 核心新增：样本维度交互模块 ----------------------
class SampleCrossAttention(nn.Module):
    """样本维度自注意力：建模样本间（站点/设备/用户）的关联"""
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        # 样本间自注意力层
        self.attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # 门控融合：控制交互信息的权重
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid()
        )
        # 层归一化
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        """
        x: [B, S, F]  B=batch, S=样本维度（站点/设备数）, F=特征维度
        return: [B, S, F]  融合样本间关联后的特征
        """
        # 样本间自注意力计算
        attn_out, attn_weights = self.attn(x, x, x)
        
        # 门控融合：原特征 + 交互特征
        gate_weight = self.gate(torch.cat([x, attn_out], dim=-1))
        out = x * (1 - gate_weight) + attn_out * gate_weight
        
        # 层归一化
        out = self.norm(out)
        return out, attn_weights

# ---------------------- 工具函数 ----------------------
def generate_causal_mask(seq_len, device):
    """生成因果掩码（禁止访问未来）"""
    mask = (torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TimeFeatureEncoder:
    """时间特征编码：提取年/月/日/小时等周期特征并嵌入"""
    def __init__(self, time_col, embed_dims=None,device='cpu'):
        self.time_col = time_col
        self.encoders = {}
        self.embed_layers = nn.ModuleDict()
        self.device = device
        
        # 默认嵌入维度
        if embed_dims is None:
            self.embed_dims = {
                'year': 2, 'month': 2, 'day': 2,'dayofweek': 2
            }
        else:
            self.embed_dims = embed_dims
    
    def to_device(self):
        self.embed_layers = self.embed_layers.to(self.device)

    def fit_static(self, range_data):
        """使用固定数值范围，拟合时间特征编码器"""
        
        # 为离散时间特征创建LabelEncoder
        for feat_name in range_data.keys():
            le = LabelEncoder()
            le.fit(range_data[feat_name])
            self.encoders[feat_name] = le            
            num_classes = len(le.classes_)
            # 创建嵌入层
            self.embed_layers[feat_name] = nn.Embedding(num_classes, self.embed_dims[feat_name])
        
        return self
    
    def transform(self, df, device='cpu'):
        """转换时间特征为嵌入向量"""
        df_time = pd.to_datetime(df[self.time_col])
        time_feats = {
            'year': df_time.dt.year.values,
            'month': df_time.dt.month.values,
            'day': df_time.dt.day.values,
            'dayofweek': df_time.dt.dayofweek.values,
            # 'week': np.array(df_time.dt.isocalendar().week.values).astype(int),
        }
        
        embed_list = []
        for feat_name, feat_vals in time_feats.items():
            encoded = torch.tensor(self.encoders[feat_name].transform(feat_vals), device=self.device)
            embed = self.embed_layers[feat_name](encoded)
            embed_list.append(embed)
        
        # 拼接所有时间嵌入
        time_embed = torch.cat(embed_list, dim=-1)  # (n_samples, time_embed_dim)
        return time_embed

    def transform_inner(self, batch_data):
        """批次内转换时间特征为嵌入向量"""
        
        embed_list = []
        for feat_name in batch_data.keys():
            feat_vals = batch_data[feat_name]
            encoded = torch.tensor(self.encoders[feat_name].transform(feat_vals.cpu()), device=self.device)
            embed = self.embed_layers[feat_name](encoded)
            embed_list.append(embed)
        
        # 拼接所有时间嵌入
        time_embed = torch.cat(embed_list, dim=-1)  # (n_samples, time_embed_dim)
        return time_embed
    
# ---------------------- TFT核心模块 ----------------------
class GatedResidualNetwork(nn.Module):
    """门控残差网络（GRN）：TFT核心特征处理模块"""
    def __init__(self, input_dim, hidden_dim, output_dim, context_combine_dim=None,dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        # 和context的融合权重
        if context_combine_dim is not None:
            self.fusion_gate = nn.Linear(context_combine_dim+1, context_combine_dim)
        # 缩放因子，减少ctx权重
        self.scale_x = 1
        self.scale_ctx = 0.1
        # 归一化层级
        self.norm_x = nn.LayerNorm(hidden_dim)
        self.norm_ctx = nn.LayerNorm(hidden_dim)         
        # 残差连接适配
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, context=None,scale_ctx=0.1,no_ctx_squeeze=False):
        """
        x: (batch, seq_len, input_dim) 或 (batch, input_dim)
        """
        # 前向传播
        x_res = self.residual(x)
        x = F.elu(self.fc1(x))
        # 融入静态上下文信息
        if context is not None:
            if not no_ctx_squeeze:
                ctx = context.unsqueeze(-2)
            else:
                ctx = context
            # 先归一化再融合
            x_scaled = self.norm_x(x)
            ctx_scaled = self.norm_ctx(ctx) * scale_ctx               
            x = x_scaled +  ctx_scaled         
        x = self.dropout(x)
        x = self.fc2(x)
        # 门控机制
        gate = torch.sigmoid(self.gate(x_res))
        x = x * gate
        
        # 残差+层归一
        x = x + x_res*(1-gate)
        x = self.layer_norm(x)
        return x

class VariableSelectionNetwork(nn.Module):
    """变量选择网络：对输入特征加权，突出重要变量"""
    def __init__(self, input_dim, num_vars, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.grn = GatedResidualNetwork(input_dim, hidden_dim, num_vars, dropout=dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        return: 
            weighted_x: (batch, seq_len, input_dim) 加权后的特征
            var_weights: (batch, seq_len, num_vars) 变量权重（可解释性）
        """
        # 计算变量权重
        var_weights = self.grn(x)  # (batch, seq_len, num_vars)
        var_weights = F.softmax(var_weights, dim=-1)
        
        # 重塑为加权矩阵
        batch_size, seq_len, _ = x.shape
        x_reshaped = x.reshape(batch_size, seq_len, self.num_vars, -1)  # (batch, seq_len, num_vars, var_dim)
        var_weights_expanded = var_weights.unsqueeze(-1)  # (batch, seq_len, num_vars, 1)
        
        # 加权特征
        weighted_x = (x_reshaped * var_weights_expanded).reshape(batch_size, seq_len, -1)
        return weighted_x, var_weights

class DecoderLayer(nn.Module):
    def __init__(
        self,
        obs_dim=6,             # 历史观测特征维度（要预测的变量）
        hidden_dim=64,         # 隐藏层维度
        dropout=0.1,
        pred_len=1,            # 预测步长
        sample_dim=3,    
    ):
        super().__init__()
        self.pred_len = pred_len
        self.sample_dim = sample_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.line_hidden_dim = 4
        # 未来协变量融合网络（关键：融合历史编码+未来协变量）
        self.future_fusion = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        # 门控注意力融合
        self.attention_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        # 输出层（预测目标变量）
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        
        self.hidden_proj = nn.Linear(hidden_dim, self.line_hidden_dim)
        self.final_proj = nn.Linear(self.line_hidden_dim, obs_dim)
        # nn.init.xavier_normal_(self.final_proj.weight, gain=0.1)
        # nn.init.zeros_(self.final_proj.bias)
        self.dec_norm = nn.BatchNorm1d(self.line_hidden_dim)
        self.fuse_drop = nn.Dropout(dropout)
        
    def forward(self,hist_summary,fut_proj=None,fur_scale=0.2):
        
        S = self.sample_dim
        P = self.pred_len
        B = int(hist_summary.shape[0]/S)
        
        # 融合历史总结+未来协变量（核心：历史指导未来预测）
        hist_summary_expanded = hist_summary.unsqueeze(1).repeat(1, P, 1)  # [B*S, P, hidden_dim]
        if fut_proj is not None:
            fut_proj = fut_proj.reshape(B*S,fut_proj.shape[-1])
            # 使用缩放避免未来协变量权重过高
            # final_feat = self.future_fusion(hist_summary_expanded,context=fut_proj,scale_ctx=fur_scale)  # [B*S, P, hidden_dim]
            final_feat = hist_summary + fut_proj * fur_scale
        else:
            # 门控融合
            final_feat = hist_summary_expanded # self.future_fusion(hist_summary_expanded)  # [B*S, P, hidden_dim]
        
        # 输出层（每个预测步独立预测）
        final_feat_flat = final_feat.reshape(B*S*P, self.hidden_dim)  # [B*S*P, hidden_dim]
        final_feat_flat = self.hidden_proj(final_feat_flat)
        final_feat_flat = self.dec_norm(final_feat_flat)  
        pred_flat = self.final_proj(final_feat_flat)        # [B*S*P, obs_dim]   
        final_feat_flat = self.fuse_drop(final_feat_flat)
           
        # 恢复维度
        pred = pred_flat.reshape(B*S, P, self.obs_dim)  # [B*S, P, obs_dim]
        pred = pred.reshape(B, S, P, self.obs_dim)      # [B, S, P, obs_dim]   
             
        return pred

class MultiScaleGlobalPool(nn.Module):
    """TFT专用多尺度全局池化层（解决信息量不足问题）"""
    def __init__(self, feature_dim, seq_len, num_scales=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_scales = num_scales  # 分3个时间尺度（短/中/长）
        self.feature_dim = feature_dim
        
        # 1. 分尺度池化（比如seq_len=40，按10/20/40步长分块）
        self.scale_steps = [seq_len]
        cur_len = seq_len
        for _ in range(num_scales-1):
            cur_len = cur_len // 2
            self.scale_steps.append(cur_len)
        # 2. 注意力融合多尺度特征（核心：让模型自主选择重要尺度）
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4, batch_first=True)
        # 3. 残差连接：保留原始时序特征的关键信息
        self.residual_fc = nn.Linear(seq_len * feature_dim, feature_dim)

    def forward(self, x):
        # x: [batch_size, node_num,seq_len, feature_dim] 原始历史特征
        x = x.reshape(x.shape[0]*x.shape[1],x.shape[2],-1)
        batch_size = x.shape[0]
        
        # 步骤1：多尺度池化，保留不同时间粒度的信息
        scale_features = []
        for step in self.scale_steps:
            # 按step分块池化（比如step=24：每24个时间步做一次平均池化）
            if step == 0:
                continue
            num_chunks = self.seq_len // step
            chunked_x = x.reshape(batch_size, num_chunks, step, self.feature_dim)
            scale_pool = torch.mean(chunked_x, dim=2)  # [batch, num_chunks, feature_dim]
            scale_features.append(scale_pool)
        
        # 步骤2：注意力融合多尺度特征（替代单一全局池化）
        concat_scales = torch.cat(scale_features, dim=1)  # [batch, total_chunks, feature_dim]
        attn_out, _ = self.attention(concat_scales, concat_scales, concat_scales)
        global_pool = torch.mean(attn_out, dim=1)  # [batch, feature_dim] 最终1维特征
        
        # 步骤3：残差连接补充细节信息（避免丢失原始时序细节）
        flat_x = x.reshape(batch_size, -1)  # [batch, seq_len*feature_dim]
        residual = self.residual_fc(flat_x)  # 降维到feature_dim
        final_out = global_pool # + 0.1 * residual  # 残差缩放，避免主导
        
        return final_out

class FeatureScaleBalancer(nn.Module):
    """静态/动态特征尺度平衡层"""
    def __init__(self, static_dim, temporal_dim, eps=1e-6):
        super().__init__()
        # 可学习的缩放因子（初始让静态/动态特征尺度相当）
        self.static_scale = nn.Parameter(torch.ones(1, static_dim) * 0.3)  # 主动降低静态特征权重
        self.temporal_scale = nn.Parameter(torch.ones(1, temporal_dim))
        self.eps = eps

    def forward(self, static_emb, temporal_feat):
        # static_emb: [batch, static_dim] 静态Embedding
        # temporal_feat: [batch, temporal_dim] 时序特征
        
        # 1. 标准化：让两类特征均值为0，方差为1
        static_emb = (static_emb - static_emb.mean(dim=0, keepdim=True)) / (static_emb.std(dim=0, keepdim=True) + self.eps)
        temporal_feat = (temporal_feat - temporal_feat.mean(dim=0, keepdim=True)) / (temporal_feat.std(dim=0, keepdim=True) + self.eps)
        
        # 2. 可学习缩放：平衡两者权重
        static_emb = static_emb * self.static_scale
        temporal_feat = temporal_feat * self.temporal_scale
        
        return static_emb, temporal_feat
           
class TFTWithFutureCovariatesEn(nn.Module):
    """带已知未来协变量+样本关联的TFT模型（无未来泄露）"""
    def __init__(
        self,
        static_num=0,          # 静态特征维度
        obs_dim=6,             # 历史观测特征维度
        fut_dim=3,             # 已知未来协变量维度（天气/节假日等）
        time_embed_dim=28,     # 时间特征嵌入维度（含节假日）
        hidden_dim=64,         # 隐藏层维度
        nhead=8,               # 注意力头数
        num_layers=2,          # Transformer层数
        dropout=0.1,
        seq_len=1,               # 历史数据步长
        pred_len=1,            # 预测步长
        sample_dim=3,          # 样本维度（站点/设备数）
        d_model=64,        # 注意力维度
        static_emb_dim=4,      # 离散特征嵌入维度
        static_cate_emb=None,  # 静态离散特征嵌入
        target_mode=0,
        device='cuda'
    ):
        super().__init__()
        self.device = device
        self.pred_len = pred_len
        self.static_dim = static_num * static_emb_dim
        self.obs_dim = obs_dim
        self.fut_dim = fut_dim
        self.sample_dim = sample_dim
        self.hidden_dim = hidden_dim
        self.target_mode = target_mode
        
        # 静态离散特征嵌入初始化
        self.static_embed_layers = nn.ParameterList()
        for key in static_cate_emb:
            num_classes = static_cate_emb[key]
            self.static_embed_layers.append(nn.Embedding(num_classes, static_emb_dim))
        cate_static_num = len(self.static_embed_layers)
        cont_static_num = static_num - cate_static_num            
        
        # 可学习的缩放因子（初始让静态/动态特征尺度相当）
        # self.static_balancer = FeatureScaleBalancer(hidden_dim, hidden_dim)
                
        # 1. 连续静态特征的全连接层
        emb_dim = cont_static_num*static_emb_dim

        self.obs_grn = GatedResidualNetwork(input_dim=obs_dim, 
                                                 hidden_dim=hidden_dim, 
                                                 output_dim=hidden_dim,
                                                 context_combine_dim=seq_len)
               
        # TFT 核心：用静态上下文调节时序特征的模型
        self.temporal_grn = GatedResidualNetwork(input_dim=hidden_dim, 
                                                 hidden_dim=hidden_dim, 
                                                 output_dim=hidden_dim,
                                                 context_combine_dim=seq_len)
        # 未来日历相关协变量的GRN
        self.fur_temporal_grn = GatedResidualNetwork(input_dim=hidden_dim, 
                                                 hidden_dim=hidden_dim, 
                                                 output_dim=hidden_dim,
                                                 context_combine_dim=seq_len)        
        # 历史日历相关协变量的GRN
        self.his_temporal_grn = GatedResidualNetwork(input_dim=hidden_dim, 
                                                 hidden_dim=hidden_dim, 
                                                 output_dim=d_model,
                                                 context_combine_dim=seq_len)                  
        # 全局池化多尺度
        self.pool_layer = MultiScaleGlobalPool(obs_dim, seq_len,num_scales=3)
        
        # 1. 样本维度交互模块（仅作用于历史观测特征）
        self.sample_cross_attn = SampleCrossAttention(
            feat_dim=obs_dim + time_embed_dim,  # 历史观测+时间嵌入
            num_heads=2,
            dropout=dropout
        )
        
        # 2. 静态特征处理
        if self.static_dim > 0:
            self.static_grn_hist = GatedResidualNetwork(static_num, hidden_dim, hidden_dim, dropout=dropout)
            self.static_context_hist = nn.Linear(hidden_dim, hidden_dim)
            # 未来阶段静态特征处理（新增）
            self.static_grn_fut = GatedResidualNetwork(static_num, hidden_dim, hidden_dim, dropout=dropout)
            self.static_context_fut = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.LayerNorm(hidden_dim))
        
        self.calendar_encoder = GatedResidualNetwork(input_dim=time_embed_dim, hidden_dim=hidden_dim,output_dim=hidden_dim,
                                                 context_combine_dim=seq_len)       
        # 3. 历史特征投影（观测+时间）
        self.obs_proj = nn.Linear(obs_dim + time_embed_dim, hidden_dim)
        self.static_norm = nn.LayerNorm(hidden_dim)
        self.obs_norm = nn.LayerNorm(hidden_dim)
        
        # 4. 未来协变量投影（✨ 修正：输入维度新增静态特征维度）
        self.fut_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.fut_single_proj = nn.Linear(time_embed_dim+hidden_dim, hidden_dim)
        
        # 5. 变量选择网络（仅对历史观测变量）
        self.var_selection = VariableSelectionNetwork(hidden_dim, obs_dim, hidden_dim, dropout)
        self.concat_line = nn.Linear(2*hidden_dim,hidden_dim)
        
        # 6. Transformer编码器（因果掩码，仅处理历史）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim*6,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,norm=encoder_norm)
        self.transformer_end = nn.Linear(d_model,hidden_dim)
        
        
    def get_dynamic_target_ratio(self,epoch,max_epochs, final_target=1.0,warmup_ratio=0.5):
        # 线性退火：前期偏向浅层，后期收敛到固定目标比例
        
        total_epochs = max_epochs
        if epoch < total_epochs * 0.2 or total_epochs==0:
            return warmup_ratio
        else:
            alpha = (epoch - total_epochs*0.2) / (total_epochs*0.8)
            return warmup_ratio * (1-alpha) + final_target * alpha
            
    def forward(self, static_feat=None, obs_feat=None, time_embed_hist=None,future_single_emb=None,current_epoch=0,max_epochs=0):
        """
        static_feat: [B, S, static_dim] - 静态特征
        obs_feat: [B, S, T, obs_dim] - 历史观测特征（仅过去）
        time_embed_hist: [B, S, T, time_embed_dim] - 历史时间嵌入
        time_embed_fut: [B, S, P, time_embed_dim] - 未来时间嵌入
        time_embed_fut_single: [B, S, time_embed_dim] - 未来单独指定日期阶段嵌入
        return: 
            pred: [B, S, P, obs_dim] - 预测结果
            var_weights: [B, S, T, obs_dim] - 变量权重
            sample_attn_weights: [B, sample_heads, S, S] - 样本间注意力权重
        """
        
        hist_summary,future_single_emb = self.forward_encode(static_feat, obs_feat, time_embed_hist, future_single_emb, current_epoch, max_epochs)
        
        return hist_summary,future_single_emb
    
    def forward_encode(self, static_feat=None, obs_feat=None, time_embed_hist=None,future_single_emb=None,current_epoch=0,max_epochs=0):    
        B, S, T, _ = obs_feat.shape  # B=batch, S=样本数, T=历史序列长度
        P = self.pred_len            # P=预测步长
        
        # 2.3 静态特征处理
        cate_static_num = len(self.static_embed_layers)
        
        if static_feat is not None and self.static_dim > 0:
            # 对离散特征，转换嵌入特征
            # static_feat_flat = static_feat.reshape(B*S, -1)  # [B*S, static_dim]
            # 历史阶段静态特征融入
            static_context_hist = self.static_grn_hist(static_feat)
            # static_context_hist = self.static_context_hist(static_feat_hist).unsqueeze(1)  # [B*S,1,hidden_dim]
            
            # ✨ 修正：未来阶段静态特征预处理
            static_feat_fut = self.static_grn_fut(static_feat)
            static_context_fut = static_feat_fut # self.static_context_fut(static_feat_fut).unsqueeze(1)  # [B*S,1,hidden_dim]
        else:
            static_context_hist = None
        
        time_embed_hist = self.calendar_encoder(time_embed_hist,context=static_context_hist,scale_ctx=0.1)
        
        if self.target_mode==5:
            scale_ctx=0.02
        else:
            scale_ctx=0.02
        obs_feat = self.obs_grn(obs_feat,context=static_context_hist,scale_ctx=scale_ctx)
        if PRINT_STD_FLAG:
            print("obs_proj std:{}".format(obs_feat.std()))          
        # 2.1 融合历史观测+时间嵌入+样本交互
        # obs_input = torch.cat([obs_feat, time_embed_hist], dim=-1)  # [B,S,T,F]
        if self.target_mode==5:
            scale_ctx=0.15
        else:
            scale_ctx=0.05       
        obs_input = self.his_temporal_grn(obs_feat,context=time_embed_hist,scale_ctx=scale_ctx,no_ctx_squeeze=True)
        
        # 2.2 展平样本维度：[B,S,T,F] → [B*S, T, F]
        obs_input = obs_input.reshape(B*S, T, -1)     
         
        # 2.5 变量选择
        # obs_proj, var_weights = self.var_selection(obs_input)  # [B*S, T, hidden_dim]
        
        # 2.6 Transformer编码（因果掩码，禁止看未来）--Cancel
        # causal_mask = generate_causal_mask(T, obs_proj.device)
        hist_encoded = self.transformer_encoder(obs_input, mask=None)  # [B*S, T, d_model]
        hist_encoded = self.transformer_end(hist_encoded) # [B*S, T, hidden_dim]
        if PRINT_STD_FLAG:
            print("hist_encoded std:{}".format(hist_encoded.std()))        
        # 2.7 取最后时间步的历史编码（历史信息总结）
        hist_summary = hist_encoded[:, -1, :]  # [B*S, hidden_dim]
        hist_summary = hist_summary.reshape([B,S,-1])
        
        return hist_summary,static_context_hist
    

class TFTWithFutureCovariatesDe(nn.Module):
    """带已知未来协变量+样本关联的TFT模型（无未来泄露）"""
    def __init__(
        self,
        static_num=0,          # 静态特征维度
        obs_dim=6,             # 历史观测特征维度
        fut_dim=3,             # 已知未来协变量维度（天气/节假日等）
        time_embed_dim=28,     # 时间特征嵌入维度（含节假日）
        hidden_dim=64,         # 隐藏层维度
        nhead=8,               # 注意力头数
        num_layers=2,          # Transformer层数
        dropout=0.1,
        seq_len=1,               # 历史数据步长
        pred_len=1,            # 预测步长
        sample_dim=3,          # 样本维度（站点/设备数）
        d_model=64,        # 注意力维度
        static_emb_dim=4,      # 离散特征嵌入维度
        static_cate_emb=None,  # 静态离散特征嵌入
        target_mode=0,
        device='cuda'
    ):

        super().__init__()
        
        self.calendar_encoder_fur = GatedResidualNetwork(input_dim=time_embed_dim, hidden_dim=hidden_dim,output_dim=hidden_dim,
                                                 context_combine_dim=1)          
        self.target_mode = target_mode
        # 分别定义完整预测序列的解码器，以及最终单独目标的解码器
        # self.seq_decoder = DecoderLayer(obs_dim=obs_dim, hidden_dim=hidden_dim,sample_dim=sample_dim,
        #              dropout=0.01, pred_len=pred_len)
        self.tar_decoder = DecoderLayer(obs_dim=obs_dim, hidden_dim=hidden_dim,sample_dim=sample_dim,
                     dropout=0.01, pred_len=1)     
             
        if self.target_mode==5:
            fur_scale = 0.02
        else:
            fur_scale = 0.018
        self.fur_scale = fur_scale
    
    def set_fur_scale(self,fur_scale):     
        self.fur_scale = fur_scale
           
    def forward(self, hist_summary,future_single_emb=None,static_context_hist=None):

        B, S, _ = hist_summary.shape
        hist_summary = hist_summary.reshape([B*S,-1])
        # 未来协变量投影
        future_single_emb = self.calendar_encoder_fur(future_single_emb.unsqueeze(-2),context=static_context_hist)
              
        if PRINT_STD_FLAG:
            print("hist_summary std:{}".format(hist_summary.std()))
            print("future_single_emb std:{}".format(future_single_emb.std()))
        
        pred_tar = self.tar_decoder(hist_summary,future_single_emb,fur_scale=self.fur_scale)        # [B*S*1, obs_dim]
        # pred_tar_tmp = pred_tar.reshape([-1,pred_tar.shape[-1]])
        
        if PRINT_STD_FLAG:
            print("pred_tar std:{}".format(pred_tar.std()))
        # # 4.4 变量权重恢复
        # var_weights = var_weights.reshape(B, S, T, self.obs_dim)  # [B, S, T, obs_dim]
        
        return pred_tar           
            
class ContinuousToDiscreteIndex(nn.Module):
    def __init__(self, num_indices=3, hidden_dim=8):
        super().__init__()
        self.num_indices = num_indices
        # 连续参考值 → 映射为离散索引的 logits
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 输入：1维连续参考值
            nn.ReLU(),
            nn.Linear(hidden_dim, num_indices)  # 输出：num_indices 个索引的分数
        )

    def forward(self, continuous_ref, tau=1.0):
        """
        continuous_ref: 连续参考值 [B, 1]
        tau: 温度系数（越小越接近硬索引）
        return: 可导的离散索引 one-hot / 软索引
        """
        logits = self.net(continuous_ref)
        # 核心：Gumbel-Softmax → 连续值可导选择离散索引
        soft_index = F.gumbel_softmax(logits, tau=tau, hard=False)
        return soft_index  # [B, num_indices] 每一行代表选中每个离散索引的概率
    
class AttScaleFeature(nn.Module):
    """按照指定尺度，实现特征处理"""

    def __init__(self,sample_dim,input_dim,seq_len=5,ins_arr=None,ins_trend_dict=None, hidden_dim=16, dropout=0.1,num_indices=3,target_mode=0,device=None):
        super().__init__()
        self.sample_dim = sample_dim
        self.scale_arr = torch.Tensor(ins_arr).long().to(device)
        self.ins_trend_dict = ins_trend_dict
        self.num_indices = num_indices
        self.seq_len = seq_len
        self.target_mode = target_mode
        
        # TOP值选取网络
        sample_dim_inner = ins_arr.shape[0]
        # ins_layer_inner = LinelessLayer(sample_dim_inner*input_dim,sample_dim_inner,hidden_size=hidden_dim,
        #                     layer_norm=False,batch_norm=False,dropout=dropout)
        # nn.init.xavier_normal_(ins_layer_inner.linear_hidden.weight, gain=1.2)
        # nn.init.xavier_normal_(ins_layer_inner.linear_output.weight, gain=1.2)
        # nn.init.zeros_(ins_layer_inner.linear_hidden.bias)
        # nn.init.zeros_(ins_layer_inner.linear_output.bias)        
        
        # 分支趋势计算网络
        trend_logits_layer = {}
        ins_layer = {}
        for key in ins_trend_dict.keys():  
            sample_dim_inner = ins_trend_dict[key].shape[0]
            # 每个大类内包含1个或多个小类，小类内包含品种，照此创建mlp网络
            ins_layer_inner = LinelessLayer(sample_dim_inner*input_dim,sample_dim_inner,hidden_size=hidden_dim,
                            layer_norm=True,batch_norm=False,dropout=dropout)    
            ins_layer[key] = ins_layer_inner           
            trend_logits_layer_inner = LinelessLayer(sample_dim_inner*input_dim,1,hidden_size=input_dim,
                                layer_norm=False,batch_norm=True,track_running_stats=True,dropout=dropout)      
            trend_logits_layer[key] = trend_logits_layer_inner
        self.ins_layer = nn.ModuleDict(ins_layer)
        self.trend_logits_layer = nn.ModuleDict(trend_logits_layer)
        # 整体趋势计算网络
        trend_layer_inner = []
        # for i in range(len(ins_trend_arr)):  
        #     sample_dim_inner = ins_trend_arr[i].shape[0]
        #     trend_layer_inner = LinelessLayer(sample_dim_inner*input_dim,1,hidden_size=input_dim,
        #                         layer_norm=True,batch_norm=False,elementwise_affine=True,dropout=dropout)      
        #     trend_layer_inner.append(trend_layer_inner)        
        # self.trend_layer = nn.ModuleList(trend_layer_inner)          
        
        
    def forward(self, x):
        # x: (batch_size, 品种S, 特征input_dim)
        batch_size, S, _ = x.shape
        
        x_part = x[:,self.scale_arr,:].reshape(batch_size,-1)
        output = []
        # 整体趋势网络计算
        output2index_trend = {}
        for key in self.ins_trend_dict.keys(): 
            ins = self.ins_trend_dict[key]
            x_l_part = x[:,ins].reshape(batch_size,-1)
            output_single = self.ins_layer[key](x_l_part).squeeze(-1) 
            output.append(output_single)
            output_trend = self.trend_logits_layer[key](x_l_part).squeeze(-1) 
            output2index_trend[key] = output_trend
        output = torch.cat(output,dim=-1)
        
        return output,output_trend,output2index_trend
                        
class SparseGateFeatureTopK(nn.Module):
    """综合TOPK选取"""
    
    def __init__(self,sample_dim,input_dim,seq_len=5,k=3, hidden_dim=16,num_heads=4, dropout=0.1,mlp_init_scale=1.5,
                 target_mode=0,scales_dict=None,device=None):
        super().__init__()
        self.target_mode = target_mode
        self.sample_dim = sample_dim
        self.k = k
        self.num_heads = num_heads
        
        # 分别按照夜盘类别、行业类别、保证金范围生成不同注意力尺度的网络计算
        self.top_global_layer = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_dim,
                                    layer_norm=False,batch_norm=False,dropout=dropout)
        scales_layer = []    
        scales_arr = concat_scale_arr(scales_dict)
        scales_trend_arr = emb_scale_arr(scales_dict)
        # scales_trend_arr = scales_trend_arr[0]
        self.scales_arr = scales_arr
        self.scales_dict = scales_dict        
        # 类别内部品种的mlp
        for i,item in enumerate(scales_arr):
            trend_arr = scales_trend_arr[item['p0']]
            instruments_dict = {key:torch.Tensor(trend_arr[key]['instruments']).to(device).long() for key in trend_arr.keys()}
            scales_layer.append(AttScaleFeature(sample_dim,input_dim,seq_len=seq_len,ins_arr=item['instruments'],target_mode=target_mode,
                                                ins_trend_dict=instruments_dict, dropout=dropout,device=device))
        self.scales_layer = nn.ModuleList(scales_layer)
        p1_count = scales_dict.shape[0]
        trend_layer = [] 
        # 小类的内部mlp及整体连接
        for i,item in scales_dict.iterrows():
            inner_sample_dim = item['instruments'].shape[0]
            branch_trend_combine_layer = LinelessLayer(inner_sample_dim*input_dim,1,hidden_size=hidden_dim,
                                    layer_norm=False,batch_norm=True,dropout=0.3)     
            nn.init.xavier_normal_(branch_trend_combine_layer.linear_hidden.weight, gain=mlp_init_scale)
            nn.init.xavier_normal_(branch_trend_combine_layer.linear_output.weight, gain=mlp_init_scale)
            nn.init.zeros_(branch_trend_combine_layer.linear_hidden.bias)
            nn.init.zeros_(branch_trend_combine_layer.linear_output.bias)
            trend_layer.append(branch_trend_combine_layer)
        self.branch_trend_combine_layer = nn.ModuleList(trend_layer)
        # self.branch_trend_combine_layer_total = nn.Sequential(nn.Linear(scales_dict.shape[0],scales_dict.shape[0]),nn.BatchNorm1d(scales_dict.shape[0]))     
        self.branch_trend_combine_layer_total = LinelessLayer(scales_dict.shape[0],scales_dict.shape[0],hidden_size=hidden_dim,relu=True,
                                    layer_norm=False,batch_norm=True,dropout=0.3)  
        # 大类的mlp
        self.branch_trend_combine_layer_main = LinelessLayer(scales_dict.shape[0],len(scales_arr),hidden_size=hidden_dim,relu=True,
                                    layer_norm=False,batch_norm=True,dropout=0.3)   
        # nn.init.xavier_normal_(self.branch_trend_combine_layer_total.linear_hidden.weight, gain=mlp_init_scale)
        # nn.init.zeros_(self.branch_trend_combine_layer_total.linear_hidden.bias)
        nn.init.xavier_normal_(self.branch_trend_combine_layer_total.linear_output.weight, gain=mlp_init_scale)
        nn.init.xavier_normal_(self.branch_trend_combine_layer_total.linear_hidden.weight, gain=mlp_init_scale)
        nn.init.xavier_normal_(self.branch_trend_combine_layer_total.linear_hidden_redu.weight, gain=mlp_init_scale)
        nn.init.zeros_(self.branch_trend_combine_layer_total.linear_output.bias)        
        nn.init.zeros_(self.branch_trend_combine_layer_total.linear_hidden.bias) 
        nn.init.zeros_(self.branch_trend_combine_layer_total.linear_hidden_redu.bias) 
        
        nn.init.xavier_normal_(self.branch_trend_combine_layer_main.linear_output.weight, gain=mlp_init_scale)
        nn.init.xavier_normal_(self.branch_trend_combine_layer_main.linear_hidden.weight, gain=mlp_init_scale)
        nn.init.xavier_normal_(self.branch_trend_combine_layer_main.linear_hidden_redu.weight, gain=mlp_init_scale)
        nn.init.zeros_(self.branch_trend_combine_layer_main.linear_output.bias)        
        nn.init.zeros_(self.branch_trend_combine_layer_main.linear_hidden.bias) 
        nn.init.zeros_(self.branch_trend_combine_layer_main.linear_hidden_redu.bias) 
                        
    def forward_combine(self, x):
        # x: (batch_size, 品种S, 特征input_dim)
        batch_size, S, input_dim = x.shape
        features_list = {}
        trend_logits_list = {}
        # 分别根据不同的业务尺度，生成1维度特征
        g_features = self.top_global_layer(x.reshape(batch_size,-1))  
        features_list['global_feature'] = g_features
        # 每个小类内部的品种计算
        for i,layer in enumerate(self.scales_layer):
            scale_features,_,trend_index_logits = layer(x)  
            scale_def = self.scales_arr[i]
            key = scale_def['p']
            # 合并主体特征和分尺度特征
            features_list[key] = scale_features
            if PRINT_STD_FLAG:
                print("features_list/{} std:{}".format(key,scale_features.std()))               
            trend_logits_list[key] = trend_index_logits
        # Total Trend
        # trend_logits_list['total'] = {'total':self.total_trend_layer(x.reshape([batch_size,-1])).squeeze(-1)}
        # 小类计算
        trend_list = []
        for i,item in self.scales_dict.iterrows():
            ins = torch.Tensor(item['instruments']).to(x.device).long()
            x_part = x[:,ins,:]
            cate_data = self.branch_trend_combine_layer[i](x_part.reshape(batch_size,-1),redu=True).squeeze(-1)
            trend_list.append(cate_data)     
        trend_list = torch.stack(trend_list).transpose(1,0)    
        if PRINT_STD_FLAG:
            print("trend_list std:{}".format(trend_list.std()))
        # 总体小类整合计算
        trend_list = self.branch_trend_combine_layer_total(trend_list,redu=False)
        if PRINT_STD_FLAG:
            print("trend_list after std:{}".format(trend_list.std()))        
        trend_list_total = trend_list
        # 大类整合计算
        trend_list_main = self.branch_trend_combine_layer_main(trend_list,redu=False)
        if PRINT_STD_FLAG:
            print("trend_list_main std:{}".format(trend_list_main.std()))           
        return trend_logits_list,features_list,trend_list_total,trend_list_main
    
    def forward(self, x,output_index=2):
        output = self.forward_combine(x)
        if output_index==1:
            output = output[output_index]
            output_combine = []
            for item in self.scales_arr :
                output_combine.append(output[item['p0']])
            output = torch.concat(output_combine,dim=-1)
        else:
            output = output[output_index]
        return output

class UnionTransCombine(nn.Module):
    """整合后的完整模型"""

    def __init__(
        self,
        max_epochs=0,
        obs_dim=6,             # 历史观测特征维度（要预测的变量）
        fut_dim=3,             # 已知未来协变量维度（天气/节假日等）
        time_embed_dim=2,
        hidden_dim=64,         # 隐藏层维度
        d_model=64,
        nhead=8,               # 注意力头数
        num_layers=2,          # Transformer层数
        dropout=0.1,
        seq_len=1,             # 历史数据步长
        pred_len=1,            # 预测步长
        sample_dim=3,          # 样本维度（站点/设备数）
        sample_heads=4,        # 样本间注意力头数
        target_feat_dim=2,
        hidden_size=16,
        static_num=4,         # 静态特征维度
        static_emb_dim=4,      # 离散特征嵌入维度
        static_cate_emb=None, # 静态离散特征嵌入
        top_num=3,            # topk数量
        scales_arr=None,
        target_mode=0,
        trans_scale=0.4,
        device='cuda'
    ):
        super().__init__()
        self.max_epochs = max_epochs
        
        dropout_trans = dropout
        self.time_embed_dim = time_embed_dim
        self.trans_model_encoder = TFTWithFutureCovariatesEn(
                static_num=static_num,
                obs_dim=obs_dim,
                fut_dim=fut_dim,
                time_embed_dim=self.time_embed_dim,
                hidden_dim=hidden_dim,
                nhead=nhead,
                d_model=d_model,
                num_layers=num_layers,
                dropout=dropout_trans,
                seq_len=seq_len,
                pred_len=pred_len,
                sample_dim=sample_dim,
                static_cate_emb=static_cate_emb,
                static_emb_dim=static_emb_dim,
                target_mode=target_mode,
                device=device,
            )
        self.trans_model_decoder = TFTWithFutureCovariatesDe(
                static_num=static_num,
                obs_dim=obs_dim,
                fut_dim=fut_dim,
                time_embed_dim=self.time_embed_dim,
                hidden_dim=hidden_dim,
                nhead=nhead,
                d_model=d_model,
                num_layers=num_layers,
                dropout=dropout_trans,
                seq_len=seq_len,
                pred_len=pred_len,
                sample_dim=sample_dim,
                static_cate_emb=static_cate_emb,
                static_emb_dim=static_emb_dim,
                target_mode=target_mode,
                device=device,
            )        
        self.pred_len = pred_len         
        self.sample_dim = sample_dim
        self.target_feat_dim = target_feat_dim  
        self.trans_scale = trans_scale
        self.target_mode = target_mode
        # 整合输出网络
        # self.ins_layer = nn.ParameterList([LinelessLayer(sample_dim*obs_dim*pred_len,sample_dim,hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3).double() for _ in range(self.target_feat_dim)])
        # 指数整合输出网络       
        self.index_combine_layer = LinelessLayer(sample_dim*obs_dim*pred_len,pred_len)     
        # TOPK选择器网络
        self.top_selector = nn.ParameterList([SparseGateFeatureTopK(sample_dim,obs_dim, k=top_num, seq_len=pred_len,target_mode=target_mode,
                        hidden_dim=hidden_size,num_heads=4, dropout=dropout,scales_dict=scales_arr,device=device) for _ in range(self.target_feat_dim)])

        ############# 中间变量调试 #############
        self.features = {}
    
    def transform_inner(self, batch_data):        
        return self.time_encoder.transform_inner(batch_data)

    def get_dynamic_target_ratio(self,epoch,max_epochs, final_target=1.2,warmup_ratio=0.4):
        # 线性退火：前期偏向浅层，后期收敛到固定目标比例
        
        total_epochs = max_epochs
        if epoch < total_epochs * 0.2 or total_epochs==0:
            return warmup_ratio
        else:
            alpha = (epoch - total_epochs*0.2) / (total_epochs*0.8)
            return warmup_ratio * (1-alpha) + final_target * alpha
          
    def forward(
        self,static_covs,past_convs_item, his_future_emb,future_single_emb,current_epoch=0,max_epochs=180,
    ):    
        
        # 基础模型的向前传播
        hist_summary,static_context_hist = self.trans_model_encoder(static_covs,past_convs_item, his_future_emb,future_single_emb,current_epoch=current_epoch,max_epochs=max_epochs)   
        pred_tar = self.trans_model_decoder(hist_summary,future_single_emb=future_single_emb,static_context_hist=static_context_hist)  
        if self.target_mode==5:
            tar_scale = 2.0 
        else:
            tar_scale = 1.8
        pred_tar = pred_tar * tar_scale
        
        trend_logits_combine = []
        cls_out_combine = []    
        trend_list_total = []
        trend_list_main_combine = []
        # 品种间比较目标的网络输出
        for i in range(self.target_feat_dim):
            # 主要比较目标输出
            trend_logits_list,features_list,trend_list,trend_list_main = self.top_selector[i].forward_combine(pred_tar.reshape(pred_tar.shape[0],self.sample_dim,-1))
            cls_out_combine.append(features_list)
            # 整体指数预测的网络输出
            trend_list_total.append(trend_list)
            trend_logits_combine.append(trend_logits_list)
            trend_list_main_combine.append(trend_list_main)
        
        return trend_logits_combine,cls_out_combine,trend_list_total,trend_list_main_combine,(tar_scale)


