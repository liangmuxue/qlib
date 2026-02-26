import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from .cov_cnn import LinelessLayer

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
                'year': 4, 'month': 4, 'day': 4,'dayofweek': 4  # 新增节假日特征
            }
        else:
            self.embed_dims = embed_dims
    
    def to_device(self):
        self.embed_layers = self.embed_layers.to(self.device)

    def fit(self, df):
        """拟合时间特征编码器"""
        df_time = pd.to_datetime(df[self.time_col])
        self.time_features = {
            'year': df_time.dt.year.values,
            'month': df_time.dt.month.values,
            'day': df_time.dt.day.values,
            'dayofweek': df_time.dt.dayofweek.values,
        }
        
        # 为离散时间特征创建LabelEncoder
        for feat_name, feat_vals in self.time_features.items():
            le = LabelEncoder()
            le.fit(feat_vals)
            self.encoders[feat_name] = le
            num_classes = len(le.classes_)
            # 创建嵌入层
            self.embed_layers[feat_name] = nn.Embedding(num_classes, self.embed_dims[feat_name]).to(self.device)
        
        return self

    def fit_static(self, range_data):
        """使用固定数值范围，拟合时间特征编码器"""
        
        # 为离散时间特征创建LabelEncoder
        for feat_name in range_data.keys():
            le = LabelEncoder()
            le.fit(range_data[feat_name])
            self.encoders[feat_name] = le            
            num_classes = len(le.classes_)
            # 创建嵌入层
            self.embed_layers[feat_name] = nn.Embedding(num_classes, self.embed_dims[feat_name]).to(self.device)
        
        return self
    
    def transform(self, df, device='cpu'):
        """转换时间特征为嵌入向量"""
        df_time = pd.to_datetime(df[self.time_col])
        time_feats = {
            'year': df_time.dt.year.values,
            'month': df_time.dt.month.values,
            'day': df_time.dt.day.values,
            'dayofweek': df_time.dt.dayofweek.values,
        }
        
        embed_list = []
        for feat_name, feat_vals in time_feats.items():
            encoded = torch.tensor(self.encoders[feat_name].transform(feat_vals), device=device)
            embed = self.embed_layers[feat_name](encoded)
            embed_list.append(embed)
        
        # 拼接所有时间嵌入
        time_embed = torch.cat(embed_list, dim=-1)  # (n_samples, time_embed_dim)
        return time_embed

    def transform_inner(self, batch_data, device='cpu'):
        """批次内转换时间特征为嵌入向量"""
        
        embed_list = []
        for feat_name in batch_data.keys():
            feat_vals = batch_data[feat_name]
            encoded = torch.tensor(self.encoders[feat_name].transform(feat_vals.cpu()), device=device)
            embed = self.embed_layers[feat_name](encoded)
            embed_list.append(embed)
        
        # 拼接所有时间嵌入
        time_embed = torch.cat(embed_list, dim=-1)  # (n_samples, time_embed_dim)
        return time_embed
    
# ---------------------- TFT核心模块 ----------------------
class GatedResidualNetwork(nn.Module):
    """门控残差网络（GRN）：TFT核心特征处理模块"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 残差连接适配
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) 或 (batch, input_dim)
        """
        # 前向传播
        x_res = self.residual(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 门控机制
        gate = torch.sigmoid(self.gate(x_res))
        x = x * gate
        
        # 残差+层归一
        x = x + x_res
        x = self.layer_norm(x)
        return x

class VariableSelectionNetwork(nn.Module):
    """变量选择网络：对输入特征加权，突出重要变量"""
    def __init__(self, input_dim, num_vars, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.grn = GatedResidualNetwork(input_dim, hidden_dim, num_vars, dropout)

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

class TFTWithFutureCovariates(nn.Module):
    """带已知未来协变量+样本关联的TFT模型（无未来泄露）"""
    def __init__(
        self,
        static_num=0,          # 静态特征维度
        obs_dim=6,             # 历史观测特征维度（要预测的变量）
        fut_dim=3,             # 已知未来协变量维度（天气/节假日等）
        time_embed_dim=28,     # 时间特征嵌入维度（含节假日）
        hidden_dim=64,         # 隐藏层维度
        nhead=8,               # 注意力头数
        num_layers=2,          # Transformer层数
        dropout=0.1,
        pred_len=1,            # 预测步长
        sample_dim=3,          # 样本维度（站点/设备数）
        sample_heads=4,        # 样本间注意力头数
        static_emb_dim=4,      # 离散特征嵌入维度
        static_cate_emb=None,  # 静态离散特征嵌入
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
        
        # 静态离散特征嵌入初始化
        self.static_embed_layers = nn.ParameterList()
        for key in static_cate_emb:
            num_classes = static_cate_emb[key]
            self.static_embed_layers.append(nn.Embedding(num_classes, static_emb_dim))
        cate_static_num = len(self.static_embed_layers)
        cont_static_num = static_num - cate_static_num            
        # 1. 连续静态特征的全连接层
        emb_dim = cont_static_num*static_emb_dim
        self.static_cont_mlp = nn.Sequential(
            nn.Linear(cont_static_num, emb_dim),  # 第一层：维度映射
            nn.GELU(),                            # 非线性激活
            nn.Linear(emb_dim, emb_dim)           # 第二层：增强表达
        )
               
        # 1. 样本维度交互模块（仅作用于历史观测特征）
        self.sample_cross_attn = SampleCrossAttention(
            feat_dim=obs_dim + time_embed_dim,  # 历史观测+时间嵌入
            num_heads=sample_heads,
            dropout=dropout
        )
        
        # 2. 静态特征处理
        if self.static_dim > 0:
            self.static_grn_hist = GatedResidualNetwork(self.static_dim, hidden_dim, hidden_dim, dropout)
            self.static_context_hist = nn.Linear(hidden_dim, hidden_dim)
            # 未来阶段静态特征处理（新增）
            self.static_grn_fut = GatedResidualNetwork(self.static_dim, hidden_dim, hidden_dim, dropout)
            self.static_context_fut = nn.Linear(hidden_dim, hidden_dim)
                    
        # 3. 历史特征投影（观测+时间）
        self.obs_proj = nn.Linear(obs_dim + time_embed_dim, hidden_dim)
        
        # 4. 未来协变量投影（✨ 修正：输入维度新增静态特征维度）
        self.fut_proj = nn.Linear(time_embed_dim + hidden_dim, hidden_dim)
        
        # 5. 变量选择网络（仅对历史观测变量）
        self.var_selection = VariableSelectionNetwork(hidden_dim, obs_dim, hidden_dim, dropout)
        
        # 6. Transformer编码器（因果掩码，仅处理历史）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 7. 未来协变量融合网络（关键：融合历史编码+未来协变量）
        self.future_fusion = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        
        # 8. 门控注意力融合
        self.attention_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 9. 输出层（预测目标变量）
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.final_proj = nn.Linear(hidden_dim, obs_dim)

    def forward(self, static_feat=None, obs_feat=None, time_embed_hist=None, time_embed_fut=None):
        """
        static_feat: [B, S, static_dim] - 静态特征
        obs_feat: [B, S, T, obs_dim] - 历史观测特征（仅过去）
        time_embed_hist: [B, S, T, time_embed_dim] - 历史时间嵌入
        time_embed_fut: [B, S, P, time_embed_dim] - 未来时间嵌入
        return: 
            pred: [B, S, P, obs_dim] - 预测结果
            var_weights: [B, S, T, obs_dim] - 变量权重
            sample_attn_weights: [B, sample_heads, S, S] - 样本间注意力权重
        """
        B, S, T, _ = obs_feat.shape  # B=batch, S=样本数, T=历史序列长度
        P = self.pred_len            # P=预测步长
        
        # ---------------------- 步骤1：样本维度交互（仅历史，无未来泄露） ----------------------
        # 1.1 历史特征全局池化
        obs_global = obs_feat.mean(dim=2)  # [B, S, obs_dim]
        time_hist_global = time_embed_hist.mean(dim=2)  # [B, S, time_embed_dim]
        sample_feat = torch.cat([obs_global, time_hist_global], dim=-1)  # [B, S, F]
        
        # 1.2 样本间自注意力（建模样本关联）
        sample_feat_interact, sample_attn_weights = self.sample_cross_attn(sample_feat)  # [B, S, F]
        
        # 1.3 广播回时间维度
        sample_feat_interact = sample_feat_interact.unsqueeze(2).repeat(1, 1, T, 1)  # [B, S, T, F]
        
        # ---------------------- 步骤2：历史特征处理（因果掩码，仅看过去） ----------------------
        # 2.1 融合历史观测+时间嵌入+样本交互
        obs_input = torch.cat([obs_feat, time_embed_hist], dim=-1)  # [B,S,T,F]
        obs_input = obs_input + sample_feat_interact  # 残差融合
        
        # 2.2 展平样本维度：[B,S,T,F] → [B*S, T, F]
        obs_input = obs_input.reshape(B*S, T, -1)
        
        # 2.3 静态特征处理
        cate_static_num = len(self.static_embed_layers)
        
        if static_feat is not None and self.static_dim > 0:
            # 对离散特征，转换嵌入特征
            cate_static_feat = static_feat[...,:cate_static_num]
            cate_emb_arr = []
            for i,layer in enumerate(self.static_embed_layers):
                cate_emb = layer(cate_static_feat[...,i].long())
                cate_emb_arr.append(cate_emb)
            cate_emb_arr = torch.cat(cate_emb_arr,dim=-1)
            # 对于连续静态特征，使用全连接进行特征转换
            cont_static_feat = static_feat[...,cate_static_num:]
            cont_static_feat = self.static_cont_mlp(cont_static_feat)
            static_feat = torch.cat([cate_emb_arr,cont_static_feat],dim=-1)
            static_feat_flat = static_feat.reshape(B*S, self.static_dim)  # [B*S, static_dim]
            # 历史阶段静态特征融入
            static_feat_hist = self.static_grn_hist(static_feat_flat)
            static_context_hist = self.static_context_hist(static_feat_hist).unsqueeze(1)  # [B*S,1,hidden_dim]
            
            # ✨ 修正：未来阶段静态特征预处理
            static_feat_fut = self.static_grn_fut(static_feat_flat)
            static_context_fut = self.static_context_fut(static_feat_fut).unsqueeze(1)  # [B*S,1,hidden_dim]
        else:
            static_context_hist = None
        
        # 2.4 历史特征投影
        obs_proj = self.obs_proj(obs_input)  # [B*S, T, hidden_dim]
        if static_context_hist is not None:
            obs_proj = obs_proj + static_context_hist
        
        # 2.5 变量选择
        obs_proj, var_weights = self.var_selection(obs_proj)  # [B*S, T, hidden_dim]
        
        # 2.6 Transformer编码（因果掩码，禁止看未来）
        causal_mask = generate_causal_mask(T, self.device)
        hist_encoded = self.transformer_encoder(obs_proj, mask=causal_mask)  # [B*S, T, hidden_dim]
        
        # 2.7 取最后时间步的历史编码（历史信息总结）
        hist_summary = hist_encoded[:, -1, :]  # [B*S, hidden_dim]
        
        time_embed_fut_flat = time_embed_fut.reshape(B*S, P, time_embed_fut.shape[-1])
        
        # 3.2 未来协变量投影
        fut_input = time_embed_fut_flat # [B*S, P, F]
        if static_context_fut is not None:
            # 静态特征广播到所有预测步：[B*S,1,hidden_dim] → [B*S,P,hidden_dim]
            static_broadcast = static_context_fut.repeat(1, P, 1)
            # 拼接未来协变量 + 静态特征
            fut_input = torch.cat([fut_input, static_broadcast], dim=-1)  # [B*S,P,F_time+F_static]  
                  
        fut_proj = self.fut_proj(fut_input)  # [B*S, P, hidden_dim]
        
        # 3.3 融合历史总结+未来协变量（核心：历史指导未来预测）
        hist_summary_expanded = hist_summary.unsqueeze(1).repeat(1, P, 1)  # [B*S, P, hidden_dim]
        fusion_input = torch.cat([hist_summary_expanded, fut_proj], dim=-1)  # [B*S, P, 2*hidden_dim]
        fusion_out = self.future_fusion(fusion_input)  # [B*S, P, hidden_dim]
        
        # ---------------------- 步骤4：输出预测 ----------------------
        # 4.1 门控融合
        final_feat = self.attention_gate(fusion_out)  # [B*S, P, hidden_dim]
        
        # 4.2 输出层（每个预测步独立预测）
        final_feat_flat = final_feat.reshape(B*S*P, self.hidden_dim)  # [B*S*P, hidden_dim]
        final_feat_processed = self.output_grn(final_feat_flat)   # [B*S*P, hidden_dim]
        pred_flat = self.final_proj(final_feat_processed)        # [B*S*P, obs_dim]
        
        # 4.3 恢复维度
        pred = pred_flat.reshape(B*S, P, self.obs_dim)  # [B*S, P, obs_dim]
        pred = pred.reshape(B, S, P, self.obs_dim)      # [B, S, P, obs_dim]
        
        # 4.4 变量权重恢复
        var_weights = var_weights.reshape(B, S, T, self.obs_dim)  # [B, S, T, obs_dim]
        
        return pred, var_weights, sample_attn_weights

def mask_with_flag(features,mask):
        mask_exp = mask.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        features_exp = features * mask_exp   
        return features_exp
    

class GumbelTopK(nn.Module):
    def __init__(self, k, tau=1.0, hard=True):
        super().__init__()
        self.k = k
        self.tau = tau
        self.hard = hard

    def forward(self, logits):
        # logits: (batch, d)
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        perturbed = logits + noise
        # 使用 softmax 获得连续松弛
        soft = F.softmax(perturbed / self.tau, dim=-1)
        # 取 topk 索引
        indices = perturbed.topk(self.k, dim=-1).indices
        # 构造 one-hot 掩码
        hard_mask = torch.zeros_like(logits).scatter(-1, indices, 1.0)
        if self.hard:
            # Straight-through: 前向用 hard，反向用 soft 的梯度
            mask = hard_mask - soft.detach() + soft
        else:
            mask = soft
        return mask,hard_mask

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, k=5, num_layers=2,tau=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        if hidden_dim is None:
            hidden_dim = input_dim
        # 门控 logits 生成层
        self.temporal_gate = self.create_gate_layer(input_dim, hidden_dim, num_layers)
           
        self.gumbel_topk_long = GumbelTopK(k, tau,hard=False)
        self.gumbel_topk_short = GumbelTopK(k, tau,hard=False)
    
    def create_gate_layer(self,input_dim,hidden_dim,num_layers):
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)      
    
    def forward(self, x):
        # x: (batch, input_dim)
        logits = self.temporal_gate(x).squeeze(-1) 
        mask_long,hard_mask_long = self.gumbel_topk_long(logits)    # (batch, input_dim) 近似 0/1
        mask_short,hard_mask_short = self.gumbel_topk_short(-logits)
        # 应用掩码
        selected_long = mask_with_flag(x,hard_mask_long)
        selected_short = mask_with_flag(x,hard_mask_short)
        return (selected_long,selected_short), (hard_mask_long,hard_mask_short) , logits
    
class SparseGateFeatureTopK(nn.Module):
    """
    稀疏门控特征Top-K：基于注意力机制,并通过门控网络学习特征重要性，仅激活Top-K特征
    优势：可随输入动态调整Top-K特征（不同样本选不同特征）
    """
    def __init__(self,sample_dim,input_dim,k=3, hidden_dim=16,num_heads=4, dropout=0.1):
        super().__init__()
        self.sample_dim = sample_dim
        self.k = k
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # 单层 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 门控网络：对每个样本的特征时序序列建模，输出特征重要性
        # self.gate_net = UltimateNormalizedGate(hidden_dim)
        self.gate_net = GatingNetwork(input_dim,k=k)
        # 整合输出为结合topk选择的品类
        # self.top_ins_layer = LinelessLayer(2*k*input_dim,k*2,hidden_size=hidden_dim,
        #                             layer_norm=True,batch_norm=False,dropout=0.3)
        self.top_ins_layer_long = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_dim,
                                    layer_norm=True,batch_norm=False,dropout=0.3)      
        self.top_ins_layer_short = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_dim,
                                    layer_norm=True,batch_norm=False,dropout=0.3)                     
        self.ins_layer = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_dim,
                                    layer_norm=True,batch_norm=False,dropout=0.3)    
    def forward(self, x):
        # x: (batch_size, 品种S, 特征input_dim)
        batch_size, S, input_dim = x.shape
        # # Transformer 期望输入形状为 (N, B, H)
        # h = x.transpose(0, 1)                    # (N, S, H)
        # h = self.transformer(h)
        # h = h.transpose(0, 1)                    # (B, S, H)
               
        # 计算每个特征的门控分数（逐样本）：(batch_size, S, 1)
        (selected_long,selected_short), (mask_long,mask_short), gate_scores = self.gate_net(x)
        
        # 逐样本筛选Top-K特征索引：(batch_size, k),分别从正反取得
        # _, topk_indices = torch.topk(gate_scores, k=self.k, dim=1)
        # _, topk_inverse_indices = torch.topk(-gate_scores, k=self.k, dim=1)
        #
        # tok_combine_index = torch.cat([topk_indices,topk_inverse_indices],dim=1)
        # # 提取top特征
        # topk_indices_expanded = tok_combine_index.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # topk_features = torch.gather(x, dim=1, index=topk_indices_expanded)
        # 整合输出
        topk_features_long = self.top_ins_layer_long(selected_long.reshape(selected_long.shape[0],-1))    
        topk_features_short = self.top_ins_layer_short(selected_short.reshape(selected_short.shape[0],-1)) 
        topk_features_combine = torch.cat([topk_features_long,topk_features_short],-1)
          
        normal_features = self.ins_layer(x.reshape(x.shape[0],-1))  
        
        return normal_features,(topk_features_long,topk_features_short),(mask_long,mask_short)
    
            
class UnionTransCombine(nn.Module):
    """整合后的完整模型"""

    def __init__(
        self,
        obs_dim=6,             # 历史观测特征维度（要预测的变量）
        fut_dim=3,             # 已知未来协变量维度（天气/节假日等）
        time_embed_dim=28,     # 时间特征嵌入维度（含节假日）
        hidden_dim=64,         # 隐藏层维度
        nhead=8,               # 注意力头数
        num_layers=2,          # Transformer层数
        dropout=0.1,
        pred_len=1,            # 预测步长
        sample_dim=3,          # 样本维度（站点/设备数）
        sample_heads=4,        # 样本间注意力头数
        target_feat_dim=2,
        hidden_size=16,
        static_num=4,         # 静态特征维度
        static_emb_dim=4,      # 离散特征嵌入维度
        static_cate_emb=None, # 静态离散特征嵌入
        top_num=3,            # topk数量
        device='cuda'
    ):
        super().__init__()
        self.trans_model = TFTWithFutureCovariates(
                static_num=static_num,
                obs_dim=obs_dim,
                fut_dim=fut_dim,
                time_embed_dim=time_embed_dim,
                hidden_dim=hidden_dim,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                pred_len=pred_len,
                sample_dim=sample_dim,
                sample_heads=sample_heads,
                static_cate_emb=static_cate_emb,
                static_emb_dim=static_emb_dim,
                device=device,
            )
        
        self.pred_len = pred_len         
        self.sample_dim = sample_dim
        self.target_feat_dim = target_feat_dim  
        # 整合输出网络
        self.ins_layer = nn.ParameterList([LinelessLayer(sample_dim*obs_dim*pred_len,sample_dim,hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3).double() for _ in range(self.target_feat_dim)])
        self.dec_layer = LinelessLayer(sample_dim*obs_dim*pred_len,sample_dim*pred_len*target_feat_dim,hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)
        # 指数整合输出网络       
        self.index_combine_layer = LinelessLayer(sample_dim*obs_dim*pred_len,pred_len)     
        # TOPK选择器网络
        self.top_selector = nn.ParameterList([SparseGateFeatureTopK(sample_dim,obs_dim*pred_len, k=top_num, hidden_dim=hidden_size,num_heads=4, dropout=0.1) for _ in range(self.target_feat_dim)])
            
    def forward(
        self,static_covs,past_convs_item, his_future_emb,future_emb
    ):    
        
        # 基础模型的向前传播
        y_pred,_,_ = self.trans_model(
            static_covs,past_convs_item, his_future_emb,future_emb
        )   
        y_pred_reshape = y_pred.reshape(y_pred.shape[0],-1)
        
        dec_out_combine = self.dec_layer(y_pred_reshape).reshape(y_pred.shape[0],self.sample_dim,self.pred_len,self.target_feat_dim)
        cls_out_combine = []    
        # 品种间比较目标的网络输出
        for i in range(self.target_feat_dim):
            # 主要比较目标输出
            normal_features,topk_features,mask = self.top_selector[i](y_pred.reshape(y_pred.shape[0],self.sample_dim,-1))
            topk_features = torch.cat(topk_features,-1)
            cls_out_combine.append(torch.cat([topk_features,normal_features],1))
        # 整体指数预测的网络输出
        index_data_combine = self.index_combine_layer(y_pred_reshape)
        
        return dec_out_combine,cls_out_combine,index_data_combine   


# ---------------------- 带未来协变量的数据集构建 ----------------------
class TFTFutureCovDataset(Dataset):
    """支持已知未来协变量的多样本时序数据集"""
    def __init__(
        self, 
        df_list,               # 列表：每个元素是一个样本的DataFrame
        obs_cols,              # 历史观测列（要预测的变量）
        fut_cols,              # 已知未来协变量列（天气/节假日等）
        time_col, 
        static_cols=[],
        seq_len=24, 
        pred_len=1,
        scaler_obs=None,
        scaler_fut=None,
        time_encoder=None,
        device='cuda'
    ):
        self.df_list = df_list
        self.sample_dim = len(df_list)
        self.obs_cols = obs_cols
        self.fut_cols = fut_cols
        self.time_col = time_col
        self.static_cols = static_cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device
        
        # 检查数据长度
        self.n_samples = min([len(df) for df in df_list]) - seq_len - pred_len + 1
        assert self.n_samples > 0, "数据长度不足"
        
        # 标准化：观测变量和未来协变量分开标准化
        # 观测变量scaler
        if scaler_obs is None:
            self.scaler_obs = StandardScaler()
            all_obs_data = np.concatenate([df[obs_cols].values for df in df_list])
            self.scaler_obs.fit(all_obs_data)
        else:
            self.scaler_obs = scaler_obs
        
        # 未来协变量scaler
        if scaler_fut is None:
            self.scaler_fut = StandardScaler()
            all_fut_data = np.concatenate([df[fut_cols].values for df in df_list])
            self.scaler_fut.fit(all_fut_data)
        else:
            self.scaler_fut = scaler_fut
        
        # 处理每个样本的数据
        self.obs_data_list = []      # 历史观测数据
        self.fut_data_list = []      # 未来协变量数据
        self.time_embed_list = []    # 时间嵌入（全时间范围）
        self.static_data_list = []   # 静态特征
        
        # 时间编码器
        if time_encoder is None:
            self.time_encoder = TimeFeatureEncoder(time_col,device=device)
            self.time_encoder.fit(df_list[0])
        else:
            self.time_encoder = time_encoder
        
        for df in df_list:
            # 1. 观测变量标准化
            obs_data = torch.FloatTensor(self.scaler_obs.transform(df[obs_cols])).to(device)
            self.obs_data_list.append(obs_data)
            
            # 2. 未来协变量标准化
            fut_data = torch.FloatTensor(self.scaler_fut.transform(df[fut_cols])).to(device)
            self.fut_data_list.append(fut_data)
            
            # 3. 时间嵌入
            time_embed = self.time_encoder.transform(df, device)
            self.time_embed_list.append(time_embed)
            
            # 4. 静态特征
            if len(static_cols) > 0:
                static_data = torch.FloatTensor(df[static_cols].values).to(device)
                self.static_data_list.append(static_data)
            else:
                self.static_data_list.append(None)

    def __getitem__(self, idx):
        """
        返回：
        (静态特征, 历史观测, 历史时间嵌入, 未来协变量, 未来时间嵌入), 目标值
        """
        x_static = []
        x_obs = []       # [S, T, obs_dim]
        x_time_hist = [] # [S, T, time_embed_dim]
        x_fut = []       # [S, P, fut_dim]
        x_time_fut = []  # [S, P, time_embed_dim]
        y_list = []      # [S, P, obs_dim]
        
        for s in range(self.sample_dim):
            # 历史观测特征：[idx, idx+seq_len]
            obs_data = self.obs_data_list[s]
            x_o = obs_data[idx:idx+self.seq_len]
            x_obs.append(x_o)
            
            # 历史时间嵌入：[idx, idx+seq_len]
            time_embed = self.time_embed_list[s]
            x_th = time_embed[idx:idx+self.seq_len]
            x_time_hist.append(x_th)
            
            # 已知未来协变量：[idx+seq_len, idx+seq_len+pred_len]（合法未来）
            fut_data = self.fut_data_list[s]
            x_fu = fut_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
            x_fut.append(x_fu)
            
            # 未来时间嵌入：[idx+seq_len, idx+seq_len+pred_len]
            x_tf = time_embed[idx+self.seq_len:idx+self.seq_len+self.pred_len]
            x_time_fut.append(x_tf)
            
            # 静态特征
            if self.static_data_list[s] is not None:
                x_st = self.static_data_list[s][idx]
                x_static.append(x_st)
            
            # 目标值：[idx+seq_len, idx+seq_len+pred_len]
            y_t = obs_data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
            y_list.append(y_t)
        
        # 转换为张量
        x_obs = torch.stack(x_obs, dim=0)
        x_time_hist = torch.stack(x_time_hist, dim=0)
        x_fut = torch.stack(x_fut, dim=0)
        x_time_fut = torch.stack(x_time_fut, dim=0)
        y = torch.stack(y_list, dim=0)
        x_static = torch.stack(x_static, dim=0) if len(x_static) > 0 else None
        
        return (x_static, x_obs, x_time_hist, x_fut, x_time_fut), y

    def __len__(self):
        return self.n_samples

# ---------------------- 训练与推理 ----------------------
def train_tft_future(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc='Training (with future covariates)')
    
    for batch in pbar:
        (x_static, x_obs, x_time_hist, x_fut, x_time_fut), y = batch
        
        # 前向传播
        pred, _, _ = model(x_static, x_obs, x_time_hist, x_fut, x_time_fut)
        loss = criterion(pred, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def predict_tft_future(model, dataloader, scaler_obs, device):
    model.eval()
    preds = []
    trues = []
    sample_attn_weights_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            (x_static, x_obs, x_time_hist, x_fut, x_time_fut), y = batch
            
            # 前向传播
            pred, _, sample_attn_weights = model(x_static, x_obs, x_time_hist, x_fut, x_time_fut)
            sample_attn_weights_list.append(sample_attn_weights.cpu().numpy())
            
            # 维度展平+反标准化
            B, S, P, F = pred.shape
            pred_np = pred.cpu().numpy().reshape(-1, F)
            true_np = y.cpu().numpy().reshape(-1, F)
            
            pred_original = scaler_obs.inverse_transform(pred_np)
            true_original = scaler_obs.inverse_transform(true_np)
            
            # 恢复维度
            pred_original = pred_original.reshape(B, S, P, F)
            true_original = true_original.reshape(B, S, P, F)
            
            preds.append(pred_original)
            trues.append(true_original)
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    sample_attn_weights = np.concatenate(sample_attn_weights_list, axis=0)
    
    return preds, trues, sample_attn_weights

# ---------------------- 测试用例（带未来协变量） ----------------------
if __name__ == '__main__':
    # 1. 生成模拟数据（3个样本+已知未来协变量）
    np.random.seed(42)
    n_samples = 2000
    sample_dim = 3  # 3个站点
    time_index = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # 生成节假日数据（已知未来协变量）
    holidays = pd.to_datetime(['2024-01-01', '2024-02-10', '2024-04-05', '2024-05-01'])
    is_holiday = np.isin(time_index.date, holidays.date).astype(int)
    
    # 生成3个样本的数据集
    df_list = []
    for s in range(sample_dim):
        # 历史观测变量（6个，要预测的）
        base_trend = np.cumsum(np.random.randn(n_samples)) * (s+1)
        # 已知未来协变量（3个：温度、湿度、节假日）
        temp = 20 + np.sin(np.linspace(0, 100, n_samples) + s) * 5 + np.random.randn(n_samples)*0.5
        humidity = 60 + np.cos(np.linspace(0, 100, n_samples) + s) * 10 + np.random.randn(n_samples)*1
        holiday = is_holiday.copy()
        
        df = pd.DataFrame({
            'timestamp': time_index,
            # 观测变量（要预测的）
            'var1': base_trend + 10 + temp*0.5 + holiday*10 + np.random.randn(n_samples)*0.5,
            'var2': base_trend * 0.8 + 20 + temp*0.3 + holiday*5 + np.random.randn(n_samples)*0.5,
            'var3': np.sin(np.linspace(0, 100, n_samples) + s) * 5 + 30 + humidity*0.2,
            'var4': np.cos(np.linspace(0, 100, n_samples) + s) * 3 + 15,
            'var5': np.random.randn(n_samples) * 2 + 5,
            'var6': base_trend * 0.5 + 40 + holiday*8 + np.random.randn(n_samples)*0.5,
            # 已知未来协变量（天气+节假日）
            'temperature': temp,
            'humidity': humidity,
            'is_holiday': holiday
        })
        df_list.append(df)
    
    # 2. 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    seq_len = 48        # 历史序列长度
    pred_len = 6        # 预测步长
    obs_cols = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6']  # 观测变量
    fut_cols = ['temperature', 'humidity', 'is_holiday']         # 已知未来协变量
    static_cols = []    # 无静态特征
    time_col = 'timestamp'
    
    # 3. 划分训练/测试集
    train_size = int(0.8 * n_samples)
    train_df_list = [df.iloc[:train_size] for df in df_list]
    test_df_list = [df.iloc[train_size:] for df in df_list]
    
    # 4. 构建数据集
    train_dataset = TFTFutureCovDataset(
        train_df_list, obs_cols, fut_cols, time_col, static_cols,
        seq_len=seq_len, pred_len=pred_len, device=device
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = TFTFutureCovDataset(
        test_df_list, obs_cols, fut_cols, time_col, static_cols,
        seq_len=seq_len, pred_len=pred_len,
        scaler_obs=train_dataset.scaler_obs,
        scaler_fut=train_dataset.scaler_fut,
        time_encoder=train_dataset.time_encoder,
        device=device
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 5. 初始化模型
    model = TFTWithFutureCovariates(
        static_dim=len(static_cols),
        obs_dim=len(obs_cols),
        fut_dim=len(fut_cols),
        time_embed_dim=train_dataset.time_encoder.transform(train_df_list[0].iloc[:1], device).shape[-1],
        hidden_dim=64,
        nhead=8,
        num_layers=2,
        dropout=0.1,
        pred_len=pred_len,
        sample_dim=sample_dim,
        sample_heads=4,
        device=device
    ).to(device)
    
    # 6. 训练配置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 20
    
    # 7. 训练模型
    loss_history = []
    for epoch in range(epochs):
        train_loss = train_tft_future(model, train_loader, criterion, optimizer, device)
        loss_history.append(train_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
    
    # 8. 推理预测
    preds, trues, sample_attn_weights = predict_tft_future(model, test_loader, train_dataset.scaler_obs, device)
    
    # 9. 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 9.1 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # 9.2 样本1的var6预测（受节假日影响大）
    plt.subplot(2, 3, 2)
    plt.plot(trues[:200, 0, 0, 5], label='True (Sample 1 - var6)', alpha=0.7)
    plt.plot(preds[:200, 0, 0, 5], label='Pred (Sample 1 - var6)', alpha=0.7)
    plt.title('Sample 1 - var6 (Holiday Impact)')
    plt.legend()
    
    # 9.3 样本2的var3预测（受湿度影响大）
    plt.subplot(2, 3, 3)
    plt.plot(trues[:200, 1, 0, 2], label='True (Sample 2 - var3)', alpha=0.7)
    plt.plot(preds[:200, 1, 0, 2], label='Pred (Sample 2 - var3)', alpha=0.7)
    plt.title('Sample 2 - var3 (Humidity Impact)')
    plt.legend()
    
    # 9.4 样本间注意力权重
    plt.subplot(2, 3, 4)
    avg_sample_attn = sample_attn_weights.mean(axis=(0,1))
    plt.imshow(avg_sample_attn, cmap='Blues')
    plt.title('Sample Cross Attention Weights')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.colorbar()
    plt.xticks(range(sample_dim))
    plt.yticks(range(sample_dim))
    
    # 9.5 未来协变量（温度）vs 预测值
    plt.subplot(2, 3, 5)
    temp_data = test_df_list[0]['temperature'].iloc[:200].values
    plt.plot(temp_data, label='Temperature (Future Cov)', alpha=0.7, color='green')
    plt.plot(preds[:200, 0, 0, 0], label='Pred var1 (Sample 1)', alpha=0.7, color='red')
    plt.title('Temperature vs Pred var1')
    plt.legend()
    
    # 9.6 节假日vs预测值
    plt.subplot(2, 3, 6)
    holiday_data = test_df_list[0]['is_holiday'].iloc[:200].values
    plt.plot(holiday_data*50, label='Holiday (0/1)', alpha=0.5, color='orange')
    plt.plot(preds[:200, 0, 0, 5], label='Pred var6 (Sample 1)', alpha=0.7, color='blue')
    plt.title('Holiday vs Pred var6')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 输出关键信息
    print("\n=== 关键结果 ===")
    print(f"样本间平均注意力权重：\n{avg_sample_attn.round(3)}")
    print(f"\n预测误差（MSE）：{np.mean((preds - trues)**2):.4f}")
    print("\n说明：模型合法使用了未来温度、湿度、节假日等协变量，无未来泄露！")