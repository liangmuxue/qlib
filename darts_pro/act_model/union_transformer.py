import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from cus_utils.common_compute import normalization_standard

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
        
        # 未来协变量融合网络（关键：融合历史编码+未来协变量）
        self.future_fusion = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        
        # 门控注意力融合
        self.attention_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 输出层（预测目标变量）
        self.output_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.final_proj = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self,hist_summary,fut_proj):
        
        S = self.sample_dim
        P = self.pred_len
        B = int(hist_summary.shape[0]/S)
        
        # 融合历史总结+未来协变量（核心：历史指导未来预测）
        hist_summary_expanded = hist_summary.unsqueeze(1).repeat(1, P, 1)  # [B*S, P, hidden_dim]
        fusion_input = torch.cat([hist_summary_expanded, fut_proj], dim=-1)  # [B*S, P, 2*hidden_dim]
        
        fusion_out = self.future_fusion(fusion_input)  # [B*S, P, hidden_dim]
        
        # ---------------------- 步骤：输出预测 ----------------------
        # 门控融合
        final_feat = self.attention_gate(fusion_out)  # [B*S, P, hidden_dim]
        
        # 输出层（每个预测步独立预测）
        final_feat_flat = final_feat.reshape(B*S*P, self.hidden_dim)  # [B*S*P, hidden_dim]
        final_feat_processed = self.output_grn(final_feat_flat)   # [B*S*P, hidden_dim]
        pred_flat = self.final_proj(final_feat_processed)        # [B*S*P, obs_dim]        
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
        # x: [batch_size, seq_len, feature_dim] 原始历史特征
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
        final_out = global_pool + 0.1 * residual  # 残差缩放，避免主导
        
        return final_out
           
class TFTWithFutureCovariates(nn.Module):
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
        
        # 全局池化多尺度
        self.pool_layer = MultiScaleGlobalPool(obs_dim, seq_len)
        
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
        self.fut_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.fut_single_proj = nn.Linear(time_embed_dim, hidden_dim)
        
        # 5. 变量选择网络（仅对历史观测变量）
        self.var_selection = VariableSelectionNetwork(hidden_dim, obs_dim, hidden_dim, dropout)
        
        # 6. Transformer编码器（因果掩码，仅处理历史）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        encoder_norm = nn.LayerNorm(hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,norm=encoder_norm)
        
        # 分别定义完整预测序列的解码器，以及最终单独目标的解码器
        self.seq_decoder = DecoderLayer(obs_dim=obs_dim, hidden_dim=hidden_dim,sample_dim=sample_dim,
                     dropout=dropout, pred_len=pred_len)
        self.tar_decoder = DecoderLayer(obs_dim=obs_dim, hidden_dim=hidden_dim,sample_dim=sample_dim,
                     dropout=dropout, pred_len=1)        
        

    def forward(self, static_feat=None, obs_feat=None, time_embed_hist=None, time_embed_fut=None,time_embed_fut_single=None):
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
        B, S, T, _ = obs_feat.shape  # B=batch, S=样本数, T=历史序列长度
        P = self.pred_len            # P=预测步长
        
        # ---------------------- 步骤1：样本维度交互（仅历史，无未来泄露） ----------------------
        # 1.1 历史特征全局池化
        obs_global = self.pool_layer(obs_feat.reshape(B*S,T,-1)).reshape(B,S,-1)  # [B, S, obs_dim]
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
        time_embed_fut_singel_flat = time_embed_fut_single.reshape(B*S, 1, time_embed_fut.shape[-1])
        
        # 未来协变量投影
        # 静态特征广播到所有预测步：[B*S,1,hidden_dim] → [B*S,P,hidden_dim]
        static_broadcast = static_context_fut.repeat(1, P, 1)
        # 拼接未来协变量 + 静态特征--未来不使用静态特征，静态特征固定比重会干扰网络传播
        # fut_input = torch.cat([time_embed_fut_flat, static_broadcast], dim=-1)  # [B*S,P,F_time+F_static]  
        fut_input = time_embed_fut_flat
        # fut_single_input = torch.cat([time_embed_fut_singel_flat, static_context_fut.repeat(1, 1, 1)], dim=-1)  # [B*S,1,F_time+F_static]  
        fut_single_input = time_embed_fut_singel_flat 
                  
        fut_proj = self.fut_proj(fut_input)  # [B*S, P, hidden_dim]
        fut_single_proj = self.fut_single_proj(fut_single_input)  # [B*S, P, hidden_dim]
        
        
        # 针对序列目标和单独阶段目标分别进行解码
        pred_seq = self.seq_decoder(hist_summary,fut_proj)        # [B*S*P, obs_dim]
        pred_tar = self.tar_decoder(hist_summary,fut_single_proj)        # [B*S*1, obs_dim]
        
        # # 4.4 变量权重恢复
        # var_weights = var_weights.reshape(B, S, T, self.obs_dim)  # [B, S, T, obs_dim]
        
        return (pred_seq,pred_tar), sample_attn_weights

class AttentionWithDualHighK(nn.Module):
    def __init__(self, input_dim,node_num=0, hidden_dim=16, k=3,nor_scale=0.3,high_scale=1.0,low_scale=2.0):
        super(AttentionWithDualHighK, self).__init__()
        self.k = k
        self.nor_scale = nor_scale
        self.high_scale = high_scale
        self.low_scale = low_scale
        
        # 注意力层：生成原始分数
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1)
        )
        # self.attention_net = nn.Linear(input_dim, 1)
        # self.attention_net = LinelessLayer(node_num*input_dim,node_num,hidden_size=hidden_dim,
        #                             layer_norm=True,batch_norm=False,dropout=0.1)     
        # 可学习的缩放因子：调整Sigmoid输入的范围，避免全趋近于1
        self.scale = nn.Parameter(torch.tensor(1.0))       
    
    def forward(self, x):
        """
        前向传播：计算注意力权重，筛选top-k（排序最高）和bottom-k（排序最低）的位置
        :param x: 输入张量，shape=[batch_size, seq_len, input_dim]
        :return: 
            attention_weights: 原始注意力权重，shape=[batch_size, seq_len]
            topk_indices: 排序前k的索引，shape=[batch_size, k]
            bottomk_indices: 排序后k的索引，shape=[batch_size, k]
        """
        batch_size, seq_len, _ = x.shape
        # 校验：序列长度必须大于2k，避免前k和后k索引重叠
        if seq_len <= 2 * self.k:
            raise ValueError(f"序列长度{seq_len}必须大于2*{self.k}={2*self.k}")
        
        # 计算注意力分数并归一化（softmax确保权重和为1）
        raw_scores = self.attention_net(x).reshape(batch_size,seq_len)  # [batch_size, seq_len]
        # 关键优化1：缩放分数，避免Sigmoid饱和（scale可学习）
        scaled_scores = raw_scores * self.scale     
        # 步骤2：Sigmoid激活（输出0~1）
        # sigmoid_weights = torch.sigmoid(scaled_scores)  # [batch, seq]
        # 关键优化2：L2归一化（核心！让权重区分高低，而非全接近1）
        # L2归一化：每个样本的权重除以其L2范数，保证权重有差异
        attn_scores = F.normalize(scaled_scores, p=2, dim=-1)
                           
        # 筛选排序前k的索引（原本权重最高）
        _, topk_indices = torch.topk(attn_scores, self.k, dim=-1)
        # 筛选排序后k的索引（原本权重最低）
        _, bottomk_indices = torch.topk(attn_scores, self.k, dim=-1, largest=False)
                
        # 填充前K高分（并缩放增强）,填充后K低分（并缩放增强，避免权重消失）
        mask_scores,mask_scores_hard = self.get_topk_bottomk_mask(attn_scores)    
        # 权重归一化
        attention_weights = F.softmax(mask_scores.squeeze(-1), dim=-1)  # [batch_size, seq_len]
        
        # 返回权重，以及硬掩码用于后续topk索引定位
        return attention_weights, mask_scores_hard

    def get_topk_bottomk_mask(self, scores):
        """
        生成仅包含前K高分和后K低分位置的掩码分数矩阵
        :param scores: 原始注意力分数 (batch, seq_len, seq_len)
        :return: mask_scores: 仅前K+后K位置有有效分数，其余为极小值
        """
        batch_size, seq_len = scores.shape
        
        # 1. 筛选前K个高分位置和分数
        topk_vals, topk_indices = torch.topk(scores, k=self.k, dim=-1)  # (batch, seq_len, k)
        
        # 筛选后K个低分位置和分数（通过取负实现bottomk）
        bottomk_vals, bottomk_indices = torch.topk(scores, k=self.k,largest=False, dim=-1)  # (batch, seq_len, k)
        bottomk_vals = -bottomk_vals  # 还原为原始低分
        
        # 初始化掩码分数矩阵，软约束版本，其他位置保留10%分数
        mask_scores = scores * self.nor_scale  
        mask_scores_hard = torch.zeros_like(scores)   
        # 填充前K高分（并缩放增强）
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, self.k)
        mask_scores[batch_idx, topk_indices] = topk_vals * self.high_scale
        mask_scores_hard[batch_idx, topk_indices] = 1
        # 填充后K低分（并缩放增强，避免权重消失）
        mask_scores[batch_idx, bottomk_indices] = bottomk_vals * self.low_scale
        mask_scores_hard[batch_idx, bottomk_indices] = -1
        
        return mask_scores,mask_scores_hard
    
class RankAttention(nn.Module):
    
    def __init__(self,input_dim, node_num,hidden_size=16, top_k=3,dropout=0.3):
        super().__init__()
        self.top_k = top_k 
        self.input_dim = input_dim      
        
        # self.line_layer = LinelessLayer(node_num*input_dim,node_num,hidden_size=hidden_size,
        #                             layer_norm=True,batch_norm=False,dropout=dropout)    
        self.att_layer = AttentionWithDualHighK(input_dim=input_dim,node_num=node_num,k=top_k)
        self.score_head =  LinelessLayer(node_num*input_dim,node_num,hidden_size=hidden_size,
                                    layer_norm=True,batch_norm=False,dropout=dropout)            

    def forward(self, batch_features):
        """
        批次前向传播
        Args:
            batch_features: (batch_size, node_num, input_dim)
            batch_masks: (batch_size, node_num) - 掩码，避免 padding 参与计算
        Returns:
            batch_scores: (batch_size, node_num) - 每个文档的排序分数
            batch_attention_weights: (batch_size, node_num) - 每个文档的注意力权重
        """
        batch_size, node_num, fea_dim = batch_features.shape
        
        # 根据分数计算注意力权重,其中注意力分数在权重里已经兼顾到得分靠前和靠后的K个候选
        attention_weights, mask_scores_hard = self.att_layer(batch_features)
        # 根据注意力权重加权生成加权特征值，并根据新特征值计算品种得分
        attention_weights_exp = attention_weights.unsqueeze(-1).repeat(1,1,fea_dim)
        att_features = batch_features * attention_weights_exp
        scores_with_att = self.score_head(att_features.reshape(batch_size,-1))
        
        return scores_with_att,mask_scores_hard,attention_weights
             
class SparseGateFeatureTopK(nn.Module):
    """
    稀疏门控特征Top-K：基于注意力机制,并通过门控网络学习特征重要性，仅激活Top-K特征
    优势：可随输入动态调整Top-K特征（不同样本选不同特征）
    """
    def __init__(self,sample_dim,input_dim,k=3, hidden_dim=16,num_heads=4, dropout=0.1):
        super().__init__()
        self.sample_dim = sample_dim
        self.k = k
        # 整合输出为结合topk选择的品类
        # self.top_ins_layer = LinelessLayer(2*k*input_dim,k*2,hidden_size=hidden_dim,
        #                             layer_norm=True,batch_norm=False,dropout=0.3)
        self.top_att_layer = RankAttention(input_dim,sample_dim,top_k=k,hidden_size=64)      
        self.ins_layer = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_dim,
                                    layer_norm=True,batch_norm=False,dropout=0.3)    
    def forward(self, x):
        # x: (batch_size, 品种S, 特征input_dim)
        batch_size, S, input_dim = x.shape
        
        features,mask_scores_hard,attention_weights = self.top_att_layer(x)    
        # normal_features = self.ins_layer(x.reshape(x.shape[0],-1))
        normal_features = features  
        
        return features,normal_features,mask_scores_hard,attention_weights

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
                seq_len=seq_len,
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
        self.top_selector = nn.ParameterList([SparseGateFeatureTopK(sample_dim,obs_dim, k=top_num, hidden_dim=hidden_size,num_heads=4, dropout=0.1) for _ in range(self.target_feat_dim)])

        ############# 中间变量调试 #############
        self.features = {}
    
    def forward(
        self,static_covs,past_convs_item, his_future_emb,future_emb,future_single_emb
    ):    
        
        # 基础模型的向前传播
        (pred_seq,pred_tar),_ = self.trans_model(
            static_covs,past_convs_item, his_future_emb,future_emb,future_single_emb
        )   
        y_pred_reshape = pred_seq.reshape(pred_seq.shape[0],-1)
        
        # dec_out_combine = self.dec_layer(y_pred_reshape).reshape(pred_seq.shape[0],self.sample_dim,self.pred_len,self.target_feat_dim)
        dec_out_combine = pred_seq[:,:,:,:self.target_feat_dim]
        cls_out_combine = []    
        # 品种间比较目标的网络输出
        for i in range(self.target_feat_dim):
            # 主要比较目标输出
            features,normal_features,topk_mask_weights,attention_weights = self.top_selector[i](pred_tar.reshape(pred_tar.shape[0],self.sample_dim,-1))
            cls_out_combine.append(torch.cat([features,topk_mask_weights,normal_features,attention_weights],1))
        # 整体指数预测的网络输出
        # index_data_combine = self.index_combine_layer(y_pred_reshape)
        index_data_combine = pred_seq[:,0,:,0]
        
        return dec_out_combine,cls_out_combine,index_data_combine   


