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
        self.fut_single_proj = nn.Linear(time_embed_dim + hidden_dim, hidden_dim)
        
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
        time_embed_fut_singel_flat = time_embed_fut_single.reshape(B*S, 1, time_embed_fut.shape[-1])
        
        # 未来协变量投影
        # 静态特征广播到所有预测步：[B*S,1,hidden_dim] → [B*S,P,hidden_dim]
        static_broadcast = static_context_fut.repeat(1, P, 1)
        # 拼接未来协变量 + 静态特征
        fut_input = torch.cat([time_embed_fut_flat, static_broadcast], dim=-1)  # [B*S,P,F_time+F_static]  
        fut_single_input = torch.cat([time_embed_fut_singel_flat, static_context_fut.repeat(1, 1, 1)], dim=-1)  # [B*S,1,F_time+F_static]  
                  
        fut_proj = self.fut_proj(fut_input)  # [B*S, P, hidden_dim]
        fut_single_proj = self.fut_single_proj(fut_single_input)  # [B*S, P, hidden_dim]
        
        
        # 针对序列目标和单独阶段目标分别进行解码
        pred_seq = self.seq_decoder(hist_summary,fut_proj)        # [B*S*P, obs_dim]
        pred_tar = self.tar_decoder(hist_summary,fut_single_proj)        # [B*S*1, obs_dim]
        
        # # 4.4 变量权重恢复
        # var_weights = var_weights.reshape(B, S, T, self.obs_dim)  # [B, S, T, obs_dim]
        
        return (pred_seq,pred_tar), sample_attn_weights

def mask_with_flag(features,mask):
        mask_exp = mask.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        features_exp = features * mask_exp   
        return features_exp
    

def straight_through_topk_bottomk(x, scores, k, temperature=1.0):
    """
    同时选择 Top‑k 和 Bottom‑k 特征（硬掩码 + 直通估计）
    Args:
        x: 输入特征，形状 (batch_size, feature_dim) 或 (batch_size, seq_len, feature_dim)
        scores: 重要性分数，形状与 x 相同
        k: 需要保留的前 k 个和后 k 个（总保留 2k 个，若 2k > 特征总数则全部保留）
        temperature: 控制软权重的平滑程度
    Returns:
        selected_x: 经过掩码后的特征（未选中位置置零）
        hard_mask: 二值掩码，形状与 x 相同
    """
    device = x.device
    batch_size = x.size(0)
    feat_dim = x.size(-1)
    
    # 1. 获取 Top‑k 和 Bottom‑k 索引
    topk_vals, topk_idx = torch.topk(scores, k, dim=-1)               # 前 k 个最大值的索引
    bottomk_vals, bottomk_idx = torch.topk(-scores, k, dim=-1)        # 前 k 个最小值的索引（通过取负实现）
    
    # 2. 构建硬掩码
    hard_mask = torch.zeros_like(scores)
    hard_mask.scatter_(-1, topk_idx, 1.0)
    hard_mask.scatter_(-1, bottomk_idx, 1.0)   # 注意：若 topk 和 bottomk 有重叠，会重复赋值（仍为1）
    
    # 3. 软权重（用于梯度近似）
    # 对分数进行温度缩放后 softmax，作为每个位置的软权重
    soft_weights = F.softmax(scores / temperature, dim=-1)
    
    # 4. Straight‑Through 估计
    weights = (hard_mask - soft_weights.detach() + soft_weights)
    weights = weights.unsqueeze(-1).repeat(1,1,x.shape[-1])
    selected_x = x * weights
    return selected_x, soft_weights
    
    
def soft_topk_bottomk_mask(x,scores, k, sigma=1.0):
    """
    生成软性两端掩码（值在 0~1 之间）
    Args:
        x: 输入特征，形状 (batch_size, feature_dim) 或 (batch_size, seq_len, feature_dim)
        scores: 重要性分数，形状 (batch_size, n)
        k: 需要保留的前 k 个和后 k 个
        sigma: 控制软掩码锐利程度的高斯核宽度
    Returns:
        soft_mask: 软掩码，形状与 scores 相同
    """
    # 1. 找到第 k 大和第 k 小的值
    topk_vals, _ = torch.topk(scores, k, dim=-1)
    bottomk_vals, _ = torch.topk(-scores, k, dim=-1)   # 取负找最小
    threshold_high = topk_vals[..., -1:]                # 第 k 大的值
    threshold_low = -bottomk_vals[..., -1:]             # 第 k 小的值（还原符号）
    
    # 2. 计算每个元素与两个阈值的距离
    diff_high = scores - threshold_high                 # 大于阈值时为正
    diff_low = threshold_low - scores                   # 小于阈值时为正
    
    # 3. 使用高斯核：距离越大，越接近 1
    # 对于大于高阈值的元素，mask_high 接近 1；对于小于低阈值的元素，mask_low 接近 1
    mask_high = torch.exp(- (diff_high / sigma) ** 2)   # 当 diff_high > 0 时 mask_high 较大
    mask_low  = torch.exp(- (diff_low / sigma) ** 2)    # 当 diff_low > 0 时 mask_low 较大
    
    # 4. 合并：取两个掩码的逐元素最大值（或求和后截断）
    soft_mask = torch.max(mask_high, mask_low)           # 并集操作
    selected = x * soft_mask.unsqueeze(-1).repeat(1,1,x.shape[-1])
    
    return selected, (mask_high, mask_low)   

class RankAttention(nn.Module):
    
    def __init__(self,input_dim, sample_dim,hidden_size=16, top_k=3,dropout=0.3):
        super().__init__()
        self.top_k = top_k 
        self.input_dim = input_dim      
        
        self.line_layer = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_size,
                                    layer_norm=True,batch_norm=False,dropout=dropout)    
        self.score_head = LinelessLayer(sample_dim*input_dim,sample_dim,hidden_size=hidden_size,
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
        batch_size, node_num, _ = batch_features.shape
        # 计算初始分数        
        flat_features = batch_features.reshape(batch_features.shape[0],-1)
        batch_raw_scores = self.line_layer(flat_features)  # (batch_size, node_num)
        
        # 计算批次注意力权重
        # selected, att_weights = straight_through_topk_bottomk(batch_features, batch_raw_scores, k=self.top_k)
        selected, att_weights = soft_topk_bottomk_mask(batch_features, batch_raw_scores, k=self.top_k, sigma=0.3)
        final_scores = self.score_head(selected.reshape(batch_size,-1))
        
        return final_scores, att_weights
             
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
        
        topk_features,attention_weights = self.top_att_layer(x)    
        normal_features = self.ins_layer(x.reshape(x.shape[0],-1))  
        
        return normal_features,topk_features,attention_weights

   
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
        self.top_selector = nn.ParameterList([SparseGateFeatureTopK(sample_dim,obs_dim, k=top_num, hidden_dim=hidden_size,num_heads=4, dropout=0.1) for _ in range(self.target_feat_dim)])
            
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
            normal_features,topk_features,topk_weights = self.top_selector[i](pred_tar.reshape(pred_tar.shape[0],self.sample_dim,-1))
            topk_weights = torch.cat(topk_weights,-1)
            cls_out_combine.append(torch.cat([topk_features,topk_weights,normal_features],1))
        # 整体指数预测的网络输出
        # index_data_combine = self.index_combine_layer(y_pred_reshape)
        index_data_combine = pred_seq[:,0,:,0]
        
        return dec_out_combine,cls_out_combine,index_data_combine   

