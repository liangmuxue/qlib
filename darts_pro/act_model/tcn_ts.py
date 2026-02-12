import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from .cov_cnn import LinelessLayer

def create_same_padding_conv3d(in_channels, out_channels, kernel_size, dilation=1):
    """创建保持空间尺寸不变的Conv3d层"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    
    # 计算保持尺寸的padding
    padding = []
    for k in kernel_size:
        padding.append((k - 1) * dilation // 2)
    
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=tuple(padding),
        dilation=dilation
    )

class FractionalStrideConvTranspose(nn.Module):
    """分数步长转置卷积模拟"""
    def __init__(self, in_channels, out_channels, input_size=3, output_size=19,drop=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # 计算近似的分数步长
        # 19/3 ≈ 6.333，我们可以使用 19/3 作为近似的步长
        self.upsample_ratio = output_size / input_size
        
        # 使用插值近似分数步长
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1, 3),
            stride=1,
            padding=(0, 0, 1)
        )

        self.norm = nn.BatchNorm3d(out_channels)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(drop)
            
    def forward(self, x):
        # 获取输入形状
        B, C, D, H, W = x.shape
        
        # 计算目标尺寸
        target_W = int(W * self.upsample_ratio)
        
        # 先插值上采样
        x = F.interpolate(
            x,
            size=(D, H, target_W),
            mode='trilinear',
            align_corners=True
        )
        # 然后应用卷积
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(self.act(x))
        
        return x

    
class CausalConv3D(nn.Module):
    def __init__(self, in_c, out_c, k=(3,3,3), d=1, act=nn.ReLU(), drop=0.1):
        super().__init__()
        kt,kc,kf = k
        dt,dc,df = d if isinstance(d,tuple) else (d,1,1)
        self.pad_t = (kt-1)*dt  # 时间因果：只填左边
        self.pad_c = (kc-1)//2
        self.pad_f = (kf-1)//2
        self.pad = nn.ConstantPad3d((self.pad_f,self.pad_f, self.pad_c,self.pad_c, self.pad_t,0), 0)
        self.conv = create_same_padding_conv3d(in_c, out_c, kernel_size=k, dilation=d)
        self.norm = nn.BatchNorm3d(out_c)
        self.act  = act
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))

class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, k=(3,3,3), d=1, drop=0.1,act=nn.GELU()):
        super().__init__()
        self.conv1 = CausalConv3D(in_c, out_c, k, d, drop=drop,act=act)
        self.conv2 = CausalConv3D(out_c, out_c, k, d, drop=drop,act=act)
        self.skip  = nn.Conv3d(in_c, out_c, 1) if in_c!=out_c else nn.Identity()
    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class TCN3DPredictor(nn.Module):
    def __init__(self, hist_feat, static_feat, fut_feat, n_channel, seq_len, pred_len,act=nn.GELU(),
                 target_feat_dim=1,tcn_channels=[64,32,16], k=(3,3,3), drop=0.1,main_feat=1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channel= n_channel
        self.main_feat = main_feat
        self.target_feat_dim = target_feat_dim

        # ========== 1) 历史特征编码（3D-TCN主干） ==========
        layers = []
        in_c = 1
        for c in tcn_channels:
            layers.append(ResBlock3D(in_c, c, k, d=len(layers)+1, drop=drop,act=act))
            in_c = c
        self.tcn = nn.Sequential(*layers)
        self.tcn_out_c = tcn_channels[-1]

        # ========== 2) 静态协变量编码（未来也要用） ==========
        self.stat_proj = nn.Sequential(
            nn.Linear(static_feat, target_feat_dim), act, nn.LayerNorm(target_feat_dim)
        )

        # ========== 3) 未来协变量编码 ==========
        # self.fut_proj = CausalConv3D(1, tcn_out_c, (1,3,3), d=1, drop=drop)
        self.fut_proj = FractionalStrideConvTranspose(1, self.tcn_out_c, input_size=fut_feat, output_size=target_feat_dim,drop=drop)

        # ========== 4) 时间映射：SEQ_LEN → PRED_LEN ==========
        self.flat_dim = seq_len * n_channel * hist_feat
        self.time_map = nn.Sequential(
            nn.Flatten(2), nn.Linear(self.flat_dim, pred_len * n_channel * target_feat_dim), act
        )
        # ========== 5)融合特征+特征维度投影 ==========
        self.fusion_feat_proj = nn.Conv3d(
            self.tcn_out_c, self.tcn_out_c,
            kernel_size=(1,1,1), 
            bias=False
        )
        # ========== 6) 输出头 ==========
        self.out_head = nn.Sequential(
            CausalConv3D(self.tcn_out_c, 1, k=(1,1,1), drop=drop),
            # nn.Conv3d(1, 1, kernel_size=(1,1,1)),
        )        


    def forward(self, x_hist, x_stat, x_fut):
        """
        x_hist: [B, T, C, F_hist]  主序列+未来历史
        x_stat: [B, C, F_static]   静态协变量 ✅ 关键输入
        x_fut:  [B, Tp, C, F_fut]  未来协变量
        return: [B, Tp, C, F_main]
        """
        B = x_hist.shape[0]

        # --------------------- 1. 历史3D编码 ---------------------
        x = x_hist.unsqueeze(1)  # [B,1,T,C,F_hist]
        feat_hist = self.tcn(x)  # [B,Ct,T,C,F_hist]

        # --------------------- 2. 时间映射到预测步 ---------------------
        feat_pred = self.time_map(feat_hist)
        feat_pred = feat_pred.unflatten(2, (self.pred_len, self.n_channel, -1))  # [B,Ct,Tp,C,F]

        # --------------------- 3. 静态协变量扩维到未来 ✅ ---------------------
        s = self.stat_proj(x_stat)                          # [B,C,F_hist]
        s = s.unsqueeze(1).repeat(1,self.pred_len,1,1)      # [B,Tp,C,F_hist]
        s = s.unsqueeze(1).repeat(1,feat_pred.shape[1],1,1,1)# [B,Ct,Tp,C,F_hist]

        # --------------------- 4. 未来协变量编码 ---------------------
        f = x_fut.unsqueeze(1)                               # [B,1,Tp,C,F_fut]
        feat_fut = self.fut_proj(f)                          # [B,Ct,Tp,C,F_hist]

        # --------------------- 5. 三源融合：历史 + 静态 + 未来 ✅ ---------------------
        fused_feat = feat_pred + s + feat_fut
        # fused_feat = self.fusion_feat_proj(fused_feat)
        
        # --------------------- 6. 输出 ---------------------
        
        out = self.out_head(fused_feat)
        out = out.squeeze(1).permute(0,2,1,3)
        return out
    
    
class UnionTcnCombine(nn.Module):
    """整合后的完整模型"""

    def __init__(
        self,
        input_dim: int,          # 历史协变量组数
        static_feat: int,          # 未来协变量组数
        fut_feat: int,            # 多目标数
        C: int,                # 全张量统一C维度（核心）
        seq_len: int = 5,     # 长度
        pred_len: int = 5,     # 预测长度
        tcn_channels=[64, 32, 16],
        k=(3,3,3),
        dropout: float = 0.1,
        act=nn.GELU(),
        hidden_size=16,
        target_feat_dim=1,
    ):
        super().__init__()
        self.tcn_model = TCN3DPredictor(
                hist_feat=input_dim,static_feat=static_feat,fut_feat=fut_feat,
                n_channel=C,seq_len=seq_len,pred_len=pred_len,target_feat_dim=target_feat_dim,
                tcn_channels=tcn_channels,k=k,drop=dropout,act=act
            )      
        self.pred_len = pred_len           
        # 整合输出网络
        self.ins_layer = LinelessLayer(C*pred_len,C,
                            hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)
        self.ins_att_layer = LinelessLayer(C*pred_len,C,
                        hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)    
        self.dec_layer = LinelessLayer(C,C,
                        hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)            
        # 指数整合输出网络       
        self.index_combine_layer = LinelessLayer(C*pred_len,pred_len)     
            
    def forward(
        self,main, static_covs,futures_convs
    ):    
        
        # 基础模型的向前传播
        y_pred = self.tcn_model(
            main, static_covs,futures_convs
        )   
        cls_out_combine = []
        index_data_combine = []
        dec_out_combine = []        
        # 品种间比较目标的网络输出
        pred_reshape_0 = y_pred[...,0].reshape(y_pred.shape[0],-1)
        # pred_reshape_1 = y_pred[...,1].reshape(y_pred.shape[0],-1)
        dec_out_combine = self.dec_layer(y_pred.permute(0,3,2,1)).permute(0,3,2,1)
        # 主要比较目标输出
        cls_out_ins = self.ins_layer(pred_reshape_0)
        # 添加辅助品种比较目标输出
        cls_out_ins_att = self.ins_att_layer(pred_reshape_0)    
        cls_out_combine.append(cls_out_ins)
        cls_out_combine.append(cls_out_ins_att)  
        # 整体指数预测的网络输出
        sw_index_data = self.index_combine_layer(pred_reshape_0)     
        index_data_combine = sw_index_data
        
        return dec_out_combine,cls_out_combine,index_data_combine     
    
    