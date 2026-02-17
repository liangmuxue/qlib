import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from .cov_cnn import LinelessLayer
from cus_utils.common_compute import check_nan

def create_dynamic_padding_conv2d(in_channels, out_channels, kernel_size, stride=(1,1),dilation=1):
    """创建保持空间宽度，高度和通道数变化的Conv2d层"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # 计算保持尺寸的padding
    padding = []
    for k in kernel_size:
        padding.append((k - 1) * dilation // 2)
    
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
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
    
class CausalConv2D(nn.Module):
    def __init__(self, in_c, out_c, k=(3,3), d=1,stride=(1,1), act=nn.ReLU(), drop=0.1):
        super().__init__()
        self.conv = create_dynamic_padding_conv2d(in_c, out_c, kernel_size=k, dilation=d,stride=stride)
        self.norm = nn.BatchNorm2d(out_c)
        self.act  = act
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x1 = self.conv(x)
        if check_nan(x1,"x1"):
            print("conv is:",self.conv)
            sys.exit("nan exit")    
        x2 = self.norm(x1)       
        if check_nan(x2,"x2"):
            print("norm is::",self.norm)
            sys.exit("nan exit")            
        x = self.norm(x2)
        return self.drop(self.act(x))

class CausalConv1D(nn.Module):
    def __init__(self, in_c, out_c, k=1, d=1,stride=1, act=nn.ReLU(), drop=0.1):
        super().__init__()
        self.conv = create_dynamic_padding_conv2d(in_c, out_c, kernel_size=k, dilation=d,stride=stride)
        self.norm = nn.BatchNorm1d(out_c)
        self.act  = act
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)       
        return self.drop(self.act(x))
    
class ResBlock2D(nn.Module):
    def __init__(self, in_c, out_c, k=(3,3), d=1, stride=(1,1),drop=0.1,act=nn.GELU()):
        super().__init__()
        self.conv1 = CausalConv2D(in_c, out_c, k, d, stride=stride,drop=drop,act=act)
        self.conv2 = CausalConv2D(out_c, out_c, k, d,stride=(1,1), drop=drop,act=act)
        self.skip  = nn.Conv2d(in_c, out_c,(1, 1), stride=stride, bias=False) if in_c!=out_c else nn.Identity()
    def forward(self, x):
        skip_x = self.skip(x)
        if check_nan(skip_x,"skip_x"):
            for name, param in self.named_parameters():
                if not "skip." in name:
                    continue                    
                print("{} skip param mean:{}".format(name,param.data.mean())) 
                print("{} skip grad mean:{}".format(name,param.grad.mean()))
            sys.exit("skip nan exit")           
        return self.conv2(self.conv1(x)) + skip_x


class TCN2DPredictor(nn.Module):
    def __init__(self, hist_feat, static_feat, fut_feat, n_channel, seq_len, pred_len,act=nn.GELU(),
                 target_feat_dim=1,tcn_channels=[64,32,16], k=(3,3), drop=0.1,main_feat=1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channel= n_channel
        self.main_feat = main_feat
        self.target_feat_dim = target_feat_dim

        # ========== 1) 历史特征编码（3D-TCN主干） ==========
        layers = []
        in_c = hist_feat
        mid_time_len = 0
        down_rate = 2
        for c in tcn_channels:
            if mid_time_len==0:
                mid_time_len = seq_len
            else:
                mid_time_len = mid_time_len/down_rate
            # 根据时间段长度比例，动态计算stride
            if mid_time_len>pred_len:
                stride = down_rate
            else:
                stride = 1
            layers.append(ResBlock2D(in_c, c, k, d=len(layers)+1,stride=(1,stride), drop=drop,act=act))
            in_c = c
        self.tcn = nn.Sequential(*layers)
        self.tcn_out_c = tcn_channels[-1] + static_feat + fut_feat

        # ========== 时间映射：SEQ_LEN → PRED_LEN ==========
        self.flat_dim = seq_len * n_channel * hist_feat + fut_feat
        
        # ========== 输出头 ==========
        self.out_seq_head = CausalConv2D(self.tcn_out_c, target_feat_dim,stride=(1,1), k=(3,3), drop=drop,act=act)    
        self.out_head = CausalConv2D(self.tcn_out_c, target_feat_dim, k=(3,3), stride=(1,self.pred_len),drop=drop,act=act)

    def forward(self, x_hist, x_stat, x_fut):
        """
        x_hist: [B, T, C, F_hist]  主序列+未来历史
        x_stat: [B, C, F_static]   静态协变量 ✅ 关键输入
        x_fut:  [B, Tp, C, F_fut]  未来协变量
        return: [B, Tp, C, F_main]
        """
        B = x_hist.shape[0]

        # --------------------- 1. 历史3D编码 ---------------------
        x_stat_his = x_stat.unsqueeze(1).repeat(1,self.seq_len,1,1) 
        x = x_hist.permute(0,1,3,2) 
        x = torch.cat([x,x_stat_his],-1) 
        x = x.permute(0,3,2,1) # [B,F_hist,C,seq_len]
        feat_hist = self.tcn(x)  # [B,F_hist,C,pred_len]
        
        if check_nan(feat_hist,"feat_hist"):
            sys.exit("nan exit")

        # --------------------- 静态协变量扩维到未来 ✅ ---------------------
        s = x_stat.unsqueeze(1).repeat(1,self.pred_len,1,1) 
        s = s.permute(0,3,2,1) 

        # --------------------- 未来协变量编码 ---------------------
        feat_fut = x_fut.permute(0,3,2,1)

        # ---------------------三源融合：历史 + 静态 + 未来 ✅ ---------------------
        fused_feat = torch.cat([feat_hist,s,feat_fut],dim=1)
        # --------------------- 输出 ---------------------
        
        out = self.out_head(fused_feat) # [B,target_dim,C,pred_len]
        out = out.permute(0,2,1,3)
        out_seq = self.out_seq_head(fused_feat) # [B,target_dim,C,pred_len]
        out_seq = out_seq.permute(0,2,3,1)
        return out_seq,out
    
    
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
        k=(3,3),
        dropout: float = 0.1,
        act=nn.GELU(),
        hidden_size=16,
        target_feat_dim=1,
    ):
        super().__init__()
        self.tcn_model = TCN2DPredictor(
                hist_feat=input_dim,static_feat=static_feat,fut_feat=fut_feat,
                n_channel=C,seq_len=seq_len,pred_len=pred_len,target_feat_dim=target_feat_dim,
                tcn_channels=tcn_channels,k=k,drop=dropout,act=act
            )      
        self.pred_len = pred_len         
        self.target_feat_dim = target_feat_dim  
        # 整合输出网络
        self.ins_layer = LinelessLayer(C,C,hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)
        self.ins_att_layer = LinelessLayer(C,C,hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)
        #CausalConv1D(C, C, k=1,stride=1, drop=0.2,act=act)  
        # self.ins_att_layer = LinelessLayer(C*pred_len,C,
        #                 hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)    
        # self.dec_layer = LinelessLayer(C*pred_len,C*pred_len,
        #                 hidden_size=hidden_size,layer_norm=True,batch_norm=False,dropout=0.3)     
        self.dec_layer = CausalConv2D(C, C, k=(1,1),stride=(1,1), drop=0.2,act=act)         
        # 指数整合输出网络       
        self.index_combine_layer = LinelessLayer(C*pred_len,pred_len)     
            
    def forward(
        self,main, static_covs,futures_convs
    ):    
        
        # 基础模型的向前传播
        y_pred,y_single = self.tcn_model(
            main, static_covs,futures_convs
        )   
        if check_nan(y_single,"y_single"):
            sys.exit("nan exit") 
        cls_out_combine = []
        index_data_combine = []
        # dec_out_combine = self.dec_layer(y_pred.permute(0,3,1,2).reshape(y_pred.shape[0],y_pred.shape[3],-1)).reshape(y_pred.shape)    
        # dec_out_combine = self.dec_layer(y_pred)
        dec_out_combine = y_pred
        # 主要比较目标输出
        # cls_out_ins = y_single.squeeze(-1)[:,:,0]
        # cls_out_ins = self.ins_layer(y_single).squeeze(-1)[:,:,0]
        # cls_out_combine.append(cls_out_ins)        
        # 品种间比较目标的网络输出
        for i in range(self.target_feat_dim):
            # 主要比较目标输出
            cls_out_ins = y_single[:,:,i,0]
            if i==0:
                cls_out_ins = self.ins_layer(cls_out_ins)
            else:
                cls_out_ins = self.ins_att_layer(cls_out_ins)
            cls_out_combine.append(cls_out_ins)
        # 整体指数预测的网络输出
        index_data_combine = self.index_combine_layer(y_pred[...,0].reshape(y_pred.shape[0],-1))    
         
        return dec_out_combine,cls_out_combine,index_data_combine     
    
    