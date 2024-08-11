import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchmetrics
import numpy as np
from .tide import Tide,Tide3D
from custom_model.embedding import embed
from cus_utils.common_compute import eps_rebuild,corr_compute,batch_cov
from torchmetrics.regression import ConcordanceCorrCoef
from darts.models.forecasting.tide_model import _ResidualBlock
from .fds import FDS

class LineClassify(nn.Module):
    """线性分类器，用于对预测结果指标进行分类"""
    
    def __init__(self, input_dim, output_dim=1,hidden_size=64,dropout=0.5,use_layer_norm=True,num_encoder_layers=3):
        super(LineClassify, self).__init__()
        
        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=input_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],          
        )
        self.lin_layer = nn.Linear(hidden_size,output_dim)
        
    def forward(self, x):
        x = self.encoders(x)
        x = self.lin_layer(x)
        return x
    
class MlpTs(nn.Module):
    """Encoder-Decoder框架，并基于DNN模式的时间序列预测"""
    
    def __init__(self, 
        input_dim: int,
        emb_output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_past: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        n_cluster=4,
        pca_dim=2,
        enc_nr_params=1,
        **kwargs
       ):    
        super(MlpTs, self).__init__()
        
        output_length = kwargs["output_chunk_length"]
        input_length = kwargs["input_chunk_length"]
        
        # Tide作为嵌入特征部分,输入和输出使用同一维度
        self.emb_layer = Tide(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,input_length,output_length,z_layer_dim=pca_dim,outer_mode=1)
        
        # 目标数据分类层，用于对降维后的预测目标进行分类
        self.target_classify = LineClassify(input_dim=pca_dim,output_dim=n_cluster)    
        # 使用分位数回归模式，所以输出维度为分位数数量
        self.z_classify = LineClassify(input_dim=hidden_size,output_dim=enc_nr_params)   
        # self.regressor = nn.Linear(pca_dim, 1)
        self.regressor = LineClassify(input_dim=pca_dim,output_dim=1)  
        self.FDS = FDS(feature_dim=pca_dim,bucket_num=n_cluster,bucket_start=0)
        # 降维后拟合高斯分布的均值和方差
        # self.mu_c = nn.Parameter(torch.FloatTensor([0.2,0,-0.08,-0.2]),requires_grad=False)
        self.mu_c = torch.FloatTensor([0.2,0,-0.08,-0.2])
        # self.sigma2_c = nn.Parameter(torch.FloatTensor([0.15,0.2,0.2,0.15]),requires_grad=False)
        self.sigma2_c = torch.FloatTensor([0.15,0.2,0.2,0.15])
        
    def forward(self, x):
        """先提取序列特征，然后根据中间变量做GCN，然后做自监督计算
           根据mode参数分为2种模式，smooth表示使用平滑特征模式
        """
        
        # 获取嵌入特征，包含中间过程结果
        x_bar,z,encoded,encoded_input_data = self.emb_layer(x)
        x_bar = x_bar.squeeze(-1).squeeze(-1)
        encoding_s = z
        cls = self.z_classify(encoded)
        x_smo = self.regressor(z)
        # 执行目标分类，用于辅助输出分类
        # tar_cls = self.target_classify(pca_target)
        return x_bar,encoding_s,cls,None,x_smo
    
    def predict_pca_cls(self,pca_data):
        
        real_classify = self.z_classify
        cls = real_classify(pca_data)
        cls = cls.detach().cpu().numpy()
        return np.argmax(cls,axis=1),(real_classify.lin_layer.weight.data.cpu().numpy(),real_classify.lin_layer.bias.data.cpu().numpy())   
    
    