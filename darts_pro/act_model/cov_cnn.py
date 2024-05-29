import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet

from darts.models.forecasting.tide_model import _ResidualBlock

from cus_utils.metrics import pca_apply

class LineClassify(nn.Module):
    """简单线性分类器，用于对预测结果指标进行分类"""
    
    def __init__(self, input_dim, output_dim):
        super(LineClassify, self).__init__()
        
        self.lin_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.lin_layer(x)
        return x

class CovCnn(nn.Module):
    """使用协方差矩阵数据，以及卷积网络进行计算"""
    
    def __init__(self, 
        input_dim=1,
        batch_size=1024,
        hidden_size=32,
        n_cluster=4,
        num_encoder_layers=3,
        dropout=0.1,
        use_layer_norm=False,
        device='cpu',
        **kwargs
       ):    
        super(CovCnn, self).__init__()
        
        self.n_cluster = n_cluster 
        self.input_dim = input_dim
        self.batch_size = batch_size
        
        # 使用带残差的多层感知机
        self.mlp_layer = nn.Sequential(
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
                    output_dim=hidden_size, # 输出维度为1，以进行目标匹配
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=1,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),    
        )
        # 分类层，输入维度为批次大小
        self.classify_layer = nn.Linear(batch_size, n_cluster)
        # 目标数据分类层，用于对降维后的预测目标进行分类
        self.target_classify = LineClassify(input_dim=2,output_dim=n_cluster)      

    def forward(self, x,pca_target=None):
        """输入数据为协方差矩阵数据"""

        # 取得感知机结果后，把维数从3降为2
        features = self.mlp_layer(x).squeeze()
        if features.shape[1]<self.batch_size:
            pad = nn.ZeroPad2d(padding=(0, self.batch_size-features.shape[1], 0, 0))
            cls = self.classify_layer(pad(features))
        else:
            # 进行分类计算
            cls = self.classify_layer(features)
        # 对特征数据进行PCA降维，后续和同样降维的目标值进行比较
        fea_pca = pca_apply(features,2)
        # 返回pca数据，以及隐含分类数据
        return cls,features,fea_pca
    
class PcaCnn(nn.Module):
    """使用降维模式数据，以及卷积网络进行计算"""
    
    def __init__(self, 
        pca_dim=4,
        input_dim=1,
        output_dim=2,
        hidden_size=32,
        n_cluster=4,
        num_encoder_layers=3,
        dropout=0.3,
        use_layer_norm=False,
        device='cpu',
        **kwargs
       ):    
        super(PcaCnn, self).__init__()
        
        self.n_cluster = n_cluster 
        self.input_dim = input_dim
        
        # 使用带残差的多层感知机
        self.mlp_layer = nn.Sequential(
            _ResidualBlock(
                input_dim=input_dim*pca_dim, # 合并输入维度
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size, # 输出维度为1，以进行目标匹配
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=output_dim,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),    
        )
        # 分类层，输入维度为批次大小
        self.classify_layer = nn.Linear(output_dim, n_cluster)
        # 目标数据分类层，用于对降维后的预测目标进行分类
        self.target_classify = LineClassify(input_dim=2,output_dim=n_cluster)      

    def forward(self, x,pca_target=None):
        """输入数据为降维后的数据"""

        # 合并维数
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        # 取得感知机结果后
        features = self.mlp_layer(x)
        # 进行分类计算
        cls = self.classify_layer(features)
        # 返回pca数据，以及隐含分类数据
        return cls,features,None
    