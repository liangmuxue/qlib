import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class Encoder(nn.Module):
    def __init__(self,input_dim=0,input_length=25,hidden_size=64,latent_length=4):
        super(Encoder,self).__init__()
        
        # 使用GRU作为内部编码器
        self.encoder = ActEncoder(number_of_features=input_dim, # 输入的特征维度
                               hidden_size=hidden_size,
                               hidden_layer_depth=2,
                               dropout=0.3,
                               block='GRU')
        self.lat = nn.Linear(hidden_size,latent_length)

    def forward(self, x):
        e = self.encoder(x.permute(1,0,2))
        lat = self.lat(e)
        return e,lat