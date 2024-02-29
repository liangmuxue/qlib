import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import Optional, Tuple
import math
import dgl.function as fn
import dgl
from dgl.nn.pytorch import GraphConv,SGConv

from cus_utils.dink_util import aug_feature_shuffle,aug_feature_dropout
from .tide import Tide
from custom_model.embedding import embed

########################对于深度聚类的扩展,适配时间序列#######################
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True,sparse=True):
        support = torch.mm(features, self.weight)
        if sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.bmm(adj, support)        
        if active:
            output = F.relu(output)
        return output
    
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.FloatTensor(out_ft))

        # init parameters
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        self.bias.data.fill_(0.0)

    def forward(self, feat, adj, sparse=True):
        h = self.fc(feat)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(h, 0)), 0)
        else:
            out = torch.bmm(adj, h)
        out += self.bias
        return self.act(out)


class SdcnTs(nn.Module):
    """融合SDCN以及Tide模式的序列处理，用于时间序列聚类方式预测"""
    
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
        v=1,
        device=None,
        **kwargs
       ):    
        super(SdcnTs, self).__init__()
        # Tide作为嵌入特征部分,输入和输出使用同一维度
        self.emb_layer = Tide(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,kwargs["input_chunk_length"],kwargs["input_chunk_length"],outer_mode=1)
        
        ###### GCN部分的定义，根据Tide模型，取得对应的分层encode数据
        n_input = self.emb_layer.encoder_dim
        gnn_dec_dim = decoder_output_dim * 4
        # 使用Tide模型第一层的实际输入维度
        self.gnn_1 = GNNLayer(n_input, gnn_dec_dim)
        # 这几个个均为Tide模型的后续encode层输入维度
        self.gnn_2 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        self.gnn_3 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        self.gnn_4 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        # 对应分类层
        self.gnn_5 = GNNLayer(gnn_dec_dim, n_cluster)

        # 聚类部分设定
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, gnn_dec_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v     

    def forward(self, x, adj,mode="pretrain"):
        """先提取序列特征，然后根据中间变量做GCN，然后做自监督计算
           根据mode参数分为2种模式，pretrain表示只进行特征提取
        """
        
        # 获取嵌入特征，包含中间过程结果
        x_bar,z,enc_data,encoded_input_data = self.emb_layer(x)
        x_bar = x_bar.squeeze()
        # pretrain模式只需要中间步骤的特征值
        if mode=="pretrain":
            return x_bar,None,None,z
        
        tra1, tra2, tra3 = enc_data
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(encoded_input_data, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*tra2, adj)
        h = self.gnn_4((1-sigma)*h + sigma*tra3, adj)
        h = self.gnn_5((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, None


  
    