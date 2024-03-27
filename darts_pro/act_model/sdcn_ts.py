import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torchmetrics
import numpy as np
from cus_utils.dink_util import aug_feature_shuffle,aug_feature_dropout
from .tide import Tide,Tide3D
from custom_model.embedding import embed
from cus_utils.common_compute import normalization,corr_compute,batch_cov
from torchmetrics.regression import ConcordanceCorrCoef

########################对于深度聚类的扩展,适配时间序列#######################
class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bn = nn.BatchNorm1d(num_features=out_features, momentum=0.3)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True,sparse=False):
        if sparse:
            support = torch.mm(features, self.weight)
            output = torch.spmm(adj, support)
            output = self.bn(output)   
        else:
            support = torch.matmul(features, self.weight)
            output = torch.bmm(adj, support)     
            output = self.bn(output.permute(0,2,1)).permute(0,2,1)  
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

    def forward(self, feat, adj, sparse=False):
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
        mode=1,
        **kwargs
       ):    
        super(SdcnTs, self).__init__()
        
        if mode==1:
            output_length = kwargs["output_chunk_length"]
        else:
            output_length = kwargs["input_chunk_length"]
            
        z_layer_dim = 10
        # Tide作为嵌入特征部分,输入和输出使用同一维度
        self.emb_layer = Tide(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,kwargs["input_chunk_length"],output_length,z_layer_dim=z_layer_dim,outer_mode=1)
        
        ###### GCN部分的定义，根据Tide模型，取得对应的分层encode数据
        n_input = self.emb_layer.encoder_dim
        gnn_dec_dim = decoder_output_dim * 4
        # 使用Tide模型第一层的实际输入维度
        self.gnn_1 = GNNLayer(n_input, gnn_dec_dim)
        # 这几个个均为Tide模型的后续encode层输入维度
        self.gnn_2 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        self.gnn_3 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        # 实际输出层，维度为预测序列长度
        self.gnn_4 = GNNLayer(gnn_dec_dim, z_layer_dim)
        # 对应分类层
        self.gnn_5 = GNNLayer(z_layer_dim, z_layer_dim)

        # 聚类部分设定
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, z_layer_dim))
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
            return x_bar,None,None,None,z
        
        tra1, tra2, tra3 = enc_data
        
        sigma = 0.5

        # GCN Module
        h1 = self.gnn_1(encoded_input_data, adj)
        h2 = self.gnn_2((1-sigma)*h1 + sigma*tra1, adj)
        h3 = self.gnn_3((1-sigma)*h2 + sigma*tra2, adj)
        # 使用GCN模块的实际输出，长度为预测序列长度
        h4 = self.gnn_4((1-sigma)*h3 + sigma*tra3, adj)
        # 分类输出
        h5 = self.gnn_5((1-sigma)*h4 + sigma*z, adj, active=False)
        
        # 自监督部分
        q = self.compute_qdis(z,mode=1)

        # 使用GCN的输出，再次进行聚类距离衡量，并返回数值
        pred = normalization(h5,mode="torch",axis=1)  
        pred = F.softmax(pred,1)
        # pred = self.compute_qdis(h5,mode=1) 

        return x_bar, q, pred, h5,z
    
    def compute_qdis(self,z,mode=1):
        if mode==1:
            # 使用MSE距离模式，计算与聚类簇心的距离
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()      
        else:
            # 使用相关系数作为计算模型,计算与聚类簇心的距离
            distance_q = corr_compute(self.cluster_layer,z)
            q = self.transfer_dis(distance_q)       
        return q  
    
    def transfer_dis(self,distance):
        q_pow = torch.pow(distance,2)
        qt = 1.0 / (1.0 + q_pow)
        q = (qt.t() / torch.sum(qt, 1)).t()
        return q     
    
    def ccc_distance_torch(self,x,y):
        flag_numpy = 0
        if isinstance(x,np.ndarray):
            flag_numpy = 1
            x = torch.Tensor(x)
            y = torch.Tensor(y)
        if len(x.shape)>1:
            x = x.transpose(1,0)
            y = y.transpose(1,0)
            concordance = ConcordanceCorrCoef(num_outputs=x.shape[-1]).to("cuda:0")
        else:
            concordance = ConcordanceCorrCoef().to("cuda:0")
        dis = 1 - concordance(x, y)
        if flag_numpy==1:
            dis = dis.cpu().numpy()
        return dis    
    
class SdcnTs3D(SdcnTs):
    """SDCNTS的3维版本"""
    
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
        mode=1,
        **kwargs
       ):    
        super(SdcnTs, self).__init__()
        
        if mode==1:
            output_length = kwargs["output_chunk_length"]
        else:
            output_length = kwargs["input_chunk_length"]
            
        z_layer_dim = 10
        # Tide作为嵌入特征部分,输入和输出使用同一维度
        self.emb_layer = Tide3D(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,kwargs["input_chunk_length"],output_length,z_layer_dim=z_layer_dim,outer_mode=1)
        
        ###### GCN部分的定义，根据Tide模型，取得对应的分层encode数据
        n_input = self.emb_layer.encoder_dim
        gnn_dec_dim = decoder_output_dim * 4
        # 使用Tide模型第一层的实际输入维度
        self.gnn_1 = GNNLayer(n_input, gnn_dec_dim)
        # 这几个个均为Tide模型的后续encode层输入维度
        self.gnn_2 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        self.gnn_3 = GNNLayer(gnn_dec_dim, gnn_dec_dim)
        # 实际输出层，维度为预测序列长度
        self.gnn_4 = GNNLayer(gnn_dec_dim, z_layer_dim)
        # 对应分类层
        self.gnn_5 = GNNLayer(z_layer_dim, n_cluster)

        # 聚类部分设定
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, z_layer_dim))
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
            return x_bar,None,None,None,z
        
        tra1, tra2, tra3 = enc_data
        
        sigma = 0.5

        # GCN Module
        h1 = self.gnn_1(encoded_input_data, adj)
        h2 = self.gnn_2((1-sigma)*h1 + sigma*tra1, adj)
        h3 = self.gnn_3((1-sigma)*h2 + sigma*tra2, adj)
        # 使用GCN模块的实际输出，长度为预测序列长度
        h4 = self.gnn_4((1-sigma)*h3 + sigma*tra3, adj)
        # 分类输出
        h5 = self.gnn_5((1-sigma)*h4 + sigma*z, adj, active=False)
        
        # 自监督部分
        # q = self.compute_qdis(z,mode=1)
        q = None
        # 使用GCN的输出，再次进行聚类距离衡量，并返回数值
        pred = normalization(torch.reshape(h5,(h5.shape[0]*h5.shape[1],h5.shape[2])),mode="torch",axis=1)
        # pred = F.softmax(pred,1)
        # pred = self.compute_qdis(h5,mode=1) 
        # pred = batch_cov(h5)

        return x_bar, q, pred, h5,z            
    