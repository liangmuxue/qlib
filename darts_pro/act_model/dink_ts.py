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

########################对于Dink-Net的扩展,适配时间序列#######################

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


class DinkTsNet(nn.Module):
    
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
        device=None,
        **kwargs
       ):    
        super(DinkTsNet, self).__init__()
        # Tide作为嵌入特征部分
        self.emb_layer = Tide(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,kwargs["input_chunk_length"],kwargs["output_chunk_length"])
        # 图卷积及聚类部分设定
        self.n_cluster = kwargs["n_cluster"]
        # 使用簇心参数，形状为:类别数*预测时间步长*
        self.cluster_center = torch.nn.Parameter(torch.Tensor(self.n_cluster, kwargs["n_out"]))
        self.gcn = GCN(kwargs["n_in"], kwargs["n_out"],kwargs["activation"])
        self.lin = nn.Linear(kwargs["n_out"], kwargs["n_out"])
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = 1e-9        

    def forward(self, x, adj):
        embed = self.embed(x, adj, sparse=False)
        return embed

    def embed(self, x, adj, power=5, sparse=True):
        # 先使用Tide模式(包含Seq2Seq)获取序列特征，输出为:batch_size*预测时间步长*特征目标数
        x = self.emb_layer(x)
        # GCN生成进一步的特征
        local_h = self.gcn(x, adj, sparse)
        global_h = local_h.clone().squeeze(0)
        # 嵌入特征计算生成
        for i in range(power):
            global_h = adj @ global_h
        global_h = global_h.unsqueeze(0)
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])
        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)        
        return h

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, h):
        """计算聚类损失
            h: 经过特征化(embedding)的数据
        """
        
        sample_center_distance = self.dis_fun(h, self.cluster_center)
        center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
        self.no_diag(center_distance, self.cluster_center.shape[0])
        loss = sample_center_distance.mean() - center_distance.mean()

        return loss, sample_center_distance

    def clustering(self, x, adj, finetune=True):
        h = self.embed(x, adj, sparse=True)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).cpu().detach().numpy()
        return cluster_results

  
    