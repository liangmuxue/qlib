import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
import os
from .tide_vade import TideVaDE

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum(w[ind[0],ind[1]])*1.0/Y_pred.size, w


def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class Encoder(nn.Module):
    def __init__(self,input_dim=784,inter_dims=[500,500,2000],hid_dim=10):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]),
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )

        self.mu_l=nn.Linear(inter_dims[-1],hid_dim)
        self.log_sigma2_l=nn.Linear(inter_dims[-1],hid_dim)

    def forward(self, x):
        e=self.encoder(x)

        mu=self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2

class Decoder(nn.Module):
    def __init__(self,input_dim=784,inter_dims=[500,500,2000],hid_dim=10):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(hid_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_pro=self.decoder(z)
        return x_pro
    
class VaDE(nn.Module):
    """融合VADE以及Tide模式的序列处理，用于时间序列聚类方式预测"""
    
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
        num_nodes=100,
        static_feat=None,
        device='cpu',
        **kwargs
       ):    
        super(VaDE, self).__init__()

        self.n_cluster = n_cluster 
        z_layer_dim = n_cluster
        # Tide作为嵌入特征部分,输入和输出使用同一维度
        self.emb_layer = TideVaDE(input_dim,emb_output_dim,future_cov_dim,static_cov_dim,nr_params,num_encoder_layers,num_decoder_layers,
                              decoder_output_dim,hidden_size,temporal_decoder_hidden,temporal_width_past,temporal_width_future,
                              use_layer_norm,dropout,
                              input_chunk_length=kwargs["input_chunk_length"],output_chunk_length=kwargs["input_chunk_length"],
                              z_layer_dim=z_layer_dim,outer_mode=1)

        enc_input_dim = (self.emb_layer.past_cov_dim+1)*kwargs["input_chunk_length"]
        inter_dims = [100,100,300]
        self.encoder=Encoder(input_dim=enc_input_dim,hid_dim=n_cluster,inter_dims=inter_dims)
        self.decoder=Decoder(input_dim=enc_input_dim,hid_dim=n_cluster,inter_dims=inter_dims)
                
        ###### VADE部分的定义
        self.pi_ = nn.Parameter(torch.FloatTensor(n_cluster,).fill_(1)/n_cluster,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(n_cluster).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(n_cluster).fill_(0),requires_grad=True)

    def forward(self, x_in, idx=None,rank_mask=None,mode="pretrain"):
        """先提取序列特征，然后根据中间变量做聚类，然后做自监督计算
           根据mode参数分为2种模式，pretrain表示只进行特征提取
        """
        
        # 获取嵌入特征，包含中间过程结果
        # x_bar,mu,log_sigma2,x_sig,x_lookback = self.emb_layer(x)
        # x_bar = x_bar.squeeze(-1)
        # x_sig = x_sig.squeeze(-1)
        
        # pretrain模式返回中间步骤的特征值
        x, x_future_covariates, x_static_covariates = x_in
        x_dynamic_past_covariates = x[...,:self.emb_layer.past_cov_dim+1]
        x_reshape = x_dynamic_past_covariates.reshape(x_dynamic_past_covariates.shape[0]*x_dynamic_past_covariates.shape[1],x_dynamic_past_covariates.shape[2]*x_dynamic_past_covariates.shape[3])
        if mode=="pretrain":
            
            z, mu = self.encoder(x_reshape)
            x_bar = self.decoder(z)                        
            return x_bar,mu,None,None,None
        
        z_mu, z_sigma2_log = self.encoder(x_reshape)
        
        # 在这里提前进行综合loss的计算
        loss = self.ELBO_compute(x_reshape, z_mu, z_sigma2_log)
        
        return None, z_mu, z_sigma2_log, None,loss
            
    def ELBO_compute(self,x,z_mu, z_sigma2_log,L=1):
        
        det=1e-10
        L_rec=0
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)
            L_rec+=F.binary_cross_entropy(x_pro,x)
            # print("L_rec:{},x_pro max:{},x_pro min:{},x.max():{},x.min():{}".format(L_rec,x_pro.max(),x_pro.min(),x.max(),x.min()))

        L_rec/=L

        Loss=L_rec*x.size(1)

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))

        return Loss
    
    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1)


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))


