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

class ActEncoder(nn.Module):
    """内部序列编码器"""
    
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, dropout, block = 'LSTM'):

        super(ActEncoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.block = block
        
        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """

        _, (h_end, c_end) = self.model(x)
        
        if self.block == 'LSTM':
            h_end = h_end[-1, :, :]
        return h_end


class ActDecoder(nn.Module):
    """内部序列解码器"""
    
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, block='LSTM',device='cpu'):

        super(ActDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = torch.DoubleTensor
        self.device = device
        
        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)

        self.sig = nn.Sigmoid()
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        
    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        batch_size = h_state.shape[0]
        decoder_inputs = torch.zeros(self.sequence_length, batch_size, 1, requires_grad=True).type(self.dtype).to(self.device)
        c_0 = torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size, requires_grad=True).type(self.dtype).to(self.device)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_inputs, (h_0, c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        out = self.sig(out)
        
        return out
            
class Encoder(nn.Module):
    def __init__(self,input_dim=0,input_length=25,hidden_size=64,latent_length=4):
        super(Encoder,self).__init__()
        
        # 使用GRU作为内部编码器
        self.encoder = ActEncoder(number_of_features=input_dim, # 输入的特征维度
                               hidden_size=hidden_size,
                               hidden_layer_depth=2,
                               dropout=0.3,
                               block='GRU')

        self.mu_l=nn.Linear(hidden_size,latent_length)
        self.log_sigma2_l=nn.Linear(hidden_size,latent_length)

    def forward(self, x):
        e = self.encoder(x.permute(1,0,2))

        mu = self.mu_l(e)
        log_sigma2=self.log_sigma2_l(e)

        return mu,log_sigma2

class Decoder(nn.Module):
    def __init__(self,sequence_length=5,batch_size=1,latent_length=4,device='cpu',block='GRU'):
        super(Decoder,self).__init__()

        self.decoder=ActDecoder(sequence_length=sequence_length,
                               batch_size=batch_size,
                               hidden_size=64,
                               hidden_layer_depth=2,
                               latent_length=latent_length,
                               output_size=1,
                               block=block,
                               device=device)

    def forward(self, z):
        x_pro=self.decoder(z)
        return x_pro
    
class VaDE(nn.Module):
    """融合VADE以及Tide模式的序列处理，用于时间序列聚类方式预测"""
    
    def __init__(self, 
        input_dim: int,
        input_length: int,
        output_length: int,
        n_cluster=4,
        batch_size=1,
        num_nodes=100,
        static_feat=None,
        device='cpu',
        **kwargs
       ):    
        super(VaDE, self).__init__()

        self.n_cluster = n_cluster 
        self.input_dim = input_dim
        self.hidden_size = 64
        
        self.encoder = Encoder(input_dim=input_dim,input_length=input_length,hidden_size=self.hidden_size,latent_length=n_cluster)
        self.decoder=Decoder(sequence_length=output_length,latent_length=n_cluster,batch_size=batch_size,device=device)
                
        ###### VADE部分的定义
        self.pi_ = torch.Tensor(np.array([0.2,0.4,0.3,0.1])) .to(device) # nn.Parameter(torch.FloatTensor(n_cluster).fill_(1)/n_cluster,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(n_cluster,n_cluster).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(n_cluster,n_cluster).fill_(0),requires_grad=True)

    def forward(self, x, target=None,mode="pretrain"):
        """先提取序列特征，然后根据中间变量做聚类，然后做自监督计算
           根据mode参数分为2种模式，pretrain表示只进行特征提取
        """
        
        # pretrain模式返回中间步骤的特征值
        x_dynamic_past_covariates = x[...,:self.input_dim]
        if mode=="pretrain":
            z, mu = self.encoder(x_dynamic_past_covariates)
            x_bar = self.decoder(z)                        
            return x_bar,mu,None,None,None
        
        z_mu, z_sigma2_log = self.encoder(x_dynamic_past_covariates)
        
        # 在这里进行综合loss的计算,由于需要应用到预测流程，因此使用target代替x
        loss,detail_loss,yita_c = self.ELBO_compute(target, z_mu, z_sigma2_log)
        
        return None, z_mu, z_sigma2_log, None,(loss,detail_loss,yita_c)
            
    def ELBO_compute(self,x,z_mu, z_sigma2_log,L=1):
        
        det=1e-10
        L_rec=0
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)
            x_pro = x_pro.permute(1,0,2).squeeze()
            # 与预测结果进行比较
            L_rec+=F.binary_cross_entropy(x_pro,x)
            # print("L_rec:{},x_pro max:{},x_pro min:{},x.max():{},x.min():{}".format(L_rec,x_pro.max(),x_pro.min(),x.max(),x.min()))
        # 对应重构损失
        L_rec/=L
        
        Loss=L_rec # *x.size(1)

        # 以下几个部分对应KL散度损失,即：高斯混合先验=>变分后验的KL散度
        pi = self.pi_ # F.softmax(self.pi_)
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        # 高斯混合先验的p(z|c)以及p(c)部分
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))

        # 变分后验的q(z,c|x)部分
        loss_xz = torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1)
        loss_xz = 0.5*torch.mean(loss_xz)
        Loss += loss_xz
        
        loss_xc = torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1)) + 0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        Loss -= loss_xc
        
        # 单独记录详细损失
        detail_loss = (L_rec,loss_xz,loss_xc)   
             
        return Loss,detail_loss,yita_c
    
    def predict(self,x):
        x_dynamic_past_covariates = x[...,:self.input_dim]
        z_mu, z_sigma2_log = self.encoder(x_dynamic_past_covariates)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1),None


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))


