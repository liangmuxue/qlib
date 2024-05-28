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

class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
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


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output,mode="pretrain"):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training and mode!="pretrain":
            # 全训练模式使用重采样方式
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            # 预训练模式使用原方式
            return self.latent_mean

class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, block='LSTM',device='cpu'):

        super(Decoder, self).__init__()

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
    
class VaRE(nn.Module):
    """融合VADE以及Recurrent模式的序列处理，用于时间序列聚类方式预测"""
    
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
        batch_size=1,
        n_cluster=4,
        sequence_length=5,
        static_feat=None,
        device='cpu',
        **kwargs
       ):    
        super(VaRE, self).__init__()
        
        hidden_layer_depth = 2
        block = 'GRU'
        self.encoder = Encoder(number_of_features=input_dim, # 输入的特征维度
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=n_cluster,
                               dropout=dropout,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=n_cluster)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=n_cluster,
                               output_size=1,
                               block=block,
                               device=device)
        
        # 序列长度
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        # 隐变量维度，相当于聚类类别数
        self.latent_length = n_cluster

        ###### VADE部分的定义
        self.pi_ = nn.Parameter(torch.FloatTensor(n_cluster,).fill_(1)/n_cluster,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(n_cluster,n_cluster).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(n_cluster,n_cluster).fill_(0),requires_grad=True)
        self.n_cluster = n_cluster
        
    def forward(self, x,mode="pretrain"):
        """使用LSTM作为编码器，同时兼顾聚类模式"""
        
        # 获取嵌入特征，包含中间过程结果
        # x_bar,mu,log_sigma2,x_sig,x_lookback = self.emb_layer(x)
        # x_bar = x_bar.squeeze(-1)
        # x_sig = x_sig.squeeze(-1)
        
        # 批次号在中间
        x = x.permute(1,0,2)
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output,mode=mode)
        # 取得隐变量的均值变量和方差变量
        z_mu = self.lmbd.latent_mean
        z_sigma2_log = self.lmbd.latent_logvar
        
        x_decoded = self.decoder(latent)        
        # 预训练模式，只返回解码数据进行重建误差评估
        if mode=="pretrain":
            return x_decoded,z_mu,z_sigma2_log,None,None
        
        # 在这里提前进行综合loss的计算(elbo)
        elbo = self.ELBO_compute(x_decoded,x)
        
        return x_decoded, z_mu, z_sigma2_log, None,elbo
            
    def ELBO_compute(self,x_decoded,x):
        
        loss_combine = []
        x = x.permute(1,0,2)
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        return kl_loss
    
    def predict(self,x):
        x = x.permute(1,0,2)
        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output,mode="complete")
        # 取得隐变量的均值变量和方差变量
        z_mu = self.lmbd.latent_mean
        z_sigma2_log = self.lmbd.latent_logvar
        
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1),(yita,z.cpu().numpy(),latent.cpu().numpy(),cell_output.cpu().numpy())


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def ccc_distance(self,input_ori,target_ori):
        if len(input_ori.shape)==1:
            input_with_dims = input_ori.unsqueeze(0)
        else:
            input_with_dims = input_ori
        if len(target_ori.shape)==1:
            target_with_dims = target_ori.unsqueeze(0)    
        else:
            target_with_dims = target_ori                    
        input = input_with_dims.flatten()
        target = target_with_dims.flatten()
        corr_tensor = torch.stack([input,target],dim=0)
        cor = torch.corrcoef(corr_tensor)[0][1]
        var_true = torch.var(target)
        var_pred = torch.var(input)
        sd_true = torch.std(target)
        sd_pred = torch.std(input)
        numerator = 2*cor*sd_true*sd_pred
        mse_part = self.mse_dis(input_with_dims,target_with_dims)
        denominator = var_true + var_pred + mse_part
        ccc = numerator/denominator
        ccc_loss = 1 - ccc
        return ccc_loss  

    def mse_dis(self,input, target):
        loss_arr = (input - target) ** 2
        distance = torch.mean(loss_arr,dim=1)
        return distance   
    