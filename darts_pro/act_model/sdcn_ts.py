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
from cus_utils.metrics import pca_apply
from torchmetrics.regression import ConcordanceCorrCoef
from darts_pro.act_model.mtgnn_layer import graph_constructor

########################使用协方差及降维模式,适配时间序列#######################

class LineClassify(nn.Module):
    """简单线性分类器，用于对预测结果指标进行分类"""
    
    def __init__(self, input_dim, output_dim):
        super(LineClassify, self).__init__()
        
        self.lin_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.lin_layer(x)
        return x
        
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
        self.lat = nn.Linear(hidden_size,latent_length)

    def forward(self, x):
        e = self.encoder(x.permute(1,0,2))
        lat = self.lat(e)
        return e,lat
    
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
        x_pro = x_pro.permute(1,0,2).squeeze()
        return x_pro
    
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    """对于时间序列进行降维，以实现多样本的横向比较"""
    
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
    
class SdcnTs(nn.Module):
    """融合GRU模式的序列处理，并在中间层实现关系比较"""
    
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
        super(SdcnTs, self).__init__()
        
        self.n_cluster = n_cluster 
        self.input_dim = input_dim
        self.hidden_size = 64
        
        self.encoder = Encoder(input_dim=input_dim,input_length=input_length,hidden_size=self.hidden_size,latent_length=n_cluster)
        self.decoder = Decoder(sequence_length=output_length,latent_length=n_cluster,batch_size=batch_size,device=device)
        # 目标数据分类层，用于对降维后的预测目标进行分类
        self.target_classify = LineClassify(input_dim=2,output_dim=n_cluster)      

    def forward(self, x,pca_target=None,mode="pretrain"):
        """先提取序列特征，然后根据中间变量做位置匹配，pretrain表示只进行特征提取"""

        # GRU模式下，舍弃静态变量        
        x_dynamic_past_covariates = x[...,:self.input_dim]
        z,lattend = self.encoder(x_dynamic_past_covariates)
        x_bar = self.decoder(lattend) 
        
        # pretrain模式只返回解码的特征值
        if mode=="pretrain":
            # 预训练阶段,针对目标数据，进行简单线性分类，提取出分割参考区域
            output_class = self.target_classify(pca_target)            
            return x_bar,None,None,None,output_class
        
        # 对编码数据进行PCA降维，后续和同样降维的目标值进行比较
        z_pca = pca_apply(z,2)
        # 正式阶段，使用降维后的变量进行分类预测,此步骤不反向传播
        with torch.no_grad():
            output_class = self.target_classify(z_pca)  
        # 返回pca数据，以及隐含分类数据
        return x_bar, z_pca,lattend,None,output_class
    
    def predict_pca_cls(self,pca_data):
        cls = self.target_classify(pca_data)
        cls = cls.detach().cpu().numpy()
        return np.argmax(cls,axis=1),(self.target_classify.lin_layer.weight.data.cpu().numpy(),self.target_classify.lin_layer.bias.data.cpu().numpy())
    
    