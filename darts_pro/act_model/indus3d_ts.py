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
from darts.models.forecasting.tide_model import _ResidualBlock,Dense
from .fds import FDS

class _Residual3DBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
        ins_dim: int, # 横向数量维度
        ins_hidden_size=128, # 横向全连接的隐含层维度     
    ):
        """增加一个维度，拓展全连接方向"""
        super().__init__()

        # 标准含残差的mlp单元
        self.forward_block = _ResidualBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        # 用于横向计算的含残差的mlp单元 
        self.cross_block = _ResidualBlock(
            input_dim=ins_dim,
            output_dim=ins_dim,
            hidden_size=ins_hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.forward_block(x)
        if len(x.shape)==3:
            x = x.permute(0,2,1)
        else:
            x = x.permute(0,2,3,1)
        x = self.cross_block(x)
        if len(x.shape)==3:
            x = x.permute(0,2,1)
        else:
            x = x.permute(0,3,1,2)        
        return x

class Indus3D(nn.Module):
    """整合行业数据，形成包含整合时间维度的3D版本的MLP网络"""
    
    def __init__(self, 
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        num_encoder_layers=3,
        num_decoder_layers=3,
        decoder_output_dim=1,
        hidden_size=64,
        temporal_decoder_hidden=32,
        temporal_width_past=None,
        temporal_width_future=None,
        use_layer_norm=True,
        dropout=0.3,
        ins_dim=0,
        sw_ins_mappings=None,
        **kwargs
       ):    
        """整合行业数据形成多层次网络输出
           Params:
               input_dim： 输入特征维度（总协变量数量）
               output_dim： 输出特征维度，一般为1
               future_cov_dim： 未来协变量数量维度
               temporal_width_past： 过去协变量投影维度
               sw_ins_mappings: 行业分类和股票的映射关系,数组类型，数组内长度为行业分类数量，每个分类包含所属成份股票索引列表
        """
        
        super(Indus3D, self).__init__()
        
        output_chunk_length = kwargs["output_chunk_length"]
        input_chunk_length = kwargs["input_chunk_length"]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.ins_dim = ins_dim
        self.sw_ins_mappings = sw_ins_mappings
        
        # past covariates handling: either feature projection, raw features, or no features
        self.past_cov_projection = None
        if self.past_cov_dim and temporal_width_past:
            # residual block for past covariates feature projection
            self.past_cov_projection = _ResidualBlock(
                input_dim=self.past_cov_dim,
                output_dim=temporal_width_past,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            past_covariates_flat_dim = self.input_chunk_length * temporal_width_past
        elif self.past_cov_dim:
            # skip projection and use raw features
            past_covariates_flat_dim = self.input_chunk_length * self.past_cov_dim
        else:
            past_covariates_flat_dim = 0

        # future covariates handling: either feature projection, raw features, or no features
        self.future_cov_projection = None
        if future_cov_dim and self.temporal_width_future:
            # residual block for future covariates feature projection
            self.future_cov_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * temporal_width_future
        elif future_cov_dim:
            # skip projection and use raw features
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0

        encoder_dim = (
            self.input_chunk_length * output_dim
            + past_covariates_flat_dim
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )
        # Set Attr For Outer Using
        self.encoder_dim = encoder_dim
        
        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
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
        # 标准未来数据解码器
        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim
                * self.output_chunk_length,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )
        # 未来数据走势残差单元
        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length
        )           
        decoder_input_dim = decoder_output_dim
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim
        # 解码器投影单元
        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=False, # 这里不能使用layer norm,否则输出值都为0
            dropout=dropout,
        )
        
        ###### 整合行业分类，形成分类预测和类内成份股票走势预测的多重输出网络 ######
        industry_layer = []
        ins_ind_layer = []
        for i in range(len(sw_ins_mappings)):
            ins_num = len(sw_ins_mappings[i])
            ins_decoder = _Residual3DBlock(
                        input_dim=decoder_input_dim,
                        output_dim=1,
                        ins_dim=ins_num, # 使用当前分类内的成份数量作为横向全连接数量
                        hidden_size=temporal_decoder_hidden,
                        use_layer_norm=True,
                        dropout=dropout,
                    )     
            # 每个行业内的股票成份走势
            ins_ind_layer.append(ins_decoder)
            # 行业整体走势
            indus_layer = nn.Linear(
                ins_num, 1
            )    
            industry_layer.append(indus_layer)
        self.ins_ind_layer = nn.ModuleList(ins_ind_layer)
        self.industry_layer = nn.ModuleList(industry_layer)
        # 行业总体走势残差单元，使用行业过去数值映射未来走势（1段）
        self.industry_lookback_skip = nn.Linear(
            self.input_chunk_length, 1
        )  
                
    def forward(self, x_in):
        """包括多个不同维度的传播"""

        # x has shape (batch_size, ins_dim,input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, ins_dim,input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, ins_dim,static_cov_dim)
        x, x_future_covariates, x_static_covariates,x_industry_past_values = x_in

        x_lookback = x[:, :, :, : self.output_dim]

        # future covariates: feature projection or raw features
        # historical future covariates need to be extracted from x and stacked with part of future covariates
        if self.future_cov_dim:
            x_dynamic_future_covariates = torch.cat(
                [
                    x[...,
                        None if self.future_cov_dim == 0 else -self.future_cov_dim :,
                    ],
                    x_future_covariates,
                ],
                dim=2,
            )
            if self.temporal_width_future:
                # project input features across all input and output time steps
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        # past covariates: feature projection or raw features
        # the past covariates are embedded in `x`
        if self.past_cov_dim:
            x_dynamic_past_covariates = x[
                ...,
                self.output_dim : self.output_dim + self.past_cov_dim,
            ]
            if self.temporal_width_past:
                # project input features across all input time steps
                x_dynamic_past_covariates = self.past_cov_projection(
                    x_dynamic_past_covariates
                )
        else:
            x_dynamic_past_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,
            x_dynamic_past_covariates,
            x_dynamic_future_covariates,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=2) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=2)
        # 编码器使用标准模式，对后续的解码器共用
        for i in range(len(self.encoders)):
            encoded = self.encoders[i](encoded)    
        
        # 解码器分为2个分支，分别对应未来输出目标（多段），以及未来数据统一评判（1段）  
        
        # 这里为标准输出解码，时间段和特征维度混合在一起
        decoded_flatten = self.decoders(encoded)

        # 分解为：未来时间段*特征维度
        decoded = decoded_flatten.view(x.shape[0], x.shape[1],self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            x_dynamic_future_covariates[:,:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=-1)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # 走势残差计算
        skip = self.lookback_skip(x_lookback.transpose(2, 3)).transpose(2, 3)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )
        y = y.view(-1, x.shape[1],self.output_chunk_length, self.output_dim)
        
        ### 行业整合解码部分 ###
        
        map_size = len(self.ins_ind_layer)
        # 切分成每个行业分类，在分类内对成份股票进行解码
        ins_decoded_data = torch.zeros([y.shape[0],y.shape[1]]).to(self.device)
        industry_decoded_data = []
        for i in range(map_size):
            m = self.ins_ind_layer[i]
            idx_list = self.sw_ins_mappings[i]
            # 通过索引映射到实际行业内股票组合
            ins_data = encoded[:,idx_list,...]
            ins_decoded = m(ins_data)
            ins_decoded_data[:,idx_list] = ins_decoded
            # 进一步进行整体行业计算
            m = self.industry_layer[i]
            industry_decoded_data.append(m(ins_decoded))
            
        industry_decoded_data = torch.cat(industry_decoded_data,dim=1)
        # 使用行业分类的过去数值范围进行残差计算
        skip = self.industry_lookback_skip(x_industry_past_values.transpose(1, 2)).transpose(1, 2)
        indus_sv = industry_decoded_data + skip
        
        return y,ins_decoded_data,indus_sv
                                      
                                      