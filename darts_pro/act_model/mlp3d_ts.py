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
    
class MlpTs3D(nn.Module):
    """增加时间维度，形成3D版本的MLP网络"""
    
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
        **kwargs
       ):    
        """同时兼顾时间维度和股票间的关系维度
           Params:
               input_dim： 输入特征维度（总协变量数量）
               output_dim： 输出特征维度，一般为1
               future_cov_dim： 未来协变量数量维度
               temporal_width_past： 过去协变量投影维度
        """
        
        super(MlpTs3D, self).__init__()
        
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
        # 涨跌幅度评判解码器
        self.sv_decoders = nn.Sequential(
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
                output_dim=decoder_output_dim,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )
        
        decoder_input_dim = decoder_output_dim
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=False, # 这里不能使用layer norm,否则输出值都为0
            dropout=dropout,
        )
        # 针对涨跌幅度的整合解码器，输出为2个维度，分别衡量未来价格涨跌幅度排名和目标值涨跌数值
        self.sv_temporal_decoder = _Residual3DBlock(
            input_dim=decoder_input_dim,
            output_dim=2,
            ins_dim=ins_dim,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=True,
            dropout=dropout,
        )        
        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length
        )            
        self.sv_lookback_skip = nn.Linear(
            ins_dim, ins_dim
        )  
                
    def forward(self, x_in):
        """包括多个不同维度的传播"""

        # x has shape (batch_size, ins_dim,input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, ins_dim,input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, ins_dim,static_cov_dim)
        x, x_future_covariates, x_static_covariates,x_past_last_values = x_in

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

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(2, 3)).transpose(2, 3)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)
        # y = temporal_decoded 
        y = y.view(-1, x.shape[1],self.output_chunk_length, self.output_dim)
        
        ### 涨跌幅度评判解码部分 ###
        
        # 从flatten的数据里进行解码
        sv_decoded = self.sv_decoders(encoded)

        # 在时间维度上对未来协变量取均值
        x_future_covariates_mean = x_dynamic_future_covariates.mean(dim=-2)
        # 融合解码数据以及未来协变量，注意此时的未来协变量属于预测整体变量，和之前的分段未来协变量不同
        temporal_decoder_input = torch.cat([
            sv_decoded,
            x_future_covariates_mean
        ],dim=-1)
        
        temporal_decoded = self.sv_temporal_decoder(temporal_decoder_input)
        # 使用过去数值范围及排名进行残差计算
        skip = self.sv_lookback_skip(x_past_last_values.transpose(1, 2)).transpose(1, 2)
        y_sv = temporal_decoded + skip
        
        return y,y_sv
                                      
                                      