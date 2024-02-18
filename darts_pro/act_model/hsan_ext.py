import pickle
import sys
import numpy as np
import pandas as pd
import torch
import tsaug
import os
from typing import Tuple, Union

from torch import nn
import torch.nn.functional as F
from darts.models.forecasting.tide_model import _ResidualBlock

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class HsanExt(nn.Module):
    """整合Hsan聚类模型，并修改其中的部分网络，添加Seq2Seq部分"""
    
    def __init__(self, 
        input_dim: int,
        output_dim: int,
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
        hidden_dim=64,
        act="ident", 
        n_num=100,
        device=None,
        **kwargs
       ):
        
        super(HsanExt, self).__init__()
        
        # Tide模型部分的初始化定义
        self.input_chunk_length = kwargs["input_chunk_length"]
        self.output_chunk_length = kwargs["output_chunk_length"]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.past_cov_dim = input_dim - output_dim - future_cov_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_past = temporal_width_past
        self.temporal_width_future = temporal_width_future

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
                * self.output_chunk_length
                * self.nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        decoder_input_dim = decoder_output_dim * self.nr_params
        if temporal_width_future and future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length * self.nr_params
        )
          
        # Hsan模型部分的初始化定义      
        self.AE1 = nn.Linear(self.output_chunk_length, hidden_dim)
        self.AE2 = nn.Linear(self.output_chunk_length, hidden_dim)

        self.SE1 = nn.Linear(n_num, hidden_dim)
        self.SE2 = nn.Linear(n_num, hidden_dim)

        self.alpha = nn.Parameter(torch.Tensor(1, ))
        self.alpha.data = torch.tensor(0.99999).to(device)

        self.pos_weight = torch.ones(n_num * 2).double().to(device)
        self.pos_neg_weight = torch.ones([n_num * 2, n_num * 2]).double().to(device)

        if act == "ident":
            self.activate = lambda x: x
        if act == "sigmoid":
            self.activate = nn.Sigmoid()
        
        self.n_num = n_num
        self.device = device
 
    def forward(
        self, x_in, A
    ) -> torch.Tensor:
        """整合TiDE与HSAN两种模型的处理"""


        ##################TiDE部分，主要是整合协变量，以及实现Sequence To Sequence部分的编解码#######################
        x, x_future_covariates, x_static_covariates = x_in

        x_lookback = x[:, :, : self.output_dim]
        
        # 拼接各种协变量
        if self.future_cov_dim:
            x_dynamic_future_covariates = torch.cat(
                [
                    x[
                        :,
                        :,
                        None if self.future_cov_dim == 0 else -self.future_cov_dim :,
                    ],
                    x_future_covariates,
                ],
                dim=1,
            )
            if self.temporal_width_future:
                # project input features across all input and output time steps
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        if self.past_cov_dim:
            x_dynamic_past_covariates = x[
                :,
                :,
                self.output_dim : self.output_dim + self.past_cov_dim,
            ]
            if self.temporal_width_past:
                # project input features across all input time steps
                x_dynamic_past_covariates = self.past_cov_projection(
                    x_dynamic_past_covariates
                )
        else:
            x_dynamic_past_covariates = None

        encoded = [
            x_lookback,
            x_dynamic_past_covariates,
            x_dynamic_future_covariates,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # 实现SEQ2SEQ模式下的编码以及解码
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )
        y = y.squeeze(-1)
        
        # 取消预测端长度连接，直接使用原维度
        # y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
           
        ##################HSAN部分，主要是整合协变量，以及实现Sequence To Sequence部分的编解码#######################
        
        # 使用之前生成的序列特征值，进行Linear操作
        Z1 = self.activate(self.AE1(y))
        Z2 = self.activate(self.AE2(y))

        Z1 = F.normalize(Z1, dim=1, p=2)
        Z2 = F.normalize(Z2, dim=1, p=2)
        
        # 对特征矩阵进行网络处理
        E1 = F.normalize(self.SE1(A), dim=1, p=2)
        E2 = F.normalize(self.SE2(A), dim=1, p=2)

        return Z1, Z2, E1, E2
    
    