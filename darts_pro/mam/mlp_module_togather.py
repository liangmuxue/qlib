import os

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import tsaug

import torchvision
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning.callbacks as pl_callbacks
from torch.utils.data import DataLoader
from torch.distributions import Normal
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from cus_utils.tensor_viz import TensorViz

import cus_utils.global_var as global_var
from darts_pro.act_model.mlp_ts import MlpTs
from cus_utils.metrics import pca_apply
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.clustering_loss import MlpLoss
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
from cus_utils.visualization import ShowClsResult
from losses.quanlity_loss import QuanlityLoss

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from .mlp_module import MlpModule

class MlpTogeModule(MlpModule):
    """MlpModule for Data Togather """
    
    def __init__(
        self,
        output_dim: Tuple[int, int],
        variables_meta_array: Tuple[Dict[str, Dict[str, List[str]]],Dict[str, Dict[str, List[str]]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        use_weighted_loss_func=False,
        past_split=None,
        step_mode="normal",
        batch_file_path=None,
        static_datas=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,batch_file_path=batch_file_path,
                                    device=device,**kwargs)  
        
    def create_real_model(self,
        output_dim: Tuple[int, int],
        variables_meta: Dict[str, Dict[str, List[str]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        model_type="tft",
        device="cpu",
        seq=0,
        **kwargs):
        
            (
                past_target,
                past_covariates,
                historic_future_covariate,
                future_covariates,
                static_covariates,
                _,
                _,
                future_target,
                _,
                price_target
            ) = self.train_sample      
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            historic_future_covariates_shape = historic_future_covariate.shape[-1]
            # 记录动态数据长度，后续需要切片
            self.dynamic_conv_shape = past_target_shape + past_covariates_shape
            input_dim = (
                past_target_shape
                + past_covariates_shape
                + historic_future_covariates_shape
            )
    
            output_dim = 1
    
            future_cov_dim = (
                future_covariates.shape[-1] if future_covariates is not None else 0
            )
            
            static_cov_dim = (
                # 由于前期增加了归一化字段用于后续替换，因此在这里减1维
                static_covariates.shape[-2] * static_covariates.shape[-1] - 1
                if static_covariates is not None
                else 0
            )
    
            nr_params = 1
            self.pca_dim = 10
            
            model = MlpTs(
                # Tide Part
                input_dim=input_dim,
                emb_output_dim=output_dim,
                future_cov_dim=future_cov_dim,
                static_cov_dim=static_cov_dim,
                nr_params=nr_params,
                num_encoder_layers=3,
                num_decoder_layers=3,
                decoder_output_dim=16,
                hidden_size=hidden_size,
                temporal_width_past=4,
                temporal_width_future=4,
                temporal_decoder_hidden=32,
                use_layer_norm=True,
                dropout=dropout,
                # Mlp Part
                enc_nr_params=len(QuanlityLoss().quantiles),
                n_cluster=len(CLASS_SIMPLE_VALUES.keys()),
                pca_dim=self.pca_dim,
                device=device,
                **kwargs,
            )           
            
            return model

    def on_train_epoch_start(self):  
        self.loss_data = []
    def on_train_epoch_end(self):  
        pass
        
    def on_validation_epoch_start(self):  
        self.import_price_result = None
        self.total_imp_cnt = 0
                    
    def training_step_real(self, train_batch, batch_idx): 
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            _,
            target_class,
            target_info,
            price_target,
            future_target
        ) = train_batch
        
        return super().training_step_real((past_target,past_covariates, historic_future_covariates,future_covariates,
                                           static_covariates,target_class,price_target,future_target),batch_idx)     
