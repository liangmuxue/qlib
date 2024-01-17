from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError

from darts.models.forecasting.tft_model import _TFTModule
from darts.models.forecasting.tide_model import _TideModule
from losses.mtl_loss import TripletLoss,UncertaintyLoss
from .series_data_utils import StatDataAssis
from tft.class_define import CLASS_SIMPLE_VALUES

hide_target = True

class BaseMixModule(PLMixedCovariatesModule):

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
        filter_conv_index=0,
        loss_number=3,
        device="cpu",
        train_sample=None,
        model_type='tft',
        **kwargs
    ):
        # 模拟初始化，实际未使用
        super().__init__(**kwargs)
        self.train_sample = train_sample
        self.past_split = past_split
        self.filter_conv_index = filter_conv_index
        self.variables_meta_array = variables_meta_array
        self.output_dim = output_dim
        model_list = []
        classify_vr_layers = []
        # 涨跌幅度分类
        vr_range_num = len(CLASS_SIMPLE_VALUES.keys()) 
        for i in range(len(past_split)):
            # 拆分过去协变量,形成不同的网络配置，给到不同的model
            model =  self.create_real_model(output_dim, variables_meta_array[i], num_static_components, 
                                            hidden_size, lstm_layers, num_attention_heads, full_attention, 
                                            feed_forward, hidden_continuous_size, categorical_embedding_sizes, 
                                            dropout, add_relative_index, norm_type,model_type=model_type,**kwargs)
            # mse损失计算                
            model.mean_squared_error = MeanSquaredError().to(device)
            model_list.append(model)
            vr_layer = self._construct_classify_layer(self.output_chunk_length,vr_range_num)  
            classify_vr_layers.append(vr_layer)
        self.sub_models = nn.ModuleList(model_list) 
        self.vr_layers = nn.ModuleList(classify_vr_layers) 
        # 序列分类层，包括目标分类和输出分类
        self.classify_vr_layer = self._construct_classify_layer(len(past_split),self.output_chunk_length,device=device)        
        self.classify_tar_layer = self._construct_classify_layer(len(past_split),self.output_chunk_length,device=device)  
        # 使用不确定多重损失函数
        if use_weighted_loss_func and not isinstance(model.criterion,UncertaintyLoss):
            self.criterion = UncertaintyLoss(device=device) 
                    
        self.val_results = {}
        # 辅助数据功能
        self.data_assis = StatDataAssis()
        # 优化器执行频次
        self.lr_freq = {"interval":"epoch","frequency":1}
        # 手动控制参数更新
        self.automatic_optimization = False
        self.freeze_mode = torch.ones(len(past_split)+2).to(device)
        
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
        **kwargs):

        if model_type == "tft":
            return _TFTModule(output_dim,variables_meta,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                                full_attention,feed_forward,hidden_continuous_size,categorical_embedding_sizes,
                                                dropout,add_relative_index,norm_type,**kwargs)     
        if model_type == "tide":  
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                (scaler,future_past_covariate),
                target_class,
                future_target,
                target_info
            ) = self.train_sample      
                  
            past_target_shape = len(variables_meta["input"]["past_target"])
            past_covariates_shape = len(variables_meta["input"]["past_covariate"])
            historic_future_covariates_shape = len(variables_meta["input"]["historic_future_covariate"])
            input_dim = (
                past_target_shape
                + past_covariates_shape
                + historic_future_covariates_shape
            )
    
            output_dim = output_dim[0]
    
            future_cov_dim = (
                future_covariates.shape[1] if future_covariates is not None else 0
            )
            static_cov_dim = (
                static_covariates.shape[0] * static_covariates.shape[1]
                if static_covariates is not None
                else 0
            )
    
            nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
    
            return _TideModule(
                input_dim=input_dim,
                output_dim=output_dim,
                future_cov_dim=future_cov_dim,
                static_cov_dim=static_cov_dim,
                nr_params=nr_params,
                num_encoder_layers=1,
                num_decoder_layers=1,
                decoder_output_dim=16,
                hidden_size=hidden_size,
                temporal_width_past=4,
                temporal_width_future=4,
                temporal_decoder_hidden=32,
                use_layer_norm=False,
                dropout=dropout,
                **kwargs,
            )        
        
    def _process_input_batch(
        self, input_batch
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """重载方法，以适应数据结构变化"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        dim_variable = 2

        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[:,:,past_conv_index[0]:past_conv_index[1]]
            # 修改协变量生成模式，只取自相关目标作为协变量
            if hide_target:
                conv_defs = [
                            past_target[:,:,i:i+1],
                            past_covariates_item,
                            historic_future_covariates,
                    ]
            else:
                conv_defs = [
                            past_target,
                            past_covariates_item,
                            historic_future_covariates,
                    ]              
            x_past = torch.cat(
                [
                    tensor
                    for tensor in conv_defs if tensor is not None
                ],
                dim=dim_variable,
            )
            x_past_array.append(x_past)
        return x_past_array, future_covariates, static_covariates        
        