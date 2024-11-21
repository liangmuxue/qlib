from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import _generate_new_dates
from darts.models.forecasting.torch_forecasting_model import _get_checkpoint_folder,_get_runs_folder,INIT_MODEL_NAME,_get_checkpoint_fname

import os
from glob import glob
import numpy as np
import pandas as pd
import torch
import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn

from .industry_model import IndustryRollModel
from darts_pro.data_extension.futures_togather_dataset import FuturesTogatherDataset
from darts_pro.mam.futures_module import FuturesTogeModule

"""把分阶段数据再次整合到一起"""
    
class FuturesModel(IndustryRollModel):    

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="train"
    ):
        """使用原数据集作为训练和测试数据集"""
        
        # 训练模式下，需要多放回一个静态数据对照集合
        if mode=="train":
            ds = FuturesTogatherDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                mode=mode
            )  
            # 透传行业分类和股票映射关系，后续使用
            self.train_sw_ins_mappings = ds.sw_ins_mappings            
        # 验证模式下，需要传入之前存储的静态数据集合
        if mode=="valid":
            ds = FuturesTogatherDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                ass_sw_ins_mappings=self.train_sw_ins_mappings, # 验证集是需要传入训练集映射关系数据，以进行审计
                mode=mode
            )
            # 透传行业分类和股票映射关系，后续使用
            self.valid_sw_ins_mappings = ds.sw_ins_mappings     
        return ds      
    
    def _create_model(self, train_sample) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            _,
            future_target,
            _,
            _,
            _,
            _
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate[0], expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate[0], expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = self.define_output_dim()
        
        # 根据拆分的过去协变量，生成多个配置，需要把原来的3维改成2维进行计算
        ori_tensors = [
            past_target[0],
            past_covariate[0],
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target[0],  # for time varying decoders
            static_covariates[0],  # for static encoder
        ]          
        variables_meta_array,categorical_embedding_sizes = self.build_variable(ori_tensors)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
               
        model = FuturesTogeModule(
                indus_dim=past_target.shape[0],
                output_dim=self.output_dim,
                variables_meta_array=variables_meta_array,
                num_static_components=n_static_components,
                hidden_size=self.hidden_size,
                lstm_layers=self.lstm_layers,
                dropout=self.dropout,
                num_attention_heads=self.num_attention_heads,
                full_attention=self.full_attention,
                feed_forward=self.feed_forward,
                hidden_continuous_size=self.hidden_continuous_size,
                categorical_embedding_sizes=self.categorical_embedding_sizes,
                add_relative_index=self.add_relative_index,
                norm_type=self.norm_type,
                use_weighted_loss_func=self.use_weighted_loss_func,
                past_split=self.past_split,
                filter_conv_index=self.filter_conv_index,
                device=self.device,
                batch_file_path=self.batch_file_path,
                step_mode=self.step_mode,
                model_type=self.model_type,
                train_sample=self.train_sample,
                train_sw_ins_mappings=self.train_sw_ins_mappings,     
                valid_sw_ins_mappings=self.valid_sw_ins_mappings,     
                **self.pl_module_params,
        )     
        return model     