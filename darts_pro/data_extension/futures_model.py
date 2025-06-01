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
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from .industry_model import IndustryRollModel
from darts_pro.data_extension.futures_industry_dataset import FuturesIndustryDataset,FuturesInferenceDataset
from darts_pro.data_extension.futures_industry_droll_dataset import FuturesIndustryDRollDataset
from darts_pro.mam.futures_industry_droll_module import FuturesIndustryDRollModule
from darts_pro.mam.futures_module import FuturesTogeModule
from darts_pro.mam.futures_industry_module import FuturesIndustryModule

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
        self.cut_len = self.ext_kwargs['cut_len']
        # 训练模式下，需要多放回一个静态数据对照集合
        if mode=="train":
            ds = FuturesIndustryDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=self.cut_len,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
                mode=mode
            )  
            # 透传行业分类和股票映射关系，后续使用
            self.train_sw_ins_mappings = ds.sw_ins_mappings            
        # 验证模式下，需要传入之前存储的静态数据集合
        if mode=="valid" or mode=="predict":
            ds = FuturesIndustryDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=self.cut_len,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
                mode=mode
            )
            # 透传行业分类和股票映射关系，后续使用
            self.valid_sw_ins_mappings = ds.sw_ins_mappings     
        return ds      

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="predict",
        pred_date=None,
    ):
        """创建推理数据集"""
        
        self.cut_len = self.ext_kwargs['cut_len']
        ds = FuturesInferenceDataset(
            target_series=target,
            covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            cut_len=self.cut_len,
            max_samples_per_ts=None,
            use_static_covariates=True,
            target_num=len(self.past_split),
            scale_mode=self.scale_mode,
            pred_date=pred_date,
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
                output_dim=self.output_dim,
                cut_len=self.cut_len,
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
                target_mode=self.target_mode,
                scale_mode=self.scale_mode,
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

    def _batch_collate_filter(self,batch):
        """批次整合,包含批次内数据归一化"""
        
        aggregated = []
        first_sample = batch[0]
        sample_len = len(first_sample)
        for i in range(sample_len):
            elem = first_sample[i]
            # 针对round数据，根据标志决定是否在批次内进行归一化
            if i==sample_len-2:
                sample_list = [sample[i] for sample in batch]
                round_data = np.stack(sample_list, axis=0)
                for j in range(len(self.past_split)):
                    if self.scale_mode[j]==5:
                        round_data_item = round_data[...,j]   
                        # round_past_data_item = round_data_item[:,:,:-1]
                        # round_past_data_item_trans = round_past_data_item.transpose(0,2,1)
                        # round_past_data_item = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(
                        #     round_past_data_item_trans.reshape(-1,round_past_data_item_trans.shape[-1])).reshape(round_past_data_item_trans.shape).transpose(0,2,1)
                        round_future_data_item = round_data_item[:,:,-1]
                        round_future_data_item = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(round_future_data_item)        
                        round_data[:,:,-1,j] = round_future_data_item  
                        # round_data[:,:,:-1,j] = round_past_data_item                   
                aggregated.append(
                    torch.from_numpy(round_data)
                )                               
            elif isinstance(elem, np.ndarray) and i!=(sample_len-2):
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.from_numpy(np.stack(sample_list, axis=0))
                )
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, List):
                aggregated.append([sample[i] for sample in batch])
            else:
                print("no match for:",elem.dtype)
                          
        return tuple(aggregated) 

class FuturesIndustryModel(FuturesModel):    

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        cut_len=2,
        mode="train",
        pred_date_begin=None
    ):
        """使用原数据集作为训练和测试数据集"""
        self.cut_len = self.ext_kwargs['cut_len']
        # 训练模式下，需要多放回一个静态数据对照集合
        if mode=="train":
            ds = FuturesIndustryDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=self.cut_len,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
                mode=mode
            )  
            # 透传行业分类和股票映射关系，后续使用
            self.train_sw_ins_mappings = ds.sw_ins_mappings            
        # 验证模式下，需要传入之前存储的静态数据集合
        if mode=="valid" or mode=="predict":
            ds = FuturesIndustryDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=self.cut_len,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
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
               
        model = FuturesIndustryModule(
                output_dim=self.output_dim,
                cut_len=self.ext_kwargs['cut_len'],
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
                target_mode=self.target_mode,
                scale_mode=self.scale_mode,
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

    def _batch_collate_filter(self,batch):
        """批次整合,包含批次内数据归一化"""
        
        aggregated = []
        first_sample = batch[0]
        sample_len = len(first_sample)
        for i in range(sample_len):
            elem = first_sample[i]
            # 针对round数据，根据标志决定是否在批次内进行归一化
            if i==sample_len-2:
                sample_list = [sample[i] for sample in batch]
                round_data = np.stack(sample_list, axis=0)
                for j in range(len(self.past_split)):
                    if self.scale_mode[j]==5:
                        round_data_item = round_data[...,j]   
                        # round_past_data_item = round_data_item[:,:,:-1]
                        # round_past_data_item_trans = round_past_data_item.transpose(0,2,1)
                        # round_past_data_item = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(
                        #     round_past_data_item_trans.reshape(-1,round_past_data_item_trans.shape[-1])).reshape(round_past_data_item_trans.shape).transpose(0,2,1)
                        round_future_data_item = round_data_item[:,:,-1]
                        round_future_data_item = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(round_future_data_item)        
                        round_data[:,:,-1,j] = round_future_data_item  
                        # round_data[:,:,:-1,j] = round_past_data_item                   
                aggregated.append(
                    torch.from_numpy(round_data)
                )                               
            elif isinstance(elem, np.ndarray) and i!=(sample_len-2):
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.from_numpy(np.stack(sample_list, axis=0))
                )
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, List):
                aggregated.append([sample[i] for sample in batch])
            else:
                print("no match for:",elem.dtype)
                          
        return tuple(aggregated) 
     
    def predict(self,series,past_covariates=None,future_covariates=None,pred_date_begin=None,batch_size=1,num_loader_workers=1):
        
        dataset = self._build_inference_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=10,
            mode="predict",
            pred_date=pred_date_begin,
        )
        
        pred_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_filter,
        )        
        self.trainer.predict(self.model, pred_loader)
        predictions = self.model.result_target
        
        return predictions
    
class FuturesIndustryDRollModel(FuturesModel):    

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        cut_len=2,
        mode="train"
    ):
        """使用原数据集作为训练和测试数据集"""
        self.cut_len = cut_len
        # 训练模式下，需要多放回一个静态数据对照集合
        if mode=="train":
            ds = FuturesIndustryDRollDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=cut_len,
                rolling_size=self.rolling_size,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
                mode=mode
            )  
            # 透传行业分类和股票映射关系，后续使用
            self.train_sw_ins_mappings = ds.sw_ins_mappings            
        # 验证模式下，需要传入之前存储的静态数据集合
        if mode=="valid" or mode=="predict":
            ds = FuturesIndustryDRollDataset(
                target_series=target,
                covariates=past_covariates,
                future_covariates=future_covariates,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                cut_len=cut_len,
                rolling_size=self.rolling_size,
                max_samples_per_ts=None,
                use_static_covariates=True,
                target_num=len(self.past_split),
                scale_mode=self.scale_mode,
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
                    for ts in [historic_future_covariate[0,0], expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate[0,0], expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = self.define_output_dim()
        
        # 根据拆分的过去协变量，生成多个配置，需要把原来的3维改成2维进行计算
        ori_tensors = [
            past_target[0,0],
            past_covariate[0,0],
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target[0,0],  # for time varying decoders
            static_covariates[0,0],  # for static encoder
        ]          
        variables_meta_array,categorical_embedding_sizes = self.build_variable(ori_tensors)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
               
        model = FuturesIndustryDRollModule(
                output_dim=self.output_dim,
                cut_len=self.cut_len,
                rolling_size=self.rolling_size,
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
                target_mode=self.target_mode,
                scale_mode=self.scale_mode,
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

    def _batch_collate_filter(self,batch):
        """批次整合,合并批次以及二次滚动计算的维度"""
        
        aggregated = []
        first_sample = batch[0]
        sample_len = len(first_sample)
        for i in range(sample_len):
            elem = first_sample[i]
            # 合并前2个维度
            if isinstance(elem, np.ndarray):
                sample_list = [sample[i] for sample in batch]
                data = np.stack(sample_list, axis=0)
                data = data.reshape(-1,*data.shape[2:])
                aggregated.append(
                    torch.from_numpy(data)
                )
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, List):
                list_data = [sample[i] for sample in batch]
                aggregated.append(np.array(list_data))
            else:
                print("no match for:",elem.dtype)
                          
        return tuple(aggregated) 
     
    def predict(self,series,past_covariates=None,future_covariates=None,batch_size=1,num_loader_workers=1):
        
        dataset = self._build_inference_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=10,
            mode="predict"
        )
        pred_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_filter,
        )        
        self.trainer.predict(self.model, pred_loader)
        predictions = self.model.result_target
        
        return predictions    
    
    