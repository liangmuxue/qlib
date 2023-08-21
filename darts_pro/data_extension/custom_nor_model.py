from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.models.forecasting.block_rnn_model import _BlockRNNModule
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.data.training_dataset import (
    MixedCovariatesTrainingDataset
)
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)

import pickle
import sys
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fastai.torch_core import requires_grad
from sklearn.preprocessing import MinMaxScaler

from cus_utils.encoder_cus import StockNormalizer

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_model import _TFTCusModule,TFTExtModel
from darts_pro.data_extension.batch_dataset import BatchDataset

class _TFTModuleAsis(_TFTCusModule):
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
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    device=device,**kwargs)  
        self.train_data = []
        self.valid_data = []
        self.train_filepath = "custom/data/asis/train_batch.pickel"
        self.valid_filepath = "custom/data/asis/valid_batch.pickel"
        
    def training_step(self, train_batch, batch_idx, optimizer_idx) -> torch.Tensor:
        """use to export data"""
        
        # train_batch = self.filter_batch_by_condition(train_batch,filter_conv_index=self.filter_conv_index)
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = train_batch 
        data = [past_target.detach().cpu().numpy(),past_covariates.detach().cpu().numpy(), historic_future_covariates.detach().cpu().numpy(),
                         future_covariates.detach().cpu().numpy(),static_covariates.detach().cpu().numpy(),scaler,target_class.cpu().detach().numpy(),
                         target.cpu().detach().numpy(),target_info]
        self.train_data += data
        print("len(self.train_data):{}".format(len(self.train_data)))
        fake_loss = torch.ones(1,requires_grad=True).to(self.device)
        return fake_loss

    def validation_step(self, val_batch_ori, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        # 只关注重点部分
        val_batch = val_batch_ori # self.filter_batch_by_condition(val_batch_ori,filter_conv_index=self.filter_conv_index)
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = val_batch 
        data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
                         future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]
        self.valid_data += data
        fake_loss = torch.ones(1).to(self.device)
        self.log("val_loss", fake_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        return fake_loss     
     
    def on_train_start(self):  
        torch.set_grad_enabled(True)
        # for param in self.sub_models[0].parameters():
        #     param.requires_grad = False        
        # for param in self.sub_models[1].parameters():
        #     param.requires_grad = False    
        # for param in self.classify_vr_layer.parameters():
        #     param.requires_grad = False    
                                     
    def on_train_epoch_end(self):
        # train_data = np.stack(self.train_data)
        # np.save(self.train_filepath,train_data)
        # valid_data = np.stack(self.valid_data)
        # np.save(self.valid_filepath,valid_data)
        with open(self.train_filepath, "wb") as fout:
            pickle.dump(self.train_data, fout)    
        with open(self.valid_filepath, "wb") as fout:
            pickle.dump(self.valid_data, fout)                     

class _TFTModuleBatch(_TFTCusModule):
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
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    device=device,**kwargs)  
        
    def training_step(self, train_batch, batch_idx, optimizer_idx) -> torch.Tensor:
        """use to export data"""
        
        return self.training_step_real(train_batch, batch_idx, optimizer_idx)

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        return self.validation_step_real(val_batch, batch_idx)    
                                        
class TFTAsisModel(TFTExtModel):
    
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: Union[int, List[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            Dict[str, Union[int, Tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_weighted_loss_func:bool = False,
        loss_number=3,
        monitor=None,
        mode="train",
        past_split=None,
        filter_conv_index=0,
        **kwargs,
    ):
        
        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,num_attention_heads,
                         full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,add_relative_index,
                         loss_fn,likelihood,norm_type,use_weighted_loss_func,loss_number,monitor,past_split=past_split,**kwargs)
    
    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            target_scaler,
            future_target_class,
            future_target,
            target_info,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = (1,1)
        
        
        # 根据拆分的过去协变量，生成多个配置
        variables_meta_array = []
        for i in range(len(self.past_split)):
            past_index = self.past_split[i]
            past_covariate_item = past_covariate[:,past_index[0]:past_index[1]]
            tensors = [
                past_target,
                past_covariate_item,
                historic_future_covariate,  # for time varying encoders
                future_covariate,
                future_target,  # for time varying decoders
                static_covariates,  # for static encoder
            ]            
            variables_meta,categorical_embedding_sizes = self._build_vriable_metas(tensors, static_covariates,seq=i)
            variables_meta_array.append(variables_meta)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        
        return _TFTModuleAsis(
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
            loss_number=self.loss_number,
            past_split=self.past_split,
            filter_conv_index=self.filter_conv_index,
            device=self.device,
            **self.pl_module_params,
        )    
        
class TFTBatchModel(TFTExtModel):
    
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: Union[int, List[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            Dict[str, Union[int, Tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_weighted_loss_func:bool = False,
        loss_number=3,
        monitor=None,
        mode="train",
        past_split=None,
        filter_conv_index=0,
        batch_file_path=None,
        **kwargs,
    ):
        
        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,num_attention_heads,
                         full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,add_relative_index,
                         loss_fn,likelihood,norm_type,use_weighted_loss_func,loss_number,monitor,
                         past_split=past_split,filter_conv_index=filter_conv_index,**kwargs)
        self.batch_file_path = batch_file_path
    
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="train"
    ) -> BatchDataset:
        
        return BatchDataset(
            filepath = "{}/{}_batch.npy".format(self.batch_file_path,mode)
        )     

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            target_scaler,
            future_target_class,
            future_target,
            target_info,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = (1,1)
        
        
        # 根据拆分的过去协变量，生成多个配置
        variables_meta_array = []
        for i in range(len(self.past_split)):
            past_index = self.past_split[i]
            past_covariate_item = past_covariate[:,past_index[0]:past_index[1]]
            tensors = [
                past_target,
                past_covariate_item,
                historic_future_covariate,  # for time varying encoders
                future_covariate,
                future_target,  # for time varying decoders
                static_covariates,  # for static encoder
            ]            
            variables_meta,categorical_embedding_sizes = self._build_vriable_metas(tensors, static_covariates,seq=i)
            variables_meta_array.append(variables_meta)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        
        return _TFTModuleBatch(
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
            loss_number=self.loss_number,
            past_split=self.past_split,
            filter_conv_index=self.filter_conv_index,
            device=self.device,
            **self.pl_module_params,
        )   
               
    def _batch_collate_filter(self,batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.from_numpy(np.stack(sample_list, axis=0))
                )
            elif isinstance(elem, MinMaxScaler):
                aggregated.append([sample[i] for sample in batch])
            elif isinstance(elem, StockNormalizer):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)
        

               
        