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
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from cus_utils.tensor_viz import TensorViz
from darts_pro.data_extension.togather_model import DateTogeModel
from darts_pro.data_extension.industry_align_dataset import IndustryShiftedDataset
from darts_pro.data_extension.industry_togather_dataset import IndustryTogatherDataset
from darts_pro.data_extension.date_align_dataset import DateShiftedDataset
from darts_pro.mam.industry_togather_module import IndustryTogeModule
from darts_pro.data_extension.custom_model import logger

"""把分阶段数据再次整合到一起"""

class IndustryModel(DateTogeModel):
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
        likelihood=None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_weighted_loss_func:bool = False,
        loss_number=3,
        monitor=None,
        past_split=None,
        filter_conv_index=0,
        **kwargs,
    ):
        """继承父类，进行数据集和模型整合"""
        
        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,num_attention_heads,
                         full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,add_relative_index,
                         loss_fn,likelihood,norm_type,use_weighted_loss_func,loss_number,monitor,
                         past_split=past_split,filter_conv_index=filter_conv_index,**kwargs)    
    
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
            ds = IndustryTogatherDataset(
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
            ds = IndustryTogatherDataset(
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
               
        model = IndustryTogeModule(
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
   
    @staticmethod           
    def _batch_collate_fn(batch: List[Tuple]) -> Tuple:
        """批次整合"""
        
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
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)    

    @staticmethod            
    def load_from_checkpoint(
        model_name: str,
        work_dir: str = None,
        file_name: str = None,
        best: bool = True,
        **kwargs,
    ):
        """重载原方法，使用自定义模型加载策略"""
        
        logger = get_logger(__name__)
        
        checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
        model_dir = _get_runs_folder(work_dir, model_name)

        # load the base TorchForecastingModel (does not contain the actual PyTorch LightningModule)
        base_model_path = os.path.join(model_dir, INIT_MODEL_NAME)
        raise_if_not(
            os.path.exists(base_model_path),
            f"Could not find base model save file `{INIT_MODEL_NAME}` in {model_dir}.",
            logger,
        )
        model = torch.load(
            base_model_path, map_location=kwargs.get("map_location")
        )
               
        # 修改原方法，对best重新界定
        if file_name is None:
            checkpoint_dir = _get_checkpoint_folder(work_dir, model_name)
            path = os.path.join(checkpoint_dir, "epoch=*")
            checklist = glob(path)            
            if len(checklist) == 0:
                raise_log(
                    FileNotFoundError(
                        "There is no file matching prefix {} in {}".format(
                            "epoch=*", checkpoint_dir
                        )
                    ),
                    logger,
                )   
            if best:
                # 如果查找best，则使用文件中的最高分数进行匹配
                min_loss = 100
                cadi_x = None
                for x in checklist:
                    cur_loss = float(x.split("=")[2][:-5])
                    cur_epoch = int(x.split("=")[1].split("-")[0])
                    # 大于一定的epoch才计算评分
                    if cur_epoch>50 and cur_loss<=min_loss:
                        min_loss = cur_loss
                        cadi_x = x
                file_name = cadi_x
            else:
                # 否则使用文件中的最大epoch进行匹配
                file_name = max(checklist, key=lambda x: int(x.split("=")[1].split("-")[0]))
                # file_name = "epoch=117-val_CNTN_loss=0.75.ckpt"
            file_name = os.path.basename(file_name)       
        
        file_path = os.path.join(checkpoint_dir, file_name)
        print("weights file_path:",file_path) 
        model.model = model._load_from_checkpoint(file_path, **kwargs)
        model.batch_file_path = kwargs["batch_file_path"]
        model.model.set_filepath(kwargs["batch_file_path"])
        
        # loss_fn is excluded from pl_forecasting_module ckpt, must be restored
        loss_fn = model.model_params.get("loss_fn")
        if loss_fn is not None:
            model.model.criterion = loss_fn
        # train and val metrics also need to be restored
        torch_metrics = model.model.configure_torch_metrics(
            model.model_params.get("torch_metrics")
        )
        model.model.train_metrics = torch_metrics.clone(prefix="train_")
        model.model.val_metrics = torch_metrics.clone(prefix="val_")

        # restore _fit_called attribute, set to False in load() if no .ckpt is found/provided
        model._fit_called = True
        model.load_ckpt_path = file_path
                
        return model 
    
    
