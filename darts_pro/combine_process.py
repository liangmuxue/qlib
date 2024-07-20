# 使用darts架构的TFT模型，定制化numpy数据集模式

from __future__ import division
from __future__ import print_function

import datetime
import os
import numpy as np
from collections import Counter
import pandas as pd
import pickle
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import gc

from darts.metrics import mape
from darts.models import TFTModel
from darts import TimeSeries, concatenate
from torchmetrics import (
    PearsonCorrCoef,
    MetricCollection,
)
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR,MultiStepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression
from custom_model.simple_model import Trainer
from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from cus_utils.metrics import corr_dis,series_target_scale,diff_dis,cel_acc_compute,vr_acc_compute
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC
from darts_pro.data_extension.custom_model import TFTExtModel
from darts_pro.data_extension.custom_nor_model import TFTAsisModel,TFTBatchModel,TFTCluBatchModel,TFTCluSerModel
from darts_pro.data_extension.batch_dataset import BatchDataset
from darts_pro.tft_series_dataset import TFTSeriesDataset

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays,date_string_transfer,get_tradedays_dur

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from cus_utils.log_util import AppLogger
logger = AppLogger()

class CombineProcess():
    def __init__(
        self,
        d_model: int = 64,
        batch_size: int = 8192,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        optargs=None,
        # 模式 opt_train:寻找最优化参数训练 "best_train":使用最优化参数训练
        type="opt_train",
        **kwargs
    ):
        # 业务参数部分
        self.optargs = optargs
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        
        self.n_jobs = n_jobs
        self.gpus = GPU
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        
        self.type = type
        self.kwargs = kwargs
        
        global_var._init()

    def fit(
        self,
        dataset: TFTSeriesDataset,
    ):
        global_var.set_value("dataset", dataset)

        if self.type.startswith("pred"):
            self.predict(dataset)
            return           
             
        
    def _build_model(self,dataset,emb_size=1000,use_model_name=True,mode=0,batch_file_path=None):
        """生成模型"""
        
        log_every_n_steps = self.kwargs["log_every_n_steps"]
        optimizer_cls = torch.optim.Adam
        # optimizer_cls = torch.optim.SGD
        scheduler_config = self.kwargs["scheduler_config"]
        optimizer_kwargs = self.kwargs["optimizer_kwargs"]
        
        scheduler = torch.optim.lr_scheduler.CyclicLR
        categorical_embedding_sizes = {"dayofweek": 5,dataset.get_group_rank_column(): emb_size}
        # categorical_embedding_sizes = None    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        past_split = self.optargs["past_split"] 
        filter_conv_index = self.optargs["filter_conv_index"] 
        model_name = self.optargs["model_name"]
        model_type = self.optargs["model_type"]
        if not use_model_name:
            model_name = None
        
        # 自定义回调函数
        lightning_callbacks = []
        if "lightning_callbacks" in  self.kwargs:
            lightning_callbacks_config = self.kwargs.get("lightning_callbacks", [])
            for config in lightning_callbacks_config:
                callback = init_instance_by_config(
                    config,
                )   
                lightning_callbacks.append(callback)             
        
        if mode==0:  
            my_model = TFTExtModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=self.optargs["forecast_horizon"],
                    hidden_size=64,
                    lstm_layers=1,
                    num_attention_heads=4,
                    dropout=self.optargs["dropout"],
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    add_relative_index=True,
                    add_encoders=None,
                    categorical_embedding_sizes=categorical_embedding_sizes,
                    # likelihood=QuantileRegression(
                    #     quantiles=quantiles
                    # ), 
                    likelihood=None,
                    # loss_fn=torch.nn.MSELoss(),
                    use_weighted_loss_func=True,
                    loss_number=4,
                    # torch_metrics=metric_collection,
                    random_state=42,
                    model_name=model_name,
                    force_reset=True,
                    log_tensorboard=True,
                    save_checkpoints=True,
                    past_split=past_split,
                    filter_conv_index=filter_conv_index,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    model_type=model_type,
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )
        elif mode==1:
            my_model = TFTAsisModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=self.optargs["forecast_horizon"],
                    hidden_size=64,
                    lstm_layers=1,
                    num_attention_heads=4,
                    dropout=self.optargs["dropout"],
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    add_relative_index=True,
                    add_encoders=None,
                    categorical_embedding_sizes=categorical_embedding_sizes,
                    # likelihood=QuantileRegression(
                    #     quantiles=quantiles
                    # ), 
                    likelihood=None,
                    # loss_fn=torch.nn.MSELoss(),
                    use_weighted_loss_func=True,
                    loss_number=4,
                    # torch_metrics=metric_collection,
                    random_state=42,
                    model_name=model_name+"_exp", # Change To exp name
                    force_reset=True,
                    log_tensorboard=True,
                    save_checkpoints=True,
                    past_split=past_split,
                    filter_conv_index=filter_conv_index,
                    batch_file_path=batch_file_path,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    model_type=model_type,
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )                
        else:
            my_model = TFTCluSerModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=self.optargs["forecast_horizon"],
                    hidden_size=64,
                    lstm_layers=1,
                    num_attention_heads=4,
                    dropout=self.optargs["dropout"],
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    add_relative_index=False,
                    add_encoders=None,
                    categorical_embedding_sizes=categorical_embedding_sizes,
                    likelihood=None,
                    use_weighted_loss_func=True,
                    loss_number=4,
                    random_state=42,
                    model_name=model_name,
                    force_reset=True,
                    log_tensorboard=True,
                    save_checkpoints=True, 
                    past_split=past_split,
                    filter_conv_index=filter_conv_index,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    batch_file_path=batch_file_path,
                    model_type=model_type,
                    step_mode=self.optargs["step_mode"],
                    pretrain_model_name=self.optargs["pretrain_model_name"],
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,
                                       "callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )            
        return my_model          

            
    def predict(self, dataset: TFTSeriesDataset):
       
        if self.type!="predict":
            return 

        path = "{}/result_view".format(self.optargs["work_dir"])
        if not os.path.exists(path):
            os.mkdir(path)

        self.batch_file_path = self.kwargs["batch_file_path"]
        ignore_exp = self.kwargs["ignore_exp"]
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        model_name = self.optargs["model_name"]
        forecast_horizon = self.optargs["forecast_horizon"]
        pred_range = dataset.kwargs["segments"]["test"] 
        
        expand_length = 30
         
        emb_size = dataset.get_emb_size()
        batch_file_path = self.batch_file_path
        model_exp = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=1,batch_file_path=batch_file_path) 
        # 根据参数决定是否从文件中加载权重
        load_weight = self.optargs["load_weight"]            
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTCluSerModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=best_weight)
            self.model.batch_size = self.batch_size  
            self.model.batch_file_path = self.batch_file_path   
            self.model.mode = "train"       
             
        # 根据日期范围逐日进行预测，得到预测结果   
        start_time = pred_range[0]
        end_time = pred_range[1]
        date_range = get_tradedays(start_time,end_time)                    
        for cur_date in date_range:      
  
            # 同时需要延长集合时间
            total_range = dataset.segments["train_total"]
            valid_range = dataset.segments["valid"]    
            last_day = get_tradedays_dur(total_range[1],expand_length)
            # 以当天为数据时间终点
            total_range[1] = cur_date
            valid_range[1] = cur_date    
            # 生成未扩展的真实数据
            dataset.build_series_data_step_range(total_range,valid_range)  
            # 为了和训练阶段保持一致处理，需要补充模拟数据
            df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
            # 重置数据区间以满足生成第一阶段数据的足够长度
            total_range[1] = last_day
            valid_range[1] = last_day            
            # 生成序列数据            
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
                dataset.build_series_data_step_range(total_range,valid_range,outer_df=df_expands)            
            # 每次使用前置模型生成一阶段数据          
            model_exp.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=1,verbose=True,num_loader_workers=6)               
            
            
            # 根据时间点，取得对应的输入时间序列范围
            total_range,val_range,missing_instruments = dataset.get_part_time_range(cur_date,ref_df=dataset.df_all)
            # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
            _,val_series_transformed,series_total,past_convariates,future_convariates = \
                dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=df_expands)            
            # 进行第二阶段预测           
            pred_result = self.model.predict(n=dataset.pred_len, series=val_series_transformed,num_samples=10,cur_date=cur_date,
                                                past_covariates=past_convariates,future_covariates=future_convariates)
            # 对预测结果进行评估
            if pred_result is None:
                print("{} pred_result None".format(cur_date))
            else:
                print("{} res len:{}".format(cur_date,len(pred_result)))
    
        