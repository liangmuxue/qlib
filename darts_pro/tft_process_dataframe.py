# 使用darts架构的TFT模型，定制化numpy数据集模式

from __future__ import division
from __future__ import print_function

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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from darts.metrics import mape
from darts.models import TFTModel

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression
from .tft_comp_stock import process

from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from darts_pro.data_extension.custom_model import TFTCusModel
from darts_pro.tft_series_dataset import TFTSeriesDataset
from .base_process import BaseNumpyModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TftDataframeModel():
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
        
    def fit(
        self,
        dataset: TFTSeriesDataset
    ):
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
            return       
              
        # 生成tft时间序列数据集,包括目标数据、协变量等
        train_series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.get_series_data()
        
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        load_weight = self.optargs["load_weight"]
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 trainer=None,epochs=self.n_epochs,verbose=True)
    
    def _build_model(self,dataset,emb_size=1000,use_model_name=True):
        optimizer_cls = torch.optim.Adam
        scheduler = CosineAnnealingLR
        scheduler_config = {
            "T_max": 5, 
            "eta_min": 0,
        }        
        # scheduler = CosineAnnealingWarmRestarts
        # scheduler_config = {
        #     "T_0": 3,
        #     "T_mult": 3
        # }     
        quantiles = [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ]     
               
        categorical_embedding_sizes = {"dayofweek": 5,dataset.get_group_rank_column(): emb_size}    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        model_name = self.optargs["model_name"]
        if not use_model_name:
            model_name = None
        my_model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.optargs["forecast_horizon"],
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            add_relative_index=False,
            add_encoders=None,
            categorical_embedding_sizes=categorical_embedding_sizes,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),  # QuantileRegression is set per default
            # loss_fn=torch.nn.MSELoss(),
            random_state=42,
            model_name=model_name,
            force_reset=True,
            log_tensorboard=True,
            save_checkpoints=True,
            work_dir=self.optargs["work_dir"],
            lr_scheduler_cls=scheduler,
            lr_scheduler_kwargs=scheduler_config,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs={"lr": 1e-2},
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}  
        )
        return my_model          

            
    def predict(self, dataset: TFTSeriesDataset):
        if self.type=="predict_dataframe":
            self.predict_dataframe(dataset)
            return        
        if self.type!="predict":
            return 
           
        model_name = self.optargs["model_name"]
        forecast_horizon = self.optargs["forecast_horizon"]
        my_model = self._build_model(dataset,emb_size=1000,use_model_name=False)
        val_series_list,past_covariates,future_covariates,static_covariates,series_total = dataset.get_series_data()
        # 首先需要进行fit设置
        my_model.super_fit(val_series_list, past_covariates=past_covariates, future_covariates=future_covariates,
                     val_series=val_series_list,val_past_covariates=past_covariates,val_future_covariates=future_covariates,
                     verbose=True,epochs=-1)            
        # 根据参数决定是否从文件中加载权重
        if model_name is not None:
            my_model = TFTCusModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"])       
    
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=forecast_horizon, series=val_series_list,
                                              num_samples=200,past_covariates=past_covariates,future_covariates=future_covariates)
        
        self.predict_show(val_series_list,pred_series_list, series_total)
        
    def view_df(self,df,target_title):
        viz_input = TensorViz(env="data_hist")
        view_data = df[["label"]].values
        viz_input.viz_matrix_var(view_data,win=target_title,title=target_title)  

     
        
        