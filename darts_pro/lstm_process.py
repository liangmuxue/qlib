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
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression

from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from darts_pro.data_extension.custom_nor_model import CusNorModel
from darts_pro.tft_series_dataset import TFTSeriesDataset
from .base_process import BaseNumpyModel

class LstmNumpyModel(BaseNumpyModel):
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
        
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def fit(
        self,
        dataset: TFTDataset
    ):         
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
            return       
        # 生成tft时间序列训练数据集
        data_train = dataset.get_custom_numpy_dataset(mode="train")
        data_validation = dataset.get_custom_numpy_dataset(mode="valid")
        self.numpy_data_view(dataset, data_train.numpy_data,title="train_data")
        self.numpy_data_view(dataset, data_validation.numpy_data,title="valid_data")
        # 使用股票代码数量作为embbding长度
        # emb_size = np.unique(dataset.data[:,:,dataset.get_target_column_index()])
        emb_size = 1000
        load_weight = self.optargs["load_weight"]
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = BlockRNNModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
        self.model.fit(data_train,data_validation,trainer=None,epochs=self.n_epochs,verbose=True)
        print("ok")
    
    def _build_model(self,dataset,emb_size=1000,use_model_name=True):
        optimizer_cls = torch.optim.Adam
        scheduler = CosineAnnealingLR
        scheduler_config = {
            "T_max": 5, 
            "eta_min": 0,
        }        

        categorical_embedding_sizes = {"dayofweek": 5,dataset.col_def["group_column"]: emb_size}    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        model_name = self.optargs["model_name"]
        if not use_model_name:
            model_name = None
        my_model = CusNorModel(
            model_type=dataset.model_type,
            model="LSTM",
            training_length=input_chunk_length,
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.optargs["forecast_horizon"],
            hidden_dim=20,
            n_rnn_layers=1,
            # loss_fn=torch.nn.L1Loss(),
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            dropout=0.1,
            model_name=model_name,
            random_state=45,
            force_reset=True,
            log_tensorboard=True,
            save_checkpoints=True,
            work_dir=self.optargs["work_dir"],
            optimizer_kwargs={"lr": 1e-3},
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}  
        )
        return my_model          
        
    def predict_numpy(self, dataset: TFTDataset):
        if self.type!="predict":
            return 
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
           
        model_name = self.optargs["model_name"]
        my_model = self._build_model(dataset,emb_size=1000,use_model_name=False)
        data_validation = dataset.get_custom_numpy_dataset(mode="valid")
        # 根据参数决定是否从文件中加载权重
        if model_name is not None:
            my_model = CusNorModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"])      
        my_model.numpy_predict(data_validation,trainer=None,epochs=self.n_epochs,verbose=True)
    
    def predict(self, dataset: TFTSeriesDataset):
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
            my_model = CusNorModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"])       
    
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=forecast_horizon, series=val_series_list,past_covariates=past_covariates)
        
        self.predict_show(val_series_list, pred_series_list, series_total)
  
        
        