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

class TftNumpyModel(BaseNumpyModel):
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
        dataset: TFTDataset
    ):
        if self.type.startswith("data_aug"):
            # 只进行数据处理模式
            self.build_aug_data(dataset)
            return      
        
        if self.type.startswith("data_view"):
            # 只进行数据处理模式
            self.data_view(dataset)
            return      
        if self.type.startswith("train_dataframe"):
            # 只进行数据处理模式
            self.train_dataframe(dataset)
            return               
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
            self.model = TFTCusModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
        self.model.fit(data_train,data_validation,trainer=None,epochs=self.n_epochs,verbose=True)
    
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
               
        categorical_embedding_sizes = {"dayofweek": 5,dataset.col_def["group_column"]: emb_size}    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        model_name = self.optargs["model_name"]
        if not use_model_name:
            model_name = None
        my_model = TFTCusModel(
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
        
    def predict_numpy(self, dataset: TFTDataset):
        if self.type!="predict":
            return 
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
           
        model_name = self.optargs["model_name"]
        my_model = self._build_model(dataset,emb_size=1000,use_model_name=False)
        data_validation = dataset.get_custom_numpy_dataset(mode="valid")
        # 根据参数决定是否从文件中加载权重
        if model_name is not None:
            my_model = TFTCusModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"])      
        my_model.numpy_predict(data_validation,trainer=None,epochs=self.n_epochs,verbose=True)

    def train_dataframe(self, dataset: TFTSeriesDataset):
         
        def view_df(df,target_title):
            viz_input = TensorViz(env="data_hist")
            view_data = df[["label"]].values
            viz_input.viz_matrix_var(view_data,win=target_title,title=target_title)  
                    
        df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
        view_df(df,"ori_label")
        # 使用后5天的移动平均值作为目标数值
        df["label"]  = df.groupby("instrument")["label"].shift(5).rolling(window=5,min_periods=1).mean()   
        df = df.dropna()    
        view_df(df,"mean_label")
        process(df)
            
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
        
    def data_view(self,dataset):
        data_path = self.optargs["data_path"]
        wave_period = self.optargs["wave_period"]
        forecast_horizon = self.optargs["forecast_horizon"]
        aug_type = self.optargs["aug_type"]
        low_threhold = self.optargs["low_threhold"]
        high_threhold = self.optargs["high_threhold"]
        over_time = self.optargs["over_time"]
        
        data = np.load(data_path,allow_pickle=True)       
        data_filter = DataFilter() 
        
        if aug_type == "combine":
            # 通过列排序规则，取得目标数据对应下标
            target_index = dataset.get_target_column_index()
            # 分别取得涨幅较大以及跌幅较大的数据
            low_data = data_filter.get_data_with_threhold(data,target_index,wave_threhold_type="less",threhold=low_threhold,
                                                          wave_period=wave_period,check_length=forecast_horizon,over_time=over_time)
            high_data = data_filter.get_data_with_threhold(data,target_index,wave_threhold_type="more",threhold=high_threhold,
                                                          wave_period=wave_period,check_length=forecast_horizon,over_time=over_time)
            nor_size = (low_data.shape[0] + high_data.shape[0])//2
            nor_index = np.random.randint(1,data.shape[0],(nor_size,))
            # 参考高低涨幅数据量，取得普通数据量，合并为目标数据
            nor_data = data[nor_index,:,:]
            combine_data = np.concatenate((low_data,high_data,nor_data),axis=0)
            self.numpy_data_view(dataset,combine_data)

     
        
        