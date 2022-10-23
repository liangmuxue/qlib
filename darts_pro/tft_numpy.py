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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

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
from darts_pro.data_extension.custom_model import TFTCusModel

class TftNumpyModel(Model):
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
        if self.type.startswith("data_aug"):
            # 只进行数据处理模式
            self.build_aug_data(dataset)
            return      
        if self.type.startswith("data_view"):
            # 只进行数据处理模式
            self.data_view(dataset)
            return           
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
            return       
              
        # 生成tft时间序列训练数据集
        data_train = dataset.get_custom_numpy_dataset(mode="train")
        data_validation = dataset.get_custom_numpy_dataset(mode="valid")
        # 使用股票代码数量作为embbding长度
        # emb_size = np.unique(dataset.data[:,:,dataset.get_target_column_index()])
        emb_size = 1000
        self.model = self._build_model(dataset,emb_size)
        self.model.fit(data_train,data_validation,trainer=None,verbose=True)
    
    def _build_model(self,dataset,emb_size):
        optimizer_cls = torch.optim.Adam
        scheduler = CosineAnnealingLR
        scheduler_config = {
            "T_max": 5, 
            "eta_min": 0,
        }     
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
        my_model = TFTCusModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.optargs["forecast_horizon"],
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=4096,
            n_epochs=self.n_epochs,
            add_relative_index=False,
            add_encoders=None,
            categorical_embedding_sizes=categorical_embedding_sizes,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),  # QuantileRegression is set per default
            # loss_fn=MSELoss(),
            random_state=42,
            # model_name="tft",
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
        
    def predict(self, dataset: TFTDataset):
        if self.type!="predict":
            return 
       
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        test_ds = dataset.get_ts_dataset(df_test,mode="valid")
        test_loader = test_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)        
        best_tft = OptimizeHyperparameters(clean_mode=True,model_path=self.optargs['weight_path'],load_weights=True,
                                           trial_no=self.optargs['best_trial_no'],
                                           epoch_no=self.optargs['best_ckpt_no']).get_tft(fig_save_path=self.fig_save_path,viz=self.optargs['viz'])
        actuals = torch.cat([y[0] for x, y in iter(test_loader)])
        predictions, x, pred_ori = best_tft.predict(test_loader,return_x=True,return_ori_outupt=True)   
        loss = (actuals - predictions).abs().mean()
        print("loss is:",loss)        
        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);  
        return pd.Series(np.concatenate(predictions), index=df_test.get_index())
    
    def build_aug_data(self,dataset):
        save_file_path = self.optargs["save_path"]
        group_column = dataset.col_def["group_column"]
        target_column = dataset.col_def["target_column"]
        forecast_horizon = self.optargs["forecast_horizon"]
        wave_threhold = self.optargs["wave_threhold"]
        over_time = self.optargs["over_time"]
        wave_threhold_type = self.optargs["wave_threhold_type"]
        
        df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 股票代码(即分组字段)转换为数值型
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')
        # 拼接所有需要使用的字段
        columns = dataset.get_seq_columns()
        df = df[columns]
        data_len = df.shape[0]
              
        data_filter = DataFilter()
        # 使用后几天（根据预测长度而定）的移动平均值作为目标数值
        df[target_column]  = df.groupby(group_column)[target_column].shift(-forecast_horizon).rolling(window=forecast_horizon,min_periods=1).mean()
        df = df.dropna().reset_index(drop=True)
        # 按照规则进行数据筛选
        wave_data = data_filter.filter_wave_data(df, target_column=target_column, group_column=group_column,
                                                 wave_threhold_type=wave_threhold_type,wave_threhold=wave_threhold,over_time=over_time)
        print("wave_data",wave_data.shape)
        np.save(save_file_path,wave_data)  
        
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
        viz_input = TensorViz(env="data_hist") 
        start = wave_period - forecast_horizon
        if aug_type == "combine":
            value_combine = None
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
            # 最后一列为观察数值,并且时间轴只使用后5个数据
            data_index = dataset.get_target_column_index()
            value = combine_data[:,start:,data_index]
            v_title = "aug_data"    
            viz_input.viz_data_hist(value.reshape(-1),numbins=10,win=v_title,title=v_title)  
            # 随机取得某些数据，并显示折线图
            for i in random_int_list(1,combine_data.shape[0],10):
                title = "line_{}".format(i)
                viz_input.viz_matrix_var(combine_data[i,:,target_index:target_index+1],win=title,title=title)

        
    
                  
        