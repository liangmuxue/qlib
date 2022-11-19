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
from darts import TimeSeries, concatenate

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
from darts_pro.data_extension.custom_model import TFTCusModel,TFTExtModel
from darts_pro.tft_series_dataset import TFTSeriesDataset
from .base_process import BaseNumpyModel
from numba.core.types import none

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
        self.series_data_view(dataset,train_series_transformed,past_convariates=past_convariates,title="train_target")
        self.series_data_view(dataset,val_series_transformed,past_convariates=None,title="val_target")
        
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        load_weight = self.optargs["load_weight"]
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
            # self.model = self.build_exp_model()
        # datapath = "custom/data/darts/dump_data"
        # train_series_transformed[0].to_pickle(datapath + "/train.pkl")
        # future_convariates[0].to_pickle(datapath + "/future.pkl")
        # val_series_transformed[0].to_pickle(datapath + "/val.pkl")
        # past_convariates[0].to_pickle(datapath + "/past.pkl")
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 trainer=None,epochs=self.n_epochs,verbose=True)
    
    def build_exp_model(self):
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
        input_chunk_length = 25
        forecast_horizon = 5
        n_epochs = 100
        my_model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=forecast_horizon,
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=128,
            n_epochs=n_epochs,
            model_name="Stock_TFT",
            add_relative_index=True,
            add_encoders=None,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),  # QuantileRegression is set per default
            # loss_fn=MSELoss(),
            random_state=42,
            log_tensorboard=True,
            force_reset=True,
            save_checkpoints=True,
            work_dir="custom/data/darts",
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}     
        )
        return my_model
                
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
        categorical_embedding_sizes = None    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        model_name = self.optargs["model_name"]
        if not use_model_name:
            model_name = None
        my_model = TFTExtModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=self.optargs["forecast_horizon"],
            hidden_size=64,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            add_relative_index=True,
            add_encoders=None,
            # categorical_embedding_sizes=categorical_embedding_sizes,
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
            # lr_scheduler_cls=scheduler,
            # lr_scheduler_kwargs=scheduler_config,
            # optimizer_cls=optimizer_cls,
            # optimizer_kwargs={"lr": 1e-2,"weight_decay":1e-4},
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
        # my_model = self._build_model(dataset,emb_size=1000,use_model_name=False)
        series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.get_series_data(type="test")
        # series_transformed,val_series_transformed,past_convariates,future_convariates = self.build_fake_data(dataset)
        self.series_data_view(dataset,series_transformed,past_convariates=past_convariates,
                              future_convariates=future_convariates,title="pred_target")
        
        # 根据参数决定是否从文件中加载权重
        if model_name is not None:
            my_model = TFTExtModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"],best=True)  
        # 需要进行fit设置
        my_model.fit(series_transformed,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,verbose=True,epochs=-1)            
        #
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=forecast_horizon, series=series_transformed,num_samples=200,
                                            past_covariates=past_convariates,future_covariates=future_convariates)

        # # 对验证集进行预测，得到预测结果   
        # pred_series_list = my_model.predict(n=5, num_samples=200)
               
        self.predict_show(val_series_transformed,pred_series_list, series_transformed)
        # self.predict_fake_show(val_series_transformed,pred_series_list, series_transformed)
    
    def build_fake_data(self,dataset):
        series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.get_series_data(type="test")
        series_transformed = series_transformed[0]
        val_series_transformed = val_series_transformed[0]
        past_convariates = past_convariates[0]
        future_convariates = future_convariates[0]
        total_cut_off = 2441
        val_cut_off = 2391
        # val_series_transformed = val_series_transformed[:val_cut_off]
        # series_transformed = series_transformed[:val_cut_off]
        # past_convariates = past_convariates[:total_cut_off]
        # future_convariates = future_convariates[:total_cut_off]
        # datapath = "custom/data/darts/dump_data"
        # series_transformed = TimeSeries.from_pickle(datapath + "/train.pkl")
        # future_convariates = TimeSeries.from_pickle(datapath + "/future.pkl")
        # val_series_transformed = TimeSeries.from_pickle(datapath + "/val.pkl")
        # past_convariates = TimeSeries.from_pickle(datapath + "/past.pkl")         
        return series_transformed,val_series_transformed,past_convariates,future_convariates
    
    def view_df(self,df,target_title):
        viz_input = TensorViz(env="data_hist")
        view_data = df[["label"]].values
        viz_input.viz_matrix_var(view_data,win=target_title,title=target_title)  

    def series_data_view(self,dataset,series_list,past_convariates=None,future_convariates=None,title="train_data"): 
        target_column = dataset.get_target_column()
        past_columns = dataset.get_past_columns()
        time_column = dataset.get_time_column()
        viz_input = TensorViz(env="data_hist") 
        
        df_total = None
        for series in series_list:
            df_item = series.pd_dataframe()
            if df_total is None:
                df_total = df_item
            else:
                df_total = pd.concat([df_total,df_item])
        # 标签数据分布图
        viz_input.viz_data_hist(df_total.values,numbins=10,win=title,title=title) 
        data_view_len = self.optargs["wave_period"]
        if past_convariates is None:
            return
        # 随机取得某些数据，并显示折线图
        for i in range(1):
            # 标签数据折线图
            df_item = series_list[i].pd_dataframe()
            past_df = past_convariates[i].pd_dataframe()
            print("past_df shape",past_df.shape)
            # viz_input.viz_matrix_var(view_data_target,win=target_title,title=target_title)              
            sub_title = title + "_past_{}".format(i)
            view_data_past = past_df[past_columns].values[:data_view_len,:]
            viz_input.viz_matrix_var(view_data_past,win=sub_title,title=sub_title,names=past_columns)        
 
    def predict_fake_show(self,val_series,pred_series,series_train):       
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"       
        forecast_horizon = self.optargs["forecast_horizon"]    
        # 整个序列比较多，只比较某几个序列
        figsize = (9, 6)
        # plot actual series
        plt.figure(figsize=figsize)
        actual_series = series_train.concatenate(val_series)
        actual_series[2400: pred_series.end_time()].plot(label="actual")
    
        # plot prediction with quantile ranges
        pred_series.plot(
            low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        )
        pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    
        plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
        plt.legend()
        plt.savefig('custom/data/darts/result_view/eval_exp.jpg')
        print("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
               
    def predict_show(self,val_series_list,pred_series_list,series_train):       
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"       
        forecast_horizon = self.optargs["forecast_horizon"]    
        # 整个序列比较多，只比较某几个序列
        figsize = (9, 6)
        # 创建比较序列，后面保持所有，或者自己指定长度
        actual_series_list = []
        for index,ser_val in enumerate(val_series_list):
            ser_train = series_train[index]
            ser_total = ser_train.concatenate(ser_val)
            # 从数据集后面截取一定长度的数据，作为比较序列
            actual_series = ser_total[
                ser_total.end_time() - (3 * forecast_horizon - 1) * ser_total.freq : 
            ]
            actual_series_list.append(ser_total)   
        mape_all = 0
        r = 3
        for i in range(len(val_series_list)):
            pred_series = pred_series_list[i]
            var_series = val_series_list[i]
            actual_series = actual_series_list[i]
            # 分位数范围显示
            pred_series.plot(
                low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
            )
            pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
            # 与实际的数据集进行比较，比较的是两个数据集的交集
            mape_item = mape(var_series, pred_series)
            mape_all = mape_all + mape_item
            if i<r:
                plt.figure(figsize=figsize)
                # 实际数据集的结尾与预测序列对齐
                pred_series.plot(label="forecast")            
                actual_series[pred_series.end_time()- 25 : pred_series.end_time()].plot(label="actual")           
                plt.title("ser_{},MAPE: {:.2f}%".format(i,mape_item))
                plt.legend()
                plt.savefig('{}/result_view/eval_{}.jpg'.format(self.optargs["work_dir"],i))
                plt.clf()   
        mape_mean = mape_all/len(val_series_list)
        print("mape_mean:",mape_mean)            