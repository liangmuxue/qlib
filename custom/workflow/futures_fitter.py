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
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
from qlib.data.dataset.handler import DataHandlerLP

from cus_utils.tensor_viz import TensorViz
from darts_pro.data_extension.futures_model import FuturesModel,FuturesIndustryModel
from darts_pro.tft_futures_dataset import TFTFuturesDataset

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays,date_string_transfer
from darts_pro.data_extension.series_data_utils import StatDataAssis
from sklearn.preprocessing import MinMaxScaler
from darts_pro.data_extension.futures_togather_dataset import FuturesTogatherDataset
from trader.utils.date_util import get_tradedays,date_string_transfer,get_tradedays_dur

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from cus_utils.log_util import AppLogger
logger = AppLogger()

class FuturesFitter():
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

        self.batch_file_path = self.kwargs["batch_file_path"]
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
    def fit(
        self,
        dataset: TFTFuturesDataset,
    ):
        global_var.set_value("dataset", dataset)
        viz_data = TensorViz(env="viz_data")
        viz_result = TensorViz(env="viz_result")
        viz_result_detail = TensorViz(env="viz_result_detail")
        viz_result_fail = TensorViz(env="train_result_fail")
        global_var.set_value("viz_data",viz_data)
        global_var.set_value("viz_result",viz_result)
        global_var.set_value("viz_result_detail",viz_result_detail)
        global_var.set_value("viz_result_fail",viz_result_fail)
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)
        self.dataset = dataset
                
        if self.type=="backtest":
            # 回测模式，加载对应的模型权重
            self.load_model(dataset,mode="predict")     
        if self.type.startswith("pred"):
            # 预测模式，加载对应的模型权重
            self.load_model(dataset,mode="predict")        

    def load_model(self,dataset,mode=None):        
        best_weight = self.optargs["best_weight"]    
        self.model = FuturesIndustryModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.model.batch_size = self.batch_size     
        self.model.mode = mode
        self.model.model.monitor = None
        return self.model
     
    def predict(self, dataset=None,pred_range=None):
        """根据预测区间参数进行预测，pred_range为2元数组，数组元素类型为date"""

        if dataset is None:
            dataset = self.dataset

        if pred_range is None:
            pred_range = dataset.kwargs["segments"]["test"] 
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        expand_length = 2 *(input_chunk_length + output_chunk_length)
         
        # 根据日期范围逐日进行预测，得到预测结果   
        start_date = pred_range[0]
        end_date = pred_range[1]
        # 同时需要延长集合时间
        total_range = dataset.segments["train_total"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        begin_day = get_tradedays_dur(end_date,-2*input_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        valid_range[0] = begin_day 
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train_total":total_range,"valid":valid_range}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 生成模拟数据后重置日期区间
        valid_range[1] = last_day  
        total_range[1] = last_day        
        # 在此生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)         
        # 先fit再predict        
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)               
        
        # 进行预测           
        pred_result = self.model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=8,num_loader_workers=0)
        # 对预测结果进行评估
        if pred_result is None:
            print("{} pred_result None".format(pred_range))
        else:
            print("{} res len:{}".format(pred_range,len(pred_result)))
        
        # 取得实际需要的日期结果数据
        trade_dates = np.array(get_tradedays(start_date,end_date)).astype(np.int)
        pred_dates = np.array(list(pred_result.keys())).astype(np.int)
        match_dates = np.intersect1d(trade_dates,pred_dates)
        pred_result_target = {}
        for key in match_dates:
            pred_result_target[key] = pred_result[key]
        
        return pred_result_target      
    
        