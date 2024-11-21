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
from qlib.data.dataset.handler import DataHandlerLP

from cus_utils.tensor_viz import TensorViz
from darts_pro.data_extension.futures_model import FuturesModel
from darts_pro.tft_futures_dataset import TFTFuturesDataset

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays,date_string_transfer
from .tft_process_dataframe import TftDataframeModel 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from cus_utils.log_util import AppLogger
logger = AppLogger()

class FuturesProcessModel(TftDataframeModel):

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
                
        if self.type.startswith("fit_futures_togather"):
            self.fit_futures_togather(dataset)
            return   
             
        print("Do Nothing")

    def fit_futures_togather(
        self,
        dataset: TFTFuturesDataset,
    ):
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
            
        df_data_path = os.path.join(self.batch_file_path,"main_data.pkl")
        df_train_path = os.path.join(self.batch_file_path,"df_train.pkl")
        df_valid_path = os.path.join(self.batch_file_path,"df_valid.pkl")
        ass_train_path = os.path.join(self.batch_file_path,"ass_data_train.pkl")
        ass_valid_path = os.path.join(self.batch_file_path,"ass_data_valid.pkl")
            
        if self.load_dataset_file:
            # 加载主要序列数据和辅助数据
            with open(df_data_path, "rb") as fin:
                train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
                    pickle.load(fin)   
            with open(ass_train_path, "rb") as fin:
                ass_data_train = pickle.load(fin)  
            with open(ass_valid_path, "rb") as fin:
                ass_data_valid = pickle.load(fin) 
            with open(df_train_path, "rb") as fin:
                dataset.df_train = pickle.load(fin)  
                dataset.prepare_inner_data(dataset.df_train)      
            with open(df_valid_path, "rb") as fin:
                dataset.df_val = pickle.load(fin)     
                dataset.prepare_inner_data(dataset.df_val)           
            global_var.set_value("ass_data_train",ass_data_train)
            global_var.set_value("ass_data_valid",ass_data_valid)
            global_var.set_value("load_ass_data",True)
        else:
            # 生成tft时间序列数据集,包括目标数据、协变量等
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
            # 保存序列数据
            if self.save_dataset_file:
                dump_data = (train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates)
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dump_data, fout)   
                # 还需要保存原始的DataFrame数据
                with open(df_train_path, "wb") as fout:
                    pickle.dump(dataset.df_train, fout)       
                with open(df_valid_path, "wb") as fout:
                    pickle.dump(dataset.df_val, fout)                                       
                global_var.set_value("ass_data_path",self.batch_file_path)
                global_var.set_value("load_ass_data",False)
                global_var.set_value("save_ass_data",True)
            else:
                global_var.set_value("load_ass_data",False)
                global_var.set_value("save_ass_data",False)  
            
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        # emb_size = 500
        load_weight = self.optargs["load_weight"]
        if "monitor" in self.optargs:
            monitor = dataset
        else:
            monitor = None
            
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            self.model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                             best=best_weight,batch_file_path=self.batch_file_path)
            self.model.batch_size = self.batch_size     
            self.model.mode = "train"
            self.model.model.monitor = monitor
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=0) 
            self.model.monitor = monitor        

                    
        if self.type=="pred_futures_togather":  
            self.model.mode = self.type
            self.model.model.mode = self.type        
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model,train_loader,val_loader = self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)
            trainer.validate(model=model,dataloaders=val_loader)
        else:
            self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=0)  



    def _build_model(self,dataset,emb_size=1000,use_model_name=True,mode=0):
        """生成模型"""
        
        log_every_n_steps = self.kwargs["log_every_n_steps"]
        optimizer_cls = torch.optim.Adam
        # optimizer_cls = torch.optim.SGD
        scheduler_config = self.kwargs["scheduler_config"]
        optimizer_kwargs = self.kwargs["optimizer_kwargs"]
        
        # scheduler = torch.optim.lr_scheduler.CyclicLR
        scheduler = torch.optim.lr_scheduler.LinearLR
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
            my_model = FuturesModel(
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
            
        return my_model
                    