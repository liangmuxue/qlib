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
from darts_pro.data_extension.futures_model import FuturesModel,FuturesIndustryModel,FuturesIndustryDRollModel
from darts_pro.tft_futures_dataset import TFTFuturesDataset

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays_dur,get_tradedays
from .tft_process_dataframe import TftDataframeModel 
from darts_pro.data_extension.series_data_utils import StatDataAssis
from sklearn.preprocessing import MinMaxScaler
from darts_pro.data_extension.futures_togather_dataset import FuturesTogatherDataset
from darts_pro.data_extension.futures_industry_dataset import FuturesIndustryDataset
from tft.class_define import CLASS_SIMPLE_VALUES,get_simple_class

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
        if self.type.startswith("fit_futures_industry"):
            self.fit_futures_industry(dataset)
            return        
        if self.type.startswith("fit_futures_droll_industry"):
            self.fit_futures_droll_industry(dataset)
            return            
        if self.type.startswith("pred_futures_industry"):
            self.fit_futures_industry(dataset)
            return            
        if self.type.startswith("pred_futures_togather"):
            self.fit_futures_togather(dataset)
            return     
        if self.type.startswith("pred_futures_droll_industry"):
            self.fit_futures_droll_industry(dataset)
            return            
        if self.type.startswith("data_corr"):
            self.data_corr(dataset)   
            return     
        if self.type.startswith("predict"):
            self.predict(dataset)   
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
            self.model.model.train_sw_ins_mappings = self.model.train_sw_ins_mappings
            self.model.model.valid_sw_ins_mappings = self.model.valid_sw_ins_mappings            
            trainer.validate(model=model,dataloaders=val_loader)
        else:
            self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=0)  


    def fit_futures_industry(
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
        
        # map_location = torch.device("cpu")
        map_location = None
        
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            self.model = FuturesIndustryModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=map_location)
            self.model.batch_size = self.batch_size     
            self.model.model.monitor = monitor
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=1) 
            self.model.monitor = monitor        
        
        self.model.mode = self.type  
        
        if self.type=="pred_futures_industry":  
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model,train_loader,val_loader = self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)
            self.model.model.mode = self.type  
            self.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            self.model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            # self.model.valid_sw_ins_mappings = val_loader.dataset.sw_ins_mappings
            # self.model.model.valid_sw_ins_mappings = val_loader.dataset.sw_ins_mappings  
            trainer.validate(model=model,dataloaders=val_loader)
        else:
            self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8)  

    def fit_futures_droll_industry(
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
        
        # map_location = torch.device("cpu")
        map_location = None
        
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            self.model = FuturesIndustryDRollModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=map_location)
            self.model.batch_size = self.batch_size     
            self.model.model.monitor = monitor
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=2) 
            self.model.monitor = monitor        
        
        self.model.mode = self.type  
        
        if self.type=="pred_futures_droll_industry":  
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model,train_loader,val_loader = self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)
            self.model.model.mode = self.type  
            self.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            self.model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            # self.model.valid_sw_ins_mappings = val_loader.dataset.sw_ins_mappings
            # self.model.model.valid_sw_ins_mappings = val_loader.dataset.sw_ins_mappings  
            trainer.validate(model=model,dataloaders=val_loader)
        else:
            self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8)  

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
        target_mode = self.optargs["target_mode"] 
        scale_mode = self.optargs["scale_mode"] 
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
                    cut_len=self.optargs["cut_len"],
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
                    target_mode=target_mode,
                    scale_mode=scale_mode,
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
        if mode==1:  
            my_model = FuturesIndustryModel(
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
                    target_mode=target_mode,
                    scale_mode=scale_mode,
                    filter_conv_index=filter_conv_index,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    model_type=model_type,
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"accelerator": "cpu","log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )
        if mode==2:  
            my_model = FuturesIndustryDRollModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=self.optargs["forecast_horizon"],
                    rolling_size=self.optargs["rolling_size"],
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
                    target_mode=target_mode,
                    scale_mode=scale_mode,
                    filter_conv_index=filter_conv_index,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    model_type=model_type,
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"accelerator": "cpu","log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )                        
        return my_model
                    
    def data_corr(
        self,
        dataset: TFTFuturesDataset,
    ):
        """对数据进行相关性分析"""
        
        scale_mode = self.optargs["scale_mode"]
        pred_len = self.optargs["forecast_horizon"]
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
            
        output_chunk_length = self.optargs["forecast_horizon"]
        input_chunk_length = self.optargs["wave_period"] - output_chunk_length
        past_split = self.optargs["past_split"] 
        
        custom_dataset_valid = FuturesIndustryDataset(
                    target_series=val_series_transformed,
                    covariates=past_convariates,
                    future_covariates=future_convariates,
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=output_chunk_length,
                    max_samples_per_ts=None,
                    use_static_covariates=True,
                    target_num=len(past_split),
                    scale_mode=scale_mode,
                    mode="valid"
                )            
        data_assis = StatDataAssis()
        col_list = dataset.col_def["col_list"] + ["label"]
        analysis_columns = ["label_ori","REV5","IMAX5","QTLUMA5","OBV5","CCI5","KMID","KLEN","KMID2","KUP","KUP2",
                            "KLOW","KLOW2","KSFT","KSFT2", 'STD5','QTLU5','CORD5','CNTD5','VSTD5','QTLUMA5','BETA5',
            'KURT5','SKEW5','CNTP5','CNTN5','SUMP5','CORR5','SUMPMA5','RANK5','RANKMA5']
        analysis_columns = ["price","QTLUMA5","CNTN5","SUMPMA5"]
        analysis_columns = ["price","QTLUMA5",'QTLU5','IMXD5','SKEW5','KURT5','BULLS','RSV5','ATR5','AOS','STD5','SUMPMA5']
        # 利用dataloader进行数据拼装
        val_loader = DataLoader(
                custom_dataset_valid,
                batch_size=1024,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_fn,
            )
        past_conv_index = past_split[0]
        past_columns = dataset.get_past_columns()
        past_columns = past_columns[past_conv_index[0]:past_conv_index[1]]       
        combine_columns = ["price"] + past_columns + dataset.get_target_column() 
        analysis_data = None
        print("total len：",len(val_loader))
        # 遍历数据集，并按照批次进行计算，汇总后取得平均值
        for index,batch_data in enumerate(val_loader):
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                past_future_covariates,
                future_target,
                target_class,
                past_round_targets,
                _,
                future_round_targets,
                target_info
            ) = batch_data
            # if index>5:
            #     break
            index_filter = []
            # 筛选指定日期数据,由于是3D模式，因此先把形状调整一下]
            target_class_flat = target_class.flatten()
            keep_index = np.where(target_class_flat>-1)[0]
            future_target_flat = future_target.reshape(-1,future_target.shape[-2],future_target.shape[-1])
            future_target = future_target_flat[keep_index]
            price_array_list = [] 
            for i,ti in enumerate(target_info):
                for ts in ti: 
                    # 忽略空值
                    if ts is not None:
                        # 计算价格差的时候，把前一日期也包括进来
                        price_array = np.array(ts["price_array"][-pred_len-1:])  
                        price_array_list.append(price_array)
                future_start_datetime = ti[0]["future_start_datetime"]
                if future_start_datetime<20220401 and future_start_datetime>=20220301 or True:
                    index_filter.append(i)    
            price_array = np.stack(price_array_list)
            price_range = ((price_array[:,1:] - price_array[:,:-1])/price_array[:,:-1])*10
            price_range = np.expand_dims(price_range,-1)
            # 对价格归一化后进行比较
            price_array_scale = MinMaxScaler().fit_transform(price_array[:,1:].transpose(1,0)).transpose(1,0)
            price_array_scale = np.expand_dims(price_array_scale,-1)
            past_future_covariates = past_future_covariates.reshape(past_future_covariates.shape[0]*past_future_covariates.shape[1],past_future_covariates.shape[2],-1)
            past_future_covariates = past_future_covariates.numpy()[keep_index]
            past_future_covariates_item = past_future_covariates[...,past_conv_index[0]:past_conv_index[1]]  
            past_future_covariates_item_trans = past_future_covariates_item.transpose(1,0,2)
            past_future_covariates_item_trans = past_future_covariates_item_trans.reshape(past_future_covariates_item_trans.shape[0],-1)
            past_future_covariates_item = MinMaxScaler().fit_transform(past_future_covariates_item_trans).reshape(past_future_covariates_item_trans.shape[0],past_future_covariates_item.shape[0],past_future_covariates_item.shape[2]).transpose(1,0,2)
            # 整合未来价格、未来目标值以及过去协变量在未来的数值，后续统一进行相关性比较
            analysis_batch = np.concatenate([price_array_scale,past_future_covariates_item,future_target],-1)   
            # 计算单个批次的数据
            df_corr_batch,df_price_batch,range_cls_stat = data_assis.custom_data_corr_analysis(analysis_batch,fit_columns=combine_columns,
                            analysis_columns=analysis_columns,target_class=target_class[index_filter],price_range=price_range)
        # 汇总所有批次的数据
        if analysis_data is None:
            analysis_data = [df_corr_batch,df_price_batch,[range_cls_stat]]
        else:
            analysis_data[0] = pd.concat([analysis_data[0],df_corr_batch])
            analysis_data[1] = pd.concat([analysis_data[1],df_price_batch])
            analysis_data[2] = analysis_data[2] + [range_cls_stat]
        print("process index:{}".format(index))           
         
        analysis_data_mean = [analysis_data[0].mean(),analysis_data[1].mean()]
        analysis_data[2] = np.stack(analysis_data[2])
        hitrate_mean = np.mean(analysis_data[2],axis=0)
        hitrate_mean = pd.DataFrame(hitrate_mean,columns=["cls_{}".format(k) for k in range(4)],index=analysis_columns[1:])
                
        print("指标走势与价格走势相关度:\n",analysis_data_mean[0])
        print("指标涨跌幅与价格涨跌幅相关度:\n",analysis_data_mean[1])
        print("hitrate price:\n",hitrate_mean)

    @staticmethod           
    def _batch_collate_fn(batch):
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
            elif isinstance(elem, List):
                aggregated.append([sample[i] for sample in batch])
            else:
                print("no match for:",elem.dtype)
        return tuple(aggregated)   
                                
    def predict(self, dataset,pred_range=None):
        """根据预测区间参数进行预测，pred_range为二元数组，数组元素类型为date。
           需要两阶段进行
        """

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]   
        
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
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,-1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        begin_day = get_tradedays_dur(end_date,-2*input_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        valid_range[0] = begin_day 
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train":[total_range[0],prev_day],"valid":[begin_day,prev_day]}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 在此生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)      
        
        # 分别加载两阶段模型，依次进行推理
        best_weight = self.optargs["best_weight"]    
        self.model = FuturesIndustryModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.droll_model = FuturesIndustryDRollModel.load_from_checkpoint(self.optargs["droll_model_name"],work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)        
        self.model.batch_size = self.batch_size     
        self.model.mode = "predict"
        self.model.model.monitor = None
           
        # 第一阶段，产生指数数据以及行业数据，执行validate,产生中间结果      
        self.droll_model.mode = "pred_futures_droll_industry" 
        self.droll_model.model.mode = "pred_futures_droll_industry"  
        self.droll_model.batch_size = self.batch_size  
        trainer,model,train_loader,val_loader = self.droll_model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)        
        self.droll_model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
        self.droll_model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings        
        trainer.validate(model=model,dataloaders=val_loader)        
        
        # 第二阶段，进行推理及预测，先fit再predict
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)               
        
        # 进行预测           
        pred_result = self.model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=self.batch_size,num_loader_workers=0)
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
        # 生成真实数据，以进行评估
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        df_target = dataset.df_all
        for key in match_dates:
            pred_result_target[key] = pred_result[key]
            items = pred_result[key][0]
            trend = pred_result[key][1]
            target_class_list = []
            for item in items:
                item_cur_idx = df_target[(df_target['instrument']==item[-1])&(df_target['datetime_number']==key)]['time_idx'].values[0]
                df_item = df_target[(df_target['instrument']==item[-1])&(df_target['time_idx']>=(item_cur_idx-1))]
                price_list = df_item['CLOSE'].values
                price_range = (price_list[4] - price_list[0])/price_list[0]
                p_taraget_class = get_simple_class(price_range)  
                if trend==0:
                    p_taraget_class = [3,2,1,0][p_taraget_class] 
                target_class_list.append(p_taraget_class)
        return pred_result_target          
        
                        