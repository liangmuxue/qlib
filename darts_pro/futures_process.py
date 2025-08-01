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
        self.dataset = dataset
        dataset.provider_file = os.path.join(self.optargs["provider_uri"],"instruments",self.optargs["market"]+".txt")
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
        if self.type=="predict":
            self.predict(dataset)   
            return   
        if self.type=="predict_indus_and_detail":
            self.predict_indus_and_detail(dataset)   
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
            self.model.model.model_name = self.optargs["model_name"]
            self.model.batch_size = self.batch_size     
            self.model.model.monitor = monitor
            self.model.model.mode = self.type 
            self.model.model.step_mode = self.optargs["step_mode"]
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
            self.model.model.batch_size = self.batch_size   
            self.model.model.monitor = monitor
            self.model.model.train_step_mode = self.optargs["step_mode"]    
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
            step_mode = self.optargs["step_mode"]
            my_model = FuturesIndustryModel(
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
                    step_mode=step_mode,
                    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                    # pl_trainer_kwargs={"accelerator": "cpu","log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )
        if mode==2:  
            train_step_mode = self.optargs["step_mode"]
            my_model = FuturesIndustryDRollModel(
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
                    train_step_mode=train_step_mode,
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
                            "KLOW","KLOW2","KSFT","RSV5", 'STD5','QTLU5','CORD5','CNTD5','VSTD5','QTLUMA5','BETA5',
            'KURT5','SKEW5','CNTP5','CNTN5','SUMP5','CORR5','SUMPMA5','RANK5','RANKMA5']
        analysis_columns = ["RSV5","QTLU5","CNTN5","SUMPMA5","CCI5"]
        # analysis_columns = ["QTLUMA5",'QTLU5','IMXD5','SKEW5','KURT5','BULLS','RSV5','ATR5','AOS','STD5','SUMPMA5']
        # analysis_columns = ["rsv_diff","qtluma_diff",'qtlu_diff','cci_diff','bulls_diff','sumpma_diff']
        
        results = []
        price_col = 'diff_range'
        # price_col = 'CLOSE'
        custom_date = None
        # custom_date = [20221025]
        
        # 行业品种相关度横向比较
        for i in range(len(analysis_columns)):
            target_df = dataset.df_val
            if custom_date is not None:
                target_df = dataset.df_val[dataset.df_val['datetime_number'].isin(custom_date)]
            tar_col = analysis_columns[i]
            df_instrument = target_df[~target_df['instrument'].str.startswith('ZS_')]
            corr_data = df_instrument[[price_col,tar_col]].corr().values
            df_indus = target_df[(target_df['instrument'].str.startswith('ZS_'))&(target_df['instrument']!='ZS_ALL')&(target_df['instrument']!='ZS_JRQH')&(target_df['instrument']!='ZS_NMFI')]
            corr_indus_data = df_indus[[price_col,tar_col]].corr().values
            results.append([corr_data[0,1],corr_indus_data[0,1]])
        results = np.array(results).transpose(1,0)
        results = pd.DataFrame(results,index=['品种相关度','行业相关度'],columns=analysis_columns)
        pd.set_option('expand_frame_repr', False)
        print("指标走势与价格走势相关度:\n",results)
        
        # print(dataset.df_val[(dataset.df_val['datetime_number']>=20221020)&(dataset.df_val['datetime_number']<=20221028)&(dataset.df_val['instrument'].str.startswith('ZS_'))&(dataset.df_val['instrument']!='ZS_ALL')&(dataset.df_val['instrument']!='ZS_JRQH')&(dataset.df_val['instrument']!='ZS_NMFI')][['instrument','diff_range','rsv_diff','RSV5','CLOSE']])

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

    def build_pred_result(self,pred_date,dataset=None):    
        """根据预测区间参数进行预测，pred_range为二元数组，数组元素类型为date"""        

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]   
        
        if dataset is None:
            dataset = self.dataset
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        expand_length = 2 *(input_chunk_length + output_chunk_length)
         
        # 根据日期范围逐日进行预测，得到预测结果   
        start_date = pred_date
        # 同时需要延长集合时间
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,-1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        begin_day = valid_range[0]
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train":[total_range[0],prev_day],"valid":[begin_day,prev_day]}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 记录实际截止日期对应的序列编号最大值，后续与模拟数据进行区分
        time_idx_mapping = dataset.df_all.groupby("instrument")["time_idx"].max()
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 再次生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)   
        # 给每个品种序列放入实际最大编号   
        for series in val_series_transformed:
            time_idx = time_idx_mapping[time_idx_mapping.index==series.instrument_code].values[0]
            series.last_time_idx = time_idx
            
        best_weight = self.optargs["best_weight"]    
        self.model = FuturesIndustryModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.model.batch_size = self.batch_size     
        self.model.mode = "predict"
        self.model.model.monitor = None
        self.model.model.mode = "predict"
        self.model.model.train_sw_ins_mappings = self.model.train_sw_ins_mappings
        self.model.model.valid_sw_ins_mappings = self.model.valid_sw_ins_mappings   
        
        # 进行推理及预测，先fit再predict
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)               
        
        # 进行预测           
        pred_result = self.model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))
        
        return pred_result        
                             
    def predict(self, dataset,pred_range=None):
        """根据预测区间参数进行预测并进行评估，pred_range为二元数组，数组元素类型为date"""

        if pred_range is None:
            pred_range = dataset.kwargs["segments"]["test"] 
            
        start_date = pred_range[0]
        end_date = pred_range[1]
        trade_dates = np.array(get_tradedays(start_date,end_date)).astype(np.int)
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        pred_result_list = {}
        for pred_date in trade_dates:
            pred_result = self.build_pred_result(str(pred_date),dataset=dataset)
            pred_result_list[pred_date] = pred_result[pred_date]
            
        # 对预测结果进行评估
        pred_dates = np.array(list(pred_result_list.keys())).astype(np.int)
        # 取得实际需要的日期结果数据
        match_dates = np.intersect1d(trade_dates,pred_dates)
        pred_result_target = {}
        # 生成真实数据，以进行评估
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        last_day = get_tradedays_dur(end_date,3*output_chunk_length)      
        segments = {"train":[total_range[0],last_day],"valid":[valid_range[0],last_day]}  
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        df_target = dataset.df_all
        import_price_result = []
        for key in match_dates:
            pred_result_target[key] = pred_result_list[key]
            target_class_list = []
            for index,row in pred_result_list[key].iterrows():
                instrument = row['instrument']
                trend = row['top_flag']
                item_cur_idx = df_target[(df_target['instrument']==instrument)&(df_target['datetime_number']==key)]['time_idx'].values[0]
                df_item = df_target[(df_target['instrument']==instrument)&(df_target['time_idx']>=(item_cur_idx-1))]
                price_list = df_item['CLOSE'].values
                diff_range = (price_list[output_chunk_length] - price_list[0])/price_list[0]
                p_taraget_class = get_simple_class(diff_range)  
                if trend==0:
                    p_taraget_class = [3,2,1,0][p_taraget_class] 
                    diff_range = -diff_range
                target_class_list.append(p_taraget_class)
                import_price_result.append([key,instrument,trend,p_taraget_class,diff_range])
        import_price_result = np.array(import_price_result)
        import_price_result = pd.DataFrame(import_price_result,
            columns=["date","instrument","trend","result","yield_rate"])
        import_price_result['trend'] = import_price_result['trend'].astype(int)
        import_price_result['result'] = import_price_result['result'].astype(int)
        import_price_result['yield_rate'] = import_price_result['yield_rate'].astype(float)
        
        print("total yield:{}".format(import_price_result["yield_rate"].sum()))
        return import_price_result          

    def build_pred_result_2step(self,pred_date,dataset=None):    
        """使用二阶段模式，根据预测区间参数进行预测。pred_range为二元数组，数组元素类型为date"""        

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]   
        
        if dataset is None:
            dataset = self.dataset
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        expand_length = 2 *(input_chunk_length + output_chunk_length)
         
        # 根据日期范围逐日进行预测，得到预测结果   
        start_date = pred_date
        # 同时需要延长集合时间
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,-1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        begin_day = valid_range[0]
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train":[total_range[0],prev_day],"valid":[begin_day,prev_day]}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 记录实际截止日期对应的序列编号最大值，后续与模拟数据进行区分
        time_idx_mapping = dataset.df_all.groupby("instrument")["time_idx"].max()
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 再次生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)   
        # 给每个品种序列放入实际最大编号   
        for series in val_series_transformed:
            time_idx = time_idx_mapping[time_idx_mapping.index==series.instrument_code].values[0]
            series.last_time_idx = time_idx
            
        best_weight = self.optargs["best_weight"]  
        # 首先使用总体模型生成总体趋势结果
        model_name_step1 = self.optargs["model_name_step1"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step1,work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        model.model.step_mode = 1
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)   
        # 先fit再赋值，则以当前数据集生成的mapping为准
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings                     
        model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                                    batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))        
        # 再用第二阶段模型生成实际品种结果
        model_name_step2 = self.optargs["model_name_step2"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step2,work_dir=self.optargs["work_dir"],
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        model.model.step_mode = 2
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        # 进行推理及预测，先fit再predict
        model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)    
        # 先fit再赋值，则以当前数据集生成的mapping为准           
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings          
        pred_result = model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))
        
        return pred_result 
    
    def predict_indus_and_detail(self, dataset,pred_range=None):
        """使用二阶段模型，根据预测区间参数进行预测并进行评估，pred_range为二元数组，数组元素类型为date"""

        if pred_range is None:
            pred_range = dataset.kwargs["segments"]["test"] 
            
        start_date = pred_range[0]
        end_date = pred_range[1]
        trade_dates = np.array(get_tradedays(start_date,end_date)).astype(np.int)
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        pred_result_list = {}
        for pred_date in trade_dates:
            pred_result = self.build_pred_result_2step(str(pred_date),dataset=dataset)
            pred_result_list[pred_date] = pred_result[pred_date]
            
        # 对预测结果进行评估
        pred_dates = np.array(list(pred_result_list.keys())).astype(np.int)
        # 取得实际需要的日期结果数据
        match_dates = np.intersect1d(trade_dates,pred_dates)
        pred_result_target = {}
        # 生成真实数据，以进行评估
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        last_day = get_tradedays_dur(end_date,3*output_chunk_length)      
        segments = {"train":[total_range[0],last_day],"valid":[valid_range[0],last_day]}  
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        df_target = dataset.df_all
        import_price_result = []
        for key in match_dates:
            pred_result_target[key] = pred_result_list[key]
            target_class_list = []
            for index,row in pred_result_list[key].iterrows():
                instrument = row['instrument']
                trend = row['top_flag']
                item_cur_idx = df_target[(df_target['instrument']==instrument)&(df_target['datetime_number']==key)]['time_idx'].values[0]
                df_item = df_target[(df_target['instrument']==instrument)&(df_target['time_idx']>=(item_cur_idx-1))]
                price_list = df_item['CLOSE'].values
                diff_range = (price_list[output_chunk_length] - price_list[0])/price_list[0]
                p_taraget_class = get_simple_class(diff_range)  
                if trend==0:
                    p_taraget_class = [3,2,1,0][p_taraget_class] 
                    diff_range = -diff_range
                target_class_list.append(p_taraget_class)
                import_price_result.append([key,instrument,trend,p_taraget_class,diff_range])
        import_price_result = np.array(import_price_result)
        import_price_result = pd.DataFrame(import_price_result,
            columns=["date","instrument","trend","result","yield_rate"])
        import_price_result['trend'] = import_price_result['trend'].astype(int)
        import_price_result['result'] = import_price_result['result'].astype(int)
        import_price_result['yield_rate'] = import_price_result['yield_rate'].astype(float)
        
        print("total yield:{}".format(import_price_result["yield_rate"].sum()))
        return import_price_result    
           
                        