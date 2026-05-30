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
import shap
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
from trader.utils.date_util import get_tradedays_dur,get_tradedays,get_next_month
from .tft_process_dataframe import TftDataframeModel 
from darts_pro.data_extension.series_data_utils import StatDataAssis
from sklearn.preprocessing import MinMaxScaler
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
        self.init_env(dataset)
        
        if self.type.startswith("fit_futures_togather"):
            self.fit_futures_togather(dataset)
            return   
        if self.type.startswith("fit_futures_industry"):
            self.fit_futures_industry(dataset)
            return        
        if self.type.startswith("fit_futures_bidi"):
            self.fit_futures_bidi(dataset)
            return    
        if self.type.startswith("fit_futures_trans"):
            self.fit_futures_trans(dataset)
            return         
        if self.type.startswith("fit_futures_tcn"):
            self.fit_futures_tcn(dataset)
            return                  
        if self.type.startswith("pred_futures_industry"):
            self.fit_futures_industry(dataset)
            return            
        if self.type.startswith("pred_futures_togather"):
            self.fit_futures_togather(dataset)
            return     
        if self.type.startswith("pred_futures_bidi"):
            self.fit_futures_bidi(dataset)
            return     
        if self.type.startswith("pred_futures_tcn"):
            self.fit_futures_tcn(dataset)
            return    
        if self.type.startswith("pred_futures_trans"):
            self.fit_futures_trans(dataset)
            return                      
        if self.type.startswith("analysis_model"):
            self.analysis_model(dataset)   
            return     
        if self.type=="predict":
            self.predict(dataset)   
            return   
        if self.type=="predict_indus_and_detail":
            self.predict_indus_and_detail(dataset)   
            return       
        if self.type.startswith("build_val_result"):
            self.build_val_result(dataset)
            return    
        if self.type.startswith("batch_pred_bidi"):
            self.batch_pred_bidi(dataset)
            return                                      
        print("Do Nothing")

    def init_env(self,dataset):
        
        self.dataset = dataset
        dataset.provider_file = os.path.join(self.optargs["provider_uri"],"instruments",self.optargs["market"]+".txt")
        global_var.set_value("dataset", dataset)
        viz_data = TensorViz(env="viz_data")
        viz_result = TensorViz(env="viz_result")
        viz_result_detail = TensorViz(env="viz_result_detail")
        viz_result_ext = TensorViz(env="viz_result_ext")
        global_var.set_value("viz_data",viz_data)
        global_var.set_value("viz_result",viz_result)
        global_var.set_value("viz_result_detail",viz_result_detail)
        global_var.set_value("viz_result_ext",viz_result_ext)
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)

    def fit_futures_trans(
        self,
        dataset: TFTFuturesDataset,
    ):
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        global_var.set_value("trend_threhold",self.optargs["trend_threhold"])
        train_data,val_data = dataset.build_series_data()
        train_series_transformed,past_convariates_train,future_convariates_train = train_data
        val_series_transformed,past_convariates_val,future_convariates_val = val_data
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)  
            
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        # emb_size = 500
        load_weight = self.optargs["load_weight"]
        # map_location = torch.device("cpu")
        device = self._build_device()
        
        outer_params = {'pred_weights':self.optargs["pred_weights"],'mode':self.type,'use_pcgrad':self.optargs['use_pcgrad'],
                        'top_num':self.optargs['loss_top_num'],'pred_top_num':self.optargs['pred_top_num'],
                        'opt_size':self.optargs['opt_size'],'candidate_inverse':self.optargs['candidate_inverse'],'pred_mode':self.optargs['pred_mode'],
                        'trend_threhold':self.optargs['trend_threhold'],
                        'pred_index_data_path':self.kwargs["pred_index_data_path"],'load_index_data':self.kwargs["load_index_data"],
                        'pred_cate_data_path':self.kwargs["pred_cate_data_path"],'load_cate_data':self.kwargs["load_cate_data"]
                        }
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            self.model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
            self.rebuild_model_params(self.model,model_name=self.optargs["model_name"])  
            self.model.model.set_outer_params(outer_params) 
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=1) 
        self.model.mode = self.type 
        self.model.set_outer_params(outer_params) 
        
        if self.type=="pred_futures_trans":  
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model,train_loader,val_loader = \
            self.model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=False)  
            self.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            self.model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            self.model.model.set_outer_params(outer_params) 
            trainer.validate(model=model,dataloaders=val_loader)
        else:
            trainer,model_inner,train_loader,val_loader= \
            self.model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=False)  

    def rebuild_model_params(self,model,model_name=None):
        
        model.model.model_name = model_name
        model.batch_size = self.batch_size     
        model.model.mode = self.type 
        model.model.step_mode = self.optargs["step_mode"]        
        
        
        gpu_params = self.gpus
        model.trainer_params['devices'] = gpu_params
        model.trainer_params['gpus'] = len(gpu_params)
        
    def _build_device(self):
        
        gpu_params = self.gpus
        cudas = ",".join([str(x) for x in gpu_params])
        device = "cuda:" + cudas
        return device
    
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
        gpu_params = self.gpus
        gpus_size = len(gpu_params)
        # 自定义回调函数
        lightning_callbacks = []
        if "lightning_callbacks" in  self.kwargs:
            lightning_callbacks_config = self.kwargs.get("lightning_callbacks", [])
            for config in lightning_callbacks_config:
                callback = init_instance_by_config(
                    config,
                )   
                lightning_callbacks.append(callback)          
                   
        pl_trainer_kwargs = {"accelerator": "cpu","log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks}    
        pl_trainer_kwargs = {"accelerator": "gpu","gpus":gpus_size, "strategy":"ddp", "devices": gpu_params,"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks}               
        if mode in [0,1,2]:  
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
                    opt_size=self.optargs["opt_size"],
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    pred_top_num=self.optargs["pred_top_num"],
                    task_weights=self.optargs["task_weights"],
                    main_task_seq=self.optargs["main_task_seq"],
                    grad_limits=self.optargs["grad_limits"],
                    pred_weights=self.optargs["pred_weights"],
                    # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )
            my_model.act_model_type = mode
        return my_model
                    
    def analysis_model(
        self,
        dataset: TFTFuturesDataset,
    ):
        """分析特征重要性"""
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        global_var.set_value("trend_threhold",self.optargs["trend_threhold"])
        train_data,val_data = dataset.build_series_data()
        train_series_transformed,past_convariates_train,future_convariates_train = train_data
        val_series_transformed,past_convariates_val,future_convariates_val = val_data
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)  
            
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        # emb_size = 500
        load_weight = self.optargs["load_weight"]
        # map_location = torch.device("cpu")
        device = self._build_device()
        
        outer_params = {'pred_weights':self.optargs["pred_weights"],'mode':self.type,'use_pcgrad':self.optargs['use_pcgrad'],
                        'top_num':self.optargs['loss_top_num'],'pred_top_num':self.optargs['pred_top_num'],
                        'opt_size':self.optargs['opt_size'],'candidate_inverse':self.optargs['candidate_inverse'],'pred_mode':self.optargs['pred_mode'],
                        'trend_threhold':self.optargs['trend_threhold'],
                        'pred_index_data_path':self.kwargs["pred_index_data_path"],'load_index_data':self.kwargs["load_index_data"],
                        'pred_cate_data_path':self.kwargs["pred_cate_data_path"],'load_cate_data':self.kwargs["load_cate_data"]
                        }
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            self.model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
            self.rebuild_model_params(self.model,model_name=self.optargs["model_name"])  
            self.model.model.set_outer_params(outer_params) 
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=1) 
        self.model.mode = self.type 
        self.model.set_outer_params(outer_params) 
        
        
        trainer,model,train_loader,val_loader = \
        self.model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                 max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=False)  
        self.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
        self.model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
        self.model.model.set_outer_params(outer_params) 

        real_model = self.model.model.sub_models[0]
        real_model.eval()        # SHAP 必须 eval
        for p in real_model.parameters():
            p.requires_grad = False  # 关闭梯度，提速防报错
        
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        # 背景数据：用少量训练集（10~100 个样本，算基准）
        cnt = 100
        past_covariate_total = []
        static_covariate_total = []
        historic_future_covariates_total = []
        for i in range(len(train_dataset)):
            data = train_dataset[i]
            past_covariate_total.append(data[1])
            historic_future_covariates_total.append(data[2])
            static_covariate_total.apend(data[4])
            if i>cnt:
                break
        past_covariate_total = torch.tensor(np.stack(past_covariate_total),dtype=torch.float32)     
        static_covariate_total = torch.tensor(np.stack(static_covariate_total),dtype=torch.float32)  
        historic_future_covariates_total = torch.tensor(np.stack(historic_future_covariates_total),dtype=torch.float32)   
        background = [past_covariate_total, historic_future_covariates_total, static_covariate_total] 
        
        cnt = 30
        past_covariate_total = []
        static_covariate_total = []
        historic_future_covariates_total = []        
        for i in range(len(val_dataset)):
            data = val_dataset[i]
            past_covariate_total.append(data[1])
            historic_future_covariates_total.append(data[2])
            static_covariate_total.apend(data[4])
            if i>cnt:
                break
        past_covariate_total = torch.tensor(np.stack(past_covariate_total),dtype=torch.float32)     
        static_covariate_total = torch.tensor(np.stack(static_covariate_total),dtype=torch.float32)  
        historic_future_covariates_total = torch.tensor(np.stack(historic_future_covariates_total),dtype=torch.float32)           
        
        # 待解释数据：选测试集一小部分
        to_explain = [past_covariate_total, historic_future_covariates_total, static_covariate_total] 
        # ---------- 选解释器（MLP/CNN 用 DeepExplainer） ----------
        explainer = shap.DeepExplainer(
            model=real_model,
            data=background  # 背景：tensor/numpy 都可
        )
        # ---------- 计算 SHAP 值 ----------
        # shap_values 形状：(n_samples, n_features)
        shap_values = explainer.shap_values(to_explain)

        print("shap_values length:", len(shap_values))
        print("feat1 SHAP shape:", shap_values[0].shape)
        print("feat2 SHAP shape:", shap_values[1].shape)
        print("feat3 SHAP shape:", shap_values[2].shape)        
        # ---------- 可视化（重要性 + 影响方向） ----------
        names1 = [f'num_{i}' for i in range(10)]
        names2 = [f'cat_{i}' for i in range(5)]
        names3 = [f'seq_{i}' for i in range(16)]
        
        # 特征重要性图（最常用）
        print("\n=== import 1 ===")
        shap.summary_plot(shap_values[0], to_explain[0].numpy(), feature_names=names1)
        
        print("\n=== import 2 ===")
        shap.summary_plot(shap_values[1], to_explain[1].numpy(), feature_names=names2)
        
        print("\n=== import 3 ===")
        shap.summary_plot(shap_values[2], to_explain[2].numpy(), feature_names=names3)        
                    

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

    def build_val_result(self,dataset=None):
        """生成验证结果,并整合保存"""
        
        self.fit_futures_industry(dataset)
        
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
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length,begin_date=int(start_date)) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 再次生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)   
        # 给每个品种序列放入实际最大编号   
        for series in val_series_transformed:
            time_idx = time_idx_mapping[time_idx_mapping.index==series.instrument_code].values[0]
            series.last_time_idx = time_idx
        
        device = self._build_device()  
        best_weight = self.optargs["best_weight"]    
        model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
        model_name = self.optargs["model_name"]  
        self.rebuild_model_params(model,model_name=model_name)  
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        
        # 进行推理及预测，先fit再predict
        model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)               
        
        # 通过植入属性的方式，设置预测品种个数
        if 'pred_top_num' in self.optargs:
            model.model.pred_top_num = self.optargs['pred_top_num']
        else:
            model.model.pred_top_num = 2
        # 进行预测           
        pred_result = model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
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

        step1_keep_col = dataset.col_def['step1_keep_col']
        step2_keep_col = dataset.col_def['step2_keep_col']
        target_mode_step1 = self.optargs["target_mode_step1"]  
        target_mode_step2 = self.optargs["target_mode_step2"]  
        scale_mode_step1 = self.optargs["scale_mode_step1"]  
        scale_mode_step2 = self.optargs["scale_mode_step2"]  
                                
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
        
        device = self._build_device()
        best_weight = self.optargs["best_weight"]  
        # 首先使用总体模型生成总体趋势结果
        model_name_step1 = self.optargs["model_name_step1"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step1,work_dir=self.optargs["work_dir"],device=device,
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.rebuild_model_params(model,model_name=model_name_step1)  
        model.model.step_mode = 1
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        
        # 2个阶段需要不同的目标值，在这里通过多配置的模式，并且动态改变val_series序列的列值来实现
        model.scale_mode = scale_mode_step1
        model.model.scale_mode = scale_mode_step1
        model.target_mode = target_mode_step1
        model.model.target_mode = target_mode_step1        
        target_column = np.array(dataset.col_def['target_column'])
        real_target_column = target_column[np.array(step1_keep_col)]
        rm_cols = np.setdiff1d(target_column, real_target_column, assume_unique=False).tolist()
        val_series_transformed_new = []
        for ser in val_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            val_series_transformed_new.append(ser_new)
        train_series_transformed_new = []
        for ser in train_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            train_series_transformed_new.append(ser_new)            
            
        model.fit(train_series_transformed_new, future_covariates=future_convariates, val_series=val_series_transformed_new,
                val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)   
        
        # 先fit再赋值，则以当前数据集生成的mapping为准
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings                     
        model.predict(series=val_series_transformed_new,past_covariates=past_convariates,future_covariates=future_convariates,
                                                    batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))        
        # 再用第二阶段模型生成实际品种结果
        model_name_step2 = self.optargs["model_name_step2"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step2,work_dir=self.optargs["work_dir"],device=device,
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.rebuild_model_params(model,model_name=model_name_step2)  
        model.model.step_mode = 2
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        
        # 2个阶段需要不同的目标值，在这里通过多配置的模式，并且动态改变val_series序列的列值来实现
        model.scale_mode = scale_mode_step2
        model.model.scale_mode = scale_mode_step2
        model.target_mode = target_mode_step2
        model.model.target_mode = target_mode_step2           
        target_column = np.array(dataset.col_def['target_column'])
        real_target_column = target_column[np.array(step2_keep_col)]
        rm_cols = np.setdiff1d(target_column, real_target_column, assume_unique=False).tolist()
        val_series_transformed_new = []
        for ser in val_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            val_series_transformed_new.append(ser_new)
        train_series_transformed_new = []
        for ser in train_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            train_series_transformed_new.append(ser_new)                       
        # For pred step2 result
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        # 进行推理及预测，先fit再predict
        model.fit(train_series_transformed_new, future_covariates=future_convariates, val_series=val_series_transformed_new,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)    
        # 先fit再赋值，则以当前数据集生成的mapping为准           
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings          
        pred_result = model.predict(series=val_series_transformed_new,past_covariates=past_convariates,future_covariates=future_convariates,
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
           
    def batch_pred_bidi(self,dataset):   
        """批量推理，并生成整合后结果"""
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
        # 生成tft时间序列数据集,包括目标数据、协变量等
        model_path = self.optargs["model_path"]
        entries = os.listdir(model_path)
        subdirectories = []
        for entry in entries:
            full_path = os.path.join(model_path, entry)
            if os.path.isdir(full_path):
                subdirectories.append(entry)
        
        total_results = []
        for sub_path in subdirectories:
            model_name = "step"
            # 拼接对应的模型目录
            work_dir = os.path.join(model_path,sub_path)
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
            global_var.set_value("load_ass_data",False)
            global_var.set_value("save_ass_data",False)  
            device = self._build_device()
            best_weight = self.optargs["best_weight"]    
            # 获取模型父路径，并遍历分别获得每个推理需要使用的模型
            model = FuturesModel.load_from_checkpoint(model_name,work_dir=work_dir,device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
            self.rebuild_model_params(model,model_name=model_name)  
            model.batch_size = self.batch_size     
            model.mode = "pred_result"
            model.model.mode = "pred_result"             
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model_real,train_loader,val_loader = model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)
            model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            trainer.validate(model=model_real,dataloaders=val_loader)    
            # 推理后保留当前子模型下一个月份的数据，作为实际结果数据
            mode_month_date = datetime.datetime.strptime(str(sub_path), "%Y%m")  
            next_month = get_next_month(mode_month_date).strftime("%Y%m")
            result_view_file_path = model.model.result_view_file_path
            coll_result = pd.read_csv(result_view_file_path)
            coll_result = coll_result[coll_result['date'].astype(str).str[:6]==next_month]
            total_results.append(coll_result)
        total_results = pd.concat(total_results)
        total_results = total_results[['date','pred_trend', 'top_index', 'instrument', 'diff_range']] 
        save_path = os.path.join(self.optargs["work_dir"],"total_coll_results.csv")
        total_results.to_csv(save_path,index=False)
        
                         