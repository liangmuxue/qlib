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
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR,MultiStepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression
from custom_model.simple_model import Trainer
from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from cus_utils.metrics import corr_dis,series_target_scale,diff_dis,cel_acc_compute,vr_acc_compute
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC
from darts_pro.data_extension.custom_model import TFTExtModel
from darts_pro.tft_series_dataset import TFTSeriesDataset

import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from cus_utils.log_util import AppLogger
logger = AppLogger()

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
        self.kwargs = kwargs
        
        global_var._init()
        
    def fit(
        self,
        dataset: TFTSeriesDataset,
    ):
        global_var.set_value("dataset", dataset)
        
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
            self.predict(dataset)
            return           
        if self.type.startswith("build_pred_result"):
            self.build_pred_result(dataset)
            return            
        if self.type.startswith("backtest"):
            # 直接进行预测,只需要加载模型参数
            print("no need fit for backtest")
            return   
        if self.type.startswith("classify_train"):
            # 直接进行预测,只需要加载模型参数
            print("no need fit for classify_train")
            self.classify_train(dataset)
            return  
             
        """对预测数据进行分类训练"""

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
          
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data(df_data_path)   
        else:
            # 生成tft时间序列数据集,包括目标数据、协变量等
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)    
        
        self.series_data_view(dataset,train_series_transformed,past_convariates=past_convariates,title="train_target")
        self.series_data_view(dataset,val_series_transformed,past_convariates=None,title="val_target")
        
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        load_weight = self.optargs["load_weight"]
        if "monitor" in self.optargs:
            monitor = dataset
        else:
            monitor = None
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
            self.model.mode = "train"
            self.model.model.monitor = monitor
        else:
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
            self.model.monitor = monitor
        
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True)
        
    def _build_model(self,dataset,emb_size=1000,use_model_name=True):
        """生成模型"""
        
        log_every_n_steps = self.kwargs["log_every_n_steps"]
        optimizer_cls = torch.optim.Adam
        scheduler = StepLR
        scheduler_config = self.kwargs["scheduler_config"]
        optimizer_kwargs = self.kwargs["optimizer_kwargs"]
        
        # # 使用余弦退火的学习率方式
        # scheduler = CosineAnnealingLR
        # scheduler_config = {
        #     "T_max": 5, 
        #     "eta_min": 0,
        # }        
        # scheduler = MultiStepLR
        # scheduler = StepLR
        # scheduler_config = {
        #     "gamma": 0.3, 
        #     # "milestones": [1,2,3,4,5,7,9,11,15,18,20],
        #     "step_size": 1
        # }       
        # scheduler_config = {
        #     "gamma": 0.8, 
        #     # "milestones": [10, 20,30,60,70, 80,90,100],
        #     "step_size": 10
        # }            
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
        # metric_collection = MetricCollection(
        #     # [PearsonCorrCoef()]
        # )               
        categorical_embedding_sizes = {"dayofweek": 5,dataset.get_group_rank_column(): emb_size}
        # categorical_embedding_sizes = None    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        model_name = self.optargs["model_name"]
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
            
        my_model = TFTExtModel(
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
            work_dir=self.optargs["work_dir"],
            lr_scheduler_cls=scheduler,
            lr_scheduler_kwargs=scheduler_config,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
            # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
        )
        return my_model          

            
    def predict(self, dataset: TFTSeriesDataset):
                
        if self.type=="classify_train":
            self.classify_train(dataset)
            return           
        if self.type=="backtest":
            self.backtest(dataset)
            return        
        if self.type!="predict":
            return 

        path = "{}/result_view".format(self.optargs["work_dir"])
        if not os.path.exists(path):
            os.mkdir(path)
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        model_name = self.optargs["model_name"]
        forecast_horizon = self.optargs["forecast_horizon"]
        pred_range = dataset.kwargs["segments"]["test"] 
        
        # 缓存全量数据
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all_pred.pkl"
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data(df_data_path)   
        else:
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data(val_ds_filter=True)
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all_pred.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)  
        
        # 根据参数决定是否从文件中加载权重
        load_weight = self.optargs["load_weight"]
        if load_weight:
            my_model = TFTExtModel.load_from_checkpoint(model_name,work_dir=self.optargs["work_dir"],best=True)  
            # my_model.trainer_params["accelerator"] = "cpu"
            # my_model.trainer_params.pop("devices")
            # my_model.batch_size = self.batch_size
        else:
            my_model = self._build_model(dataset,emb_size=1000,use_model_name=True)       
        logger.info("begin fit")     
        # # 需要进行fit设置
        # my_model.fit(series_total,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
        #              val_past_covariates=past_convariates, val_future_covariates=future_convariates,verbose=True,epochs=-1)            
        logger.info("begin predict")   
        my_model.mode = "predict"
        
        # 根据日期范围逐日进行预测，得到预测结果   
        start_time = pred_range[0]
        end_time = pred_range[1]
        date_range = get_tradedays(start_time,end_time)
        for cur_date in date_range:        
            # 根据时间点，取得对应的输入时间序列范围
            total_range,val_range,missing_instruments = dataset.get_part_time_range(cur_date,ref_df=dataset.df_all)
            # 如果不满足预测要求，则返回空
            if total_range is None:
                self.log("pred series none")
                return None
            # 如果包含不符合的数据，再次进行过滤
            if len(missing_instruments)>0:
                outer_df_filter = dataset.df_all[~dataset.df_all[self.dataset.get_group_column()].isin(missing_instruments)]
            else:
                outer_df_filter = dataset.df_all     
            # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
                self.dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=outer_df_filter)
            pred_combine = my_model.predict(n=dataset.pred_len, series=val_series_transformed,num_samples=10,
                                                past_covariates=past_convariates,future_covariates=future_convariates)
            pred_series_list = [item[0] for item in pred_combine]
            pred_class_total = [item[1] for item in pred_combine]       
            vr_class_total = [item[2] for item in pred_combine]   
            logger.info("do predict mesure")  
            pred_class_total = torch.stack(pred_class_total)
            pred_class = dataset.combine_pred_class(pred_class_total)
            vr_class_total = torch.stack(vr_class_total)[:,0,:]
            vr_class = self.combine_vr_class(vr_class_total)
            # 可视化
            result = self.predict_mesure(val_series_transformed,pred_series_list,pred_class[1],vr_class[1],vr_class[0],
                              series_total=series_total,dataset=dataset,do_scale=False,scaler_map=None)
        
        
    
    def combine_vr_class(self,vr_class_total):
        vr_class = F.softmax(vr_class_total,dim=-1)
        vr_class = torch.max(vr_class,dim=-1)
        return vr_class
       
    def build_pred_result(self, dataset: TFTSeriesDataset):
        """针对连续天，逐个生成对应的预测数据"""
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.pred_data_file = self.kwargs["pred_data_file"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
        load_weight = self.optargs["load_weight"]
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            # 使用股票代码数量作为embbding长度
            emb_size = dataset.get_emb_size()            
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
                    
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path,val_ds_filter=True,no_series_data=True)    
            return
        
        dataset.build_series_data(no_series_data=True)
        if self.save_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            with open(df_data_path, "wb") as fout:
                pickle.dump(dataset.df_all, fout)           
        
    def backtest(self, dataset: TFTSeriesDataset):
        """实现回测功能"""
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
        load_weight = self.optargs["load_weight"]
        if load_weight:
            # self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=False)
            self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=False)
            self.model.batch_size = self.batch_size     
        else:
            # 使用股票代码数量作为embbding长度
            emb_size = dataset.get_emb_size()            
            self.model = self._build_model(dataset,emb_size=emb_size,use_model_name=True) 
                    
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path,no_series_data=True)    
            self.df_ref = dataset.df_all
            return
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
        if self.save_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            with open(df_data_path, "wb") as fout:
                pickle.dump(dataset.df_all, fout)           
                
        self.series_data_view(dataset,train_series_transformed,past_convariates=past_convariates,title="train_target")
        self.series_data_view(dataset,val_series_transformed,past_convariates=None,title="val_target")

        
    def classify_train(self, dataset: TFTSeriesDataset):
        """对预测数据进行分类训练"""
        
        # from tsai.all import *
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.pred_data_file = self.kwargs["pred_data_file"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path,no_series_data=True)    
        else:
            series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)       
                      
    
    def filter_series_by_db(self,df_data,dataset=None): 
        """通过之前存储在数据库中的指标,筛选更加合适的序列"""
        
        dbaccessor = DbAccessor({})
        group_column = dataset.get_group_column()
        
        result_rows = dbaccessor.do_query("select instrument from pred_result_detail where result_id=46 and corr>0.5 and cross_metric<2")
        result_rows = np.array(result_rows)[:,0].astype(int)
        target_df = df_data[df_data[group_column].astype(int).isin(result_rows)]
        return target_df
        
    def build_fake_data(self,dataset):
        data = []
        class_num = 3
        cnt = 20000
        for i in range(cnt):
            r = i % class_num
            data_item = [i*10 + r for j in range(5)]
            data.append(data_item)
        label = [i % class_num + 1 for i in range(20)]
        split_train = [x for x in range(12)]
        split_valid = [x+12 for x in range(8)]
        data = np.expand_dims(np.array(data),axis=1).astype(np.float)
        label = np.array(label)
        return data,label,(split_train,split_valid)

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
        label_datas = df_total[target_column].values
        label_range_rate = (label_datas[1:]  - label_datas[:-1])/label_datas[:-1]
        label_range_rate = label_range_rate[(label_range_rate>-0.1)&(label_range_rate<0.1)]
        viz_input.viz_data_hist(label_range_rate,numbins=10,win=title,title=title) 
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
               
    def predict_mesure(self,val_series_list,pred_series_list,pred_class_list,vr_class_list,vr_class_confi_list,
                       series_total=None,dataset=None,do_scale=False,scaler_map=None):       

        group_rank_column = dataset.get_group_rank_column()
        # pred_series_list = dataset.reverse_transform_preds(pred_series_list)
        # series_total = dataset.reverse_transform_preds(series_total)
        forecast_horizon = self.optargs["forecast_horizon"] 
        
        # 创建比较序列，后面保持所有，或者自己指定长度
        actual_series_list = []
        for index,ser_val in enumerate(val_series_list):          
            ser_total = series_total[index]
            # 从数据集后面截取一定长度的数据，作为比较序列
            actual_series = ser_total[
                ser_total.end_time() - (3 * forecast_horizon - 1) * ser_total.freq : 
            ]
            actual_series_list.append(ser_total)   
        mape_all = 0
        corr_all = 0
        diff_all = 0
        vr_acc_all = 0
        vr_imp_all = 0
        vr_imp_recall = 0
        vr_imp_pred_all = 0
        vr_imp_pred_acc = 0
        vr_imp_pred_price_acc = 0
        vr_imp_pred_price_nag = 0
        vr_imp_filter_pred_price_nag = 0
        vr_imp_filter_all = 0
        vr_imp_filter_acc = 0
        vr_imp_pred_result = []
        # r = 5
        # view_list = random_int_list(1,len(val_series_list)-1,r)
        
        # 根据参数，只显示指定股票图形
        instrument_pick = dataset.kwargs["instrument_pick"]
        if len(instrument_pick)==0:
            df_pick = dataset.df_all
        else:
            df_pick = dataset.get_data_by_group_code(instrument_pick)
        result = []
        for i in range(len(pred_series_list)):
            filter_flag = False
            pred_series = pred_series_list[i]
            pred_class = pred_class_list[i]
            vr_class = vr_class_list[i]
            vr_class_confi = vr_class_confi_list[i]
            total_series = series_total[i]
            actual_series = actual_series_list[i]
            group_rank_code = pred_series.static_covariates[group_rank_column].values[0]
            group_code = dataset.get_group_code_by_rank(group_rank_code)
            # 与实际的数据集进行比较，比较的是两个数据集的交集
            mape_item = mape(total_series, pred_series)
            mape_all = mape_all + mape_item
            # 计算相关度
            corr_item = corr_dis(total_series, pred_series)  
            corr_all = corr_all + corr_item   
            # 涨跌幅度类别的准确率
            vr_acc_item = vr_acc_compute(total_series, pred_series,vr_class)  
            vr_acc_all = vr_acc_all + vr_acc_item[0]  
            # 取得指定股票，如果不存在则不进行可视化
            df_item = df_pick[df_pick[group_rank_column]==group_rank_code]            
            # 重点关注上涨类别的召回率和准确率
            if vr_acc_item[1]==CLASS_SIMPLE_VALUE_MAX:
                vr_imp_all += 1
                vr_imp_recall = vr_imp_recall + vr_acc_item[0]  
            if vr_class==CLASS_SIMPLE_VALUE_MAX:
                vr_imp_pred_all += 1
                item_result = {"group_code":group_code,"filter":0,"match":0,"correct":0}
                result.append(item_result)                
                vr_imp_pred_result.append(item_result)
                if vr_acc_item[1]==CLASS_SIMPLE_VALUE_MAX:
                    item_result["match"] = 1                 
                # 通过价格判断是否准确
                start = pred_series.time_index.start
                end = pred_series.time_index.stop
                price_data = df_item[(df_item["time_idx"]>=start-1)&(df_item["time_idx"]<end)]["label_ori"].values
                cur_price = price_data[0]
                price_list = price_data[1:]
                price_raise_range = (price_list.max() - cur_price)/cur_price
                price_down_range = (price_list.min() - cur_price)/cur_price
                if price_raise_range > 0.05:
                    correct = 2
                elif price_raise_range > 0.03:
                    correct = 1
                elif price_raise_range > 0.01:
                    correct = 0         
                elif price_down_range > -0.01:
                    correct = 0        
                elif price_down_range > -0.03:
                    correct = -1
                else:
                    correct = -2                          
                item_result["correct"] = correct                      
                # 进一步筛选后，统计准确率
                filter_flag = self.filter_judge(pred_series, total_series,actual_series,dataset=dataset)
                if filter_flag:
                    vr_imp_filter_all += 1
                    item_result["filter"] = 1
            # 开始结束差的距离衡量
            diff_item = diff_dis(total_series, pred_series) 
            diff_all = diff_all + diff_item      
            # 作图
            if df_item.shape[0]>0:
                # 显示重点上涨类别判断情况
                if vr_class==CLASS_SIMPLE_VALUE_MAX and filter_flag:
                    self.show_pred_result(df_item, pred_series, actual_series, 
                                          vr_class=vr_class,tar_class=vr_acc_item[1],
                                          mape_item=mape_item,corr_item=corr_item,dataset=dataset)
        # 总体统计
        mape_mean = mape_all/len(val_series_list)
        corr_mean = corr_all/len(val_series_list)
        diff_mean = diff_all/len(val_series_list)
        # cross_mean = cross_all/len(val_series_list)
        vr_acc_mean = vr_acc_all/len(val_series_list)   
        vr_recall_imp_mean = vr_imp_recall/vr_imp_all    
        print("mape_mean:{},diff mean:{},cross acc mean:{},vr_acc_mean mean:{},vr_recall_imp_mean:{} with {}/{}".
              format(mape_mean,corr_mean,diff_mean,vr_acc_mean,vr_recall_imp_mean,vr_imp_recall,vr_imp_all)) 
        # 统计重点计算准确度
        for item in result:
            if not item["filter"]==1:
                continue
            vr_imp_filter_all += 1
            if item["match"]==1:
                vr_imp_filter_acc += 1
            if item["correct"]==2:
                vr_imp_pred_price_acc += 1
            if item["correct"]==-2:
                vr_imp_pred_price_nag += 1                
        # vr_acc_imp_sec_mean = vr_imp_sec_pred_acc/vr_imp_sec_pred_all  
        print("vr_imp_filter_acc:{}/{},vr_imp_price_acc:{}/{}/{}".
              format(vr_imp_filter_acc,vr_imp_filter_all,vr_imp_pred_price_acc,vr_imp_pred_price_nag,vr_imp_filter_all))     
        return result
    
    def filter_judge(self,pred_series,total_series,actual_series,raise_range=3,head_range=3,dataset=None):
        # return True
        recent_length = dataset.step_len - dataset.pred_len
        time_range = [pred_series.time_index.start,pred_series.time_index.stop]
        recent_begin_index = time_range[0] - recent_length
        recent_data = actual_series.all_values()[:,0,0][recent_begin_index:time_range[0]]
        
        pred_data = pred_series.all_values()[:,0,:]
        pred_data = np.mean(pred_data,axis=1)
        # 整体需要一定涨幅
        if (pred_data[-1] - pred_data[0])/pred_data[0]<(raise_range/100):
            return False
        # 与近期高点比较，不能差太多
        recent_max= recent_data.max()
        if pred_data[0]<recent_max and (recent_max-pred_data[0])/recent_max>head_range/100:
            return False
        return True
    
    def show_pred_result(self,df_item,pred_series,actual_series,tar_class=None,vr_class=None,mape_item=None,corr_item=None,dataset=None):
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"       
           
        # 整个序列比较多，只比较某几个序列
        figsize = (9, 6)
                
        group_column = dataset.get_group_column()
        group_code = df_item[group_column].values[0].astype(int)
        # 分位数范围显示
        pred_series.plot(
            low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        )
        pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)                
        plt.figure(figsize=figsize)
        # 实际数据集的结尾与预测序列对齐
        pred_series.plot(label="forecast")            
        instrument_code = df_item[group_column].values[0]
        begin_index = pred_series.end_time()- 25
        end_index = pred_series.end_time()+1
        act_series = actual_series[begin_index:end_index]
        act_series.plot(label="actual")   
        # 构造价格序列用于对比
        
        df_price = df_item[(df_item["time_idx"]>=begin_index)&(df_item["time_idx"]<end_index)]
        price_series = TimeSeries.from_dataframe(df_price,
                                                time_col="time_idx",
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 value_cols="label_ori")       
        price_series.plot(label="price")
        # fig, ax = plt.subplots(**kwargs)    
        date_range = dataset.get_datetime_with_index(group_code,pred_series.start_time(),pred_series.end_time())
        range_show = "[{}-{}]".format(date_range[0],date_range[-1])
        # ax.set_xticks(date_range)                                
        pred_class_str = "{}/{}".format(vr_class,tar_class)
        plt.title("{},ser_{},MAPE: {:.2f}%,corr:{},class(pred/real):{}".format(range_show,instrument_code,mape_item,corr_item,pred_class_str))
        plt.legend()
        plt.savefig('{}/result_view/eval_{}.jpg'.format(self.optargs["work_dir"],instrument_code))
        plt.clf()          
           
    def save_pred_result(self,result,dataset=None,update=False):
        """保存预测记录"""
        
        dbaccessor = DbAccessor({})
        if not update:
            # 记录主批次信息
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sql = "insert into pred_result(batch_no,create_time) values(%s,%s)"
            dbaccessor.do_inserto_withparams(sql, (1,dt)) 
        for item in result:
            group_rank_code = item["instrument"]   
            group_code = dataset.get_group_code_by_rank(group_rank_code)
            mape_item = item["instrument"]         
            corr_item = item["corr_item"] 
            cross_item = item["cross_item"] 
            mape_item = item["instrument"]   
            result_id = dbaccessor.do_query("select max(id) from pred_result")[0][0]
            dbaccessor.do_inserto_withparams("delete from pred_result_detail where result_id=%s", (result_id,))
            sql = "insert into pred_result_detail(result_id,instrument_rank,instrument,mape,corr,cross_metric) values(%s,%s,%s,%s,%s,%s)"
            params = (result_id, group_rank_code,group_code, mape_item,corr_item,cross_item)
            dbaccessor.do_inserto_withparams(sql, params)            
        