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
from torchmetrics import (
    PearsonCorrCoef,
    MetricCollection,
)
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression
from custom_model.simple_model import Trainer
from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from cus_utils.metrics import corr_dis,series_target_scale,diff_dis
from losses.mtl_loss import CorrLoss,UncertaintyLoss
from darts_pro.data_extension.custom_model import TFTCusModel,TFTExtModel
from darts_pro.tft_series_dataset import TFTSeriesDataset
from .base_process import BaseNumpyModel
from numba.core.types import none
from gunicorn import instrument

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from cus_utils.db_accessor import DbAccessor
from threading import _enumerate

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
        
    def fit(
        self,
        dataset: TFTSeriesDataset
    ):
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
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
        if self.type.startswith("pred_data_view"):
            print("no need fit for pred_data_view")
            self.pred_data_view(dataset)
            return         
        """对预测数据进行分类训练"""
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        train_series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.build_series_data()
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
            
  
        
        self.model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 trainer=None,epochs=self.n_epochs,verbose=True)
    
    def _build_model(self,dataset,emb_size=1000,use_model_name=True):
        """生成模型"""
        
        # 使用多任务下的不确定损失作为损失函数
        device = "cuda:" + str(self.gpus)
        
        optimizer_cls = torch.optim.Adam
        optimizer_kwargs={"lr": 1e-2,"weight_decay":1e-4}
        # 使用余弦退火的学习率方式
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
        # metric_collection = MetricCollection(
        #     # [PearsonCorrCoef()]
        # )               
        categorical_embedding_sizes = {"dayofweek": 5,dataset.get_group_rank_column(): emb_size}
        # categorical_embedding_sizes = None    
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
            categorical_embedding_sizes=categorical_embedding_sizes,
            # likelihood=QuantileRegression(
            #     quantiles=quantiles
            # ), 
            likelihood=None,
            loss_fn=torch.nn.MSELoss(),
            use_weighted_loss_func=True,
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
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0],"log_every_n_steps":50}  
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
 
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        model_name = self.optargs["model_name"]
        forecast_horizon = self.optargs["forecast_horizon"]
        
        # 缓存全量数据
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all_pred.pkl"
            series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.build_series_data(df_data_path)    
        else:
            series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.build_series_data()
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all_pred.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)  
                    
        self.series_data_view(dataset,series_transformed,past_convariates=past_convariates,
                              future_convariates=future_convariates,title="pred_target")
        
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
        # 需要进行fit设置
        my_model.fit(series_transformed,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,verbose=True,epochs=-1)            
        logger.info("begin predict")   
        # 对验证集进行预测，得到预测结果   
        pred_series_list,scaler_map = my_model.predict(n=dataset.pred_len, series=series_transformed,num_samples=10,
                                            past_covariates=past_convariates,future_covariates=future_convariates)
        # 保存结果到数据库
        self.save_pred_result(pred_series_list,val_series_transformed,dataset=dataset,update=False)     
        logger.info("do predict_show")  
        # 可视化
        self.predict_show(val_series_transformed,pred_series_list, series_transformed,dataset=dataset,do_scale=False,scaler_map=scaler_map)
        self.model = my_model
        return pred_series_list,val_series_transformed

    def build_pred_result(self, dataset: TFTSeriesDataset):
        """针对连续天，逐个生成对应的预测数据"""
        
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
            dataset.build_series_data(df_data_path)    
            return
        
        dataset.build_series_data()
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
            dataset.build_series_data(df_data_path)    
            self.df_ref = dataset.df_all
            # self.df_ref = self.df_ref[self.df_ref["instrument"].isin([600033,600035,600036])]
            return
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        train_series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.build_series_data()
        if self.save_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            with open(df_data_path, "wb") as fout:
                pickle.dump(dataset.df_all, fout)           
                
        self.series_data_view(dataset,train_series_transformed,past_convariates=past_convariates,title="train_target")
        self.series_data_view(dataset,val_series_transformed,past_convariates=None,title="val_target")

    def pred_data_view(self, dataset: TFTSeriesDataset):
        """对预测数据进行归纳查看"""
        
        # from tsai.all import *
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
        self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=True)
                    
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path)    
        
        # 生成分类训练数据
        pred_data_path = self.kwargs["pred_data_path"]
        complex_df_train = dataset.combine_complex_df_data(dataset.df_all, dataset.train_range, 
                                                  pred_data_path=pred_data_path, load_cache=True,type="train")
        complex_df_valid = dataset.combine_complex_df_data(dataset.df_all, dataset.valid_range, 
                                                  pred_data_path=pred_data_path, load_cache=True,type="valid")    
        complex_df = pd.concat([complex_df_train,complex_df_valid])
        self.view_complex_data(complex_df,type="total",dataset=dataset)  
        
    def classify_train(self, dataset: TFTSeriesDataset):
        """对预测数据进行分类训练"""
        
        # from tsai.all import *
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]
        
        self.model = TFTExtModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],best=True)
                    
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path)    
        else:
            series_transformed,val_series_transformed,past_convariates,future_convariates = dataset.build_series_data()
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)       
                      
        # 生成分类训练数据
        complex_df_train = dataset.combine_complex_df_data(dataset.df_all, dataset.train_range, 
                                                  pred_data_path=self.pred_data_path, load_cache=False,type="train")
        complex_df_valid = dataset.combine_complex_df_data(dataset.df_all, dataset.valid_range, 
                                                  pred_data_path=self.pred_data_path, load_cache=False,type="valid")    
        complex_df = pd.concat([complex_df_train,complex_df_valid])
        self.view_complex_data(complex_df,type="total",dataset=dataset)  
        train_data,ref_train_data_array = dataset.combine_complex_data(dataset.df_all, dataset.train_range, 
                                                  pred_data_path=self.pred_data_path, load_cache=True,type="train")
        valid_data,ref_valide_data_array = dataset.combine_complex_data(dataset.df_all, dataset.valid_range, 
                                                  pred_data_path=self.pred_data_path, load_cache=True,type="valid")
        # 执行标准化
        scaler = StandardScaler()
        train_data_X = train_data[:,:-1] 
        train_data_X = scaler.fit_transform(train_data[:,:-1])
        # train_data_X = dataset.transfer_pred_data(train_data_X)
        train_data_X = np.expand_dims(train_data_X,axis=0).transpose(1,0,2)
        valid_data_X = valid_data[:,:-1] 
        valid_data_X = scaler.transform(valid_data[:,:-1])
        # valid_data_X = dataset.transfer_pred_data(valid_data_X)
        valid_data_X = np.expand_dims(valid_data_X,axis=0).transpose(1,0,2)
        # 统一成一个数据集以及分割数组
        total_X = np.concatenate((train_data_X,valid_data_X),axis=0)
        total_Y = np.concatenate((train_data[:,-1],valid_data[:,-1]),axis=0)
        # self.view_classify_data(train_data_X,train_data[:,-1],ref_train_data_array,type="train")
        self.stat_classify_data(train_data_X,train_data[:,-1],ref_train_data_array,type="train")
        train_size = train_data_X.shape[0]
        valid_size = valid_data_X.shape[0]
        splits_train = [i for i in range(train_size)]
        splits_valid = [train_size+i for i in range(valid_size)]
        # total_X,total_Y,(splits_train,splits_valid) = self.build_fake_data(dataset)
        
        # 使用tsai模型进行分类训练和测试
        n_epochs = 200
        # batch_tfms = TSStandardize(by_sample=True)
        # mv_clf = TSClassifier(total_X, total_Y, splits=(splits_train,splits_valid), path='models', arch=InceptionTimePlus, batch_tfms=batch_tfms, metrics=accuracy, cbs=None)
        # mv_clf.fit_one_cycle(n_epochs, 1e-1)
        # mv_clf.export("mv_clf.pkl")
        trainer = Trainer(n_input=5,n_hidden=20,n_output=3,batch_size=64,n_epochs=n_epochs)
        total_Y = total_Y - 1
        trainer.build_dataset(total_X.reshape(total_X.shape[0],total_X.shape[2]),total_Y,(splits_train,splits_valid))
        trainer.train()
        
    def save_pred_result(self,pred_series_list,val_series_list,dataset=None,update=False):
        """保存预测记录"""
        
        dbaccessor = DbAccessor({})
        if not update:
            # 记录主批次信息
            sql = "insert into pred_result(batch_no) values(%s)"
            dbaccessor.do_inserto_withparams(sql, (1)) 
        result_id = dbaccessor.do_query("select max(id) from pred_result")[0][0]
        dbaccessor.do_inserto_withparams("delete from pred_result_detail where result_id=%s", (result_id))
        group_rank_column = dataset.get_group_rank_column()
        for i in range(len(val_series_list)):
            pred_series = pred_series_list[i]
            val_series = val_series_list[i]           
            group_rank_code = pred_series.static_covariates[group_rank_column].values[0]
            group_code = str(dataset.get_group_code_by_rank(group_rank_code))
            # 分别为每个股票序列记录得分数据
            mape_item = mape(val_series, pred_series)
            # 计算相关度
            corr_item = corr_dis(val_series, pred_series) 
            sql = "insert into pred_result_detail(result_id,instrument_rank,instrument,mape,corr) values(%s,%s,%s,%s,%s)"
            params = (result_id, group_rank_code,group_code, mape_item,corr_item)
            dbaccessor.do_inserto_withparams(sql, params)            
        
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
    
    def stat_classify_data(self,X_data,Y_data,ref_data_array,type="train"):
        viz_input = TensorViz(env="data_pred")
        pred_len = 5
        X_data = X_data.reshape(X_data.shape[0],X_data.shape[2])   
        X_mean = np.expand_dims(np.mean(X_data,axis=-1),axis=1)
        X_data = np.concatenate((X_data,X_mean),axis=-1)
        Y_data = np.expand_dims(Y_data,axis=-1)
        combine_data = np.concatenate((X_data,Y_data),axis=1)
        columns_pred = ["pred_{}".format(i) for i in range(pred_len)]
        columns = columns_pred + ["t_value","label"]
        df = pd.DataFrame(combine_data,columns=columns)
        target_df = df[df["label"]==3]
        view_data = target_df[columns_pred].values.transpose(1,0)
        target_title = "classify_stat_{}".format(3)
        names = ["value_{}".format(i) for i in target_df[["t_value"]].values]
        view_data = view_data[:,:20]
        names = names[:20]
        viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,names=names)              
        # target_df = df
        # for i in range(pred_len):
        #     target_df = target_df[target_df["pred_{}".format(i)]>0]
        # print(target_df)   
        # target_df = target_df[target_df["t_value"]>1]
        # print(target_df)

    def view_complex_data(self,complex_data,type="train",dataset=None):
        viz_input = TensorViz(env="data_complex")
        data_columns = ["instrument","date"]
        # 预测数据
        pred_columns = ["pred_{}".format(i) for i in range(dataset.pred_len)]
        # 标签数据）
        label_columns = ["label_{}".format(i) for i in range(dataset.pred_len)]
        # 实际价格（滑动窗之前的原始数据）
        price_columns = ["price_{}".format(i) for i in range(dataset.pred_len*2)]
        data_columns = data_columns + pred_columns + label_columns + price_columns
        start_time = "20210201"   
        end_time = "20210331" 
        df_range = complex_data[(complex_data["date"]>=pd.to_datetime(start_time)) & (complex_data["date"]<pd.to_datetime(end_time))]
        date_list = df_range["date"].dt.strftime('%Y%m%d').unique()   
        pred_threhold = 1.8
        for date in date_list:     
            match_cnt = 0
            target_data = complex_data[(complex_data["date"]==date)]
            pad_items = np.array([0.0 for i in range(dataset.pred_len)])
            names = ["pred","label","price"]
            for idx,row in target_data.iterrows():
                # rise_cnt = 0
                # total_price = 0
                # # 总上涨数量
                # for i in range(dataset.pred_len):
                #     total_price += row["price_{}".format(i+dataset.pred_len)]
                #     if row["pred_{}".format(i)]>0:
                #         rise_cnt += 1
                # # 最后两个数据不能连续下降
                # end_match_flag = ((row["pred_{}".format(dataset.pred_len-1)]>row["pred_{}".format(dataset.pred_len-2)]) or 
                #     (row["pred_{}".format(dataset.pred_len-2)]>row["pred_{}".format(dataset.pred_len-3)]))                   
                # if row["pred_4"]<pred_threhold or rise_cnt<4 or not end_match_flag:
                #     continue
                if idx<20 or idx>30:
                    continue
                total_price = row["price_{}".format(2*dataset.pred_len-1)] - row["price_0"]
                total_price = round(total_price, 2)
                target_title = "{}_{}_{}".format(int(row["instrument"]),row["date"].strftime("%Y%m%d"),total_price)
                pred_line = row[pred_columns].values
                # pred_line = np.concatenate((pad_items,pred_line),axis=0)
                label_line = row[label_columns].values
                # label_line = np.concatenate((pad_items,label_line),axis=0)
                price_line = row[price_columns].values
                view_data = np.stack((pred_line,label_line),axis=0).transpose(1,0)
                viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,names=names)  
                match_cnt += 1
                # print("price mean:{} and label:{} and pred:{}".format(np.mean(price_line[5:]),label_line[-1],pred_line[-1]))
            print("date:{} and match_cnt:{}".format(date,match_cnt))
            
    def view_classify_data(self,X_data,Y_data,ref_data_array,type="train"):
        viz_input = TensorViz(env="data_pred")
        for i in range(10):
            view_data = X_data[i][0]
            view_data = np.stack((view_data,np.array(ref_data_array[i])),axis=0).transpose(1,0)
            target_title = "classify_data_{}_{}".format(type,i)
            names = ["line_{},class:{}".format(i,int(Y_data[i]))] + ["ref data"]
            viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,names=names)  

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

               
    def predict_show(self,val_series_list,pred_series_list,series_train,dataset=None,do_scale=False,scaler_map=None):       
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"       
        forecast_horizon = self.optargs["forecast_horizon"]    
        # 整个序列比较多，只比较某几个序列
        figsize = (9, 6)
        group_rank_column = dataset.get_group_rank_column()
        group_column = dataset.get_group_column()
        # 创建比较序列，后面保持所有，或者自己指定长度
        actual_series_list = []
        for index,ser_val in enumerate(val_series_list):
            # 根据标志，对实际数据进行缩放，与同样保持缩放的预测数据进行比较
            if do_scale:
                group_rank_code = ser_val.static_covariates[group_rank_column].values[0]
                scaler = scaler_map[group_rank_code]
                ser_val = series_target_scale(ser_val,scaler=scaler)            
            ser_train = series_train[index]
            ser_total = ser_train.concatenate(ser_val)
            # 从数据集后面截取一定长度的数据，作为比较序列
            actual_series = ser_total[
                ser_total.end_time() - (3 * forecast_horizon - 1) * ser_total.freq : 
            ]
            actual_series_list.append(ser_total)   
        mape_all = 0
        corr_all = 0
        diff_all = 0
        # r = 5
        # view_list = random_int_list(1,len(val_series_list)-1,r)
        
        # 根据参数，只显示指定股票图形
        instrument_pick = dataset.kwargs["instrument_pick"]
        if len(instrument_pick)==0:
            df_pick = dataset.df_all
        else:
            df_pick = dataset.get_data_by_group_code(instrument_pick)
        for i in range(len(val_series_list)):
            pred_series = pred_series_list[i]
            val_series = val_series_list[i]
            actual_series = actual_series_list[i]
            group_rank_code = pred_series.static_covariates[group_rank_column].values[0]
            # 与实际的数据集进行比较，比较的是两个数据集的交集
            mape_item = mape(val_series, pred_series)
            mape_all = mape_all + mape_item
            # 计算相关度
            corr_item = corr_dis(val_series, pred_series)  
            corr_all = corr_all + corr_item   
            # 开始结束差的距离衡量
            diff_item = diff_dis(val_series, pred_series) 
            diff_all = diff_all + diff_item      
            # 取得指定股票，如果不存在则不进行可视化
            df_item = df_pick[df_pick[group_rank_column]==group_rank_code]
            if df_item.shape[0]>0:
                # 分位数范围显示
                pred_series.plot(
                    low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
                )
                pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)                
                plt.figure(figsize=figsize)
                # 实际数据集的结尾与预测序列对齐
                pred_series.plot(label="forecast")            
                instrument_code = df_item[group_column].values[0]
                actual_series[pred_series.end_time()- 25 : pred_series.end_time()+1].plot(label="actual")           
                plt.title("ser_{},MAPE: {:.2f}%,corr:{}".format(instrument_code,mape_item,corr_item))
                plt.legend()
                plt.savefig('{}/result_view/eval_{}.jpg'.format(self.optargs["work_dir"],instrument_code))
                plt.clf()   
        mape_mean = mape_all/len(val_series_list)
        corr_mean = corr_all/len(val_series_list)
        diff_mean = diff_all/len(val_series_list)
        print("mape_mean:{},corr mean:{},diff mean:{}".format(mape_mean,corr_mean,diff_mean))           
        
        