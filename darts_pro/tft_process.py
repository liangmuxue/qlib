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

from cus_utils.common_compute import compute_price_class,compute_price_class_batch
from darts_pro.data_extension.batch_dataset import BatchDataset,BatchOutputDataset
from darts_pro.data_extension.clustering_dataset import ClustringBatchOutputDataset,VareBatchOutputDataset
from darts_pro.data_extension.series_data_utils import StatDataAssis
from darts_pro.tft_series_dataset import TFTSeriesDataset
from darts_pro.data_extension.custom_nor_model import TFTAsisModel,TFTBatchModel
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC,SLOPE_SHAPE_SMOOTH,CLASS_LAST_VALUE_MAX

import cus_utils.global_var as global_var
from darts_pro.data_extension.custom_tcn_model import ClassifierTrainer

class TftDatafAnalysis():
    
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
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]     
                
        if self.load_dataset_file:
            df_data_path = self.pred_data_path + "/df_all.pkl"
            dataset.build_series_data(df_data_path,no_series_data=True)  
        else:         
            dataset.build_series_data(no_series_data=True)
            if self.save_dataset_file:
                df_data_path = self.pred_data_path + "/df_all.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(dataset.df_all, fout)  
                                
        global_var.set_value("dataset", dataset)  
        if self.type.startswith("data_pca"):
            self.data_pca(dataset)
        if self.type.startswith("data_lstm"):
            self.data_lstm(dataset)
        if self.type.startswith("data_corr"):
            self.data_corr(dataset)
        if self.type.startswith("output_corr"):
            self.output_corr(dataset)               
        if self.type.startswith("data_linear_reg"):
            self.data_linear_reg(dataset)            
        if self.type.startswith("batch_data_ana"):
            self.batch_data_ana(dataset) 
        if self.type.startswith("batch_ot_data_ana"):
            self.batch_ot_data_ana(dataset)         
        if self.type.startswith("clustering_output"):
            self.clustering_output(dataset)                  
               
    def data_pca(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行主成分分析"""
         
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_batch.pickel".format(batch_file_path)
        col_list = dataset.col_def["col_list"]
        col_list.remove("label_ori")
        col_list.remove("REV5_ORI")
        col_list = ["CCI5"]
        ds = BatchDataset(batch_file,fit_names=col_list)
        ret_file = "{}/pca_ret_cci.npy".format(batch_file_path)
        ds.analysis_df_pca(fit_names=col_list,range_num=3000,ret_file=ret_file)     
        
    def data_lstm(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行主成分分析"""
         
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_part_batch.pickel".format(batch_file_path)
        col_list = dataset.col_def["col_list"]
        col_list.remove("label_ori")
        col_list.remove("REV5_ORI")
        col_list = ["CCI5"]
        train_ds = BatchDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[0,10000])
        valid_ds = BatchDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[10000,12000])
        trainer = ClassifierTrainer(train_ds,valid_ds,input_dim=len(col_list))
        trainer.training()
        
    def data_corr(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行相关性分析"""
        
        data_assis = StatDataAssis()
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_batch.pickel".format(batch_file_path)   
        batch_file = "{}/valid_output_batch.pickel".format(batch_file_path)   
        col_list = dataset.col_def["col_list"] + ["label"]
        analysis_columns = ["OBV5","CLOSE","MASCOPE5","OBV5","RSI5","MACD",'KDJ_K','KDJ_D','KDJ_J','CCI5','RESI5']
        analysis_columns = ["label_ori","REV5","IMAX5","QTLUMA5","OBV5","CCI5","KMID","KLEN","KMID2","KUP","KUP2",
                            "KLOW","KLOW2","KSFT","KSFT2", 'STD5','QTLU5','CORD5','CNTD5','VSTD5','QTLUMA5','BETA5',
            'KURT5','SKEW5','CNTP5','CNTN5','SUMP5','CORR5','SUMPMA5','RANK5','RANKMA5']
        # analysis_columns = ['label_ori','label_ori','MOMENTUM','RVI','OBV5','RSI5','CCI5','MA5','label','REV5','REV5_ORI','MACD','TURNOVER_CLOSE','VOLUME_CLOSE','MASCOPE5',
        #       'PRICE_SCOPE','RESI5','CLOSE','OPEN','HIGH', 'LOW','RSI10','KDJ_K','KDJ_D','KDJ_J','IMAX5',"ATR5","AOS",'WVMA5','HIGH_QTLU5','RSQR5',
        #       "KMID","KLEN","KMID2","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2", 'STD5','QTLU5','CORD5','CNTD5','VSTD5','QTLUMA5','BETA5',
        #       'KURT5','SKEW5','CNTP5','CNTN5','SUMP5','CORR5','SUMPMA5','RANK5','RANKMA5'] 
        # col_list.remove("label_ori")
        # col_list.remove("REV5_ORI")
        # train_ds = BatchDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[0,10000])
        train_ds = BatchOutputDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[0,10000])
        data_assis.data_corr_analysis(train_ds,analysis_columns=analysis_columns)

    def output_corr(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对output数据进行相关性分析"""
        
        data_assis = StatDataAssis()
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/valid_output_batch.pickel".format(batch_file_path)   
        # batch_file = "{}/train_output_batch.pickel".format(batch_file_path)  
        col_list = dataset.col_def["col_list"] + ["label"]
        # 重点比较的输出
        target_col = ['MACD']
        # 分析的列，第一个是实际目标列，后面的是输出列
        analysis_columns = ["MACD","MACD_output","RANKMA5_output","QTLU_output"]
        fit_names = ["MACD","RANKMA5","QTLU"]
        diff_columns = ["QTLU_output"]
        range_num = None
        # range_num = [0,1000]
        train_ds = BatchOutputDataset(batch_file,target_col=target_col,fit_names=fit_names,mode="analysis_output",range_num=range_num)
        data_assis.output_corr_analysis(train_ds,analysis_columns=analysis_columns,fit_names=fit_names,target_col=target_col,diff_columns=diff_columns)
        # data_assis.output_target_viz(train_ds,fit_names=fit_names)
        
    def data_linear_reg(
        self,
        dataset: TFTSeriesDataset,
    ):
        """线性回归任务"""
        
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_batch.pickel".format(batch_file_path)
        col_list = ['MASCOPE5','OBV5','RSI5']
        col_list = ['MASCOPE5','RSI5']
        target_col = ['PRICE_SCOPE']

        base_size = 10000
        mode = "analysis_reg_ota"
        # mode = "analysis_reg"
        if mode=="analysis_reg_ota":
            input_index = [1,4]
            input_dim = input_index[1] - input_index[0] 
            fit_names = input_index
            file_name = "reg_conv.pth"
        if mode=="analysis_reg":
            fit_names = col_list
            input_dim = len(col_list)
            file_name = "reg.pth"
        range_num_train = [0,base_size]
        range_num_valid = [base_size,int(base_size*1.2)]
        train_ds = BatchDataset(batch_file,target_col=target_col,fit_names=fit_names,mode=mode,range_num=range_num_train)
        valid_ds = BatchDataset(batch_file,target_col=target_col,fit_names=fit_names,mode=mode,range_num=range_num_valid)
        trainer = ClassifierTrainer(train_ds,valid_ds,input_dim=input_dim)
        trainer.reg_training(load_model=False,file_name=file_name)
               
    def batch_data_ana(
        self,
        dataset: TFTSeriesDataset,
    ):
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        
        type = "train"
        type = "valid"
        
        ds = BatchDataset(
            filepath = "{}/{}_batch.pickel".format(self.batch_file_path,type),mode="process"
        )   
        data_assis = StatDataAssis()
        data_assis.batch_data_ana(ds)
        
    def batch_ot_data_ana(
        self,
        dataset: TFTSeriesDataset,
    ):
        
        batch_file = "{}/valid_output_batch.pickel".format(self.kwargs["batch_file_path"])  
        ds = BatchOutputDataset(
            filepath = batch_file,mode="process"
        )   
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info)  = ds.target_data
        price_array = np.array([ts["price_array"] for ts in target_info])
        output = ds.output_data
        
        raise_bool = ((output[:,-1,:] - output[:,0,:])/np.abs(output[:,0,:])*100)>3
        raise_index = [np.where(raise_bool[:,i])[0] for i in range(3)]
        
        price_target_array = price_array[:,25:]
        p_target_class = compute_price_class_batch(price_target_array,mode="first_last")[0]
        import_price_index = np.where(p_target_class==CLASS_SIMPLE_VALUE_MAX)[0]
        
        for i in range(3):
            match_index = np.intersect1d(raise_index[i],import_price_index)
            print("match cnt:{}".format(match_index.shape[0]))
        
        print("raise cnt:",[ri.shape[0] for ri in raise_index])
        print("import_price_index cnt:",import_price_index.shape[0])
        
    def clustering_output(
        self,
        dataset: TFTSeriesDataset,
    ):
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/valid_output_batch.pickel".format(batch_file_path)   
        
        # ds = VareBatchOutputDataset(batch_file,mode="process") 
        ds = ClustringBatchOutputDataset(batch_file,mode="process") 
        data_assis = StatDataAssis()
        data_assis.analysis_compare_output(ds)
                
        