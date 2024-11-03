# 使用darts架构的TFT模型，定制化numpy数据集模式

from __future__ import division
from __future__ import print_function

from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional, Sequence, Tuple, Union
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

from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import compute_price_class,compute_price_class_batch
from darts_pro.data_extension.custom_dataset import CustomSequentialDataset
from darts_pro.data_extension.batch_dataset import BatchDataset,BatchOutputDataset
from darts_pro.data_extension.clustering_dataset import ClustringBatchOutputDataset,VareBatchOutputDataset
from darts_pro.data_extension.series_data_utils import StatDataAssis
from darts_pro.tft_series_dataset import TFTSeriesDataset
from darts_pro.data_extension.custom_nor_model import TFTAsisModel,TFTBatchModel
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC,SLOPE_SHAPE_SMOOTH,CLASS_LAST_VALUE_MAX

import cus_utils.global_var as global_var

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

        global_var.set_value("dataset", dataset)
        viz_result_suc = TensorViz(env="train_result_suc")
        viz_result_fail = TensorViz(env="train_result_fail")
        viz_target = TensorViz(env="viz_target")
        global_var.set_value("viz_target",viz_target)
        global_var.set_value("viz_result_suc",viz_result_suc)
        global_var.set_value("viz_result_fail",viz_result_fail)
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)    
                    
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
            with open(df_valid_path, "rb") as fin:
                dataset.df_val = pickle.load(fin)      
            dataset.df_all = pd.concat([dataset.df_train,dataset.df_val])   
              
            global_var.set_value("ass_data_train",ass_data_train)
            global_var.set_value("ass_data_valid",ass_data_valid)
            global_var.set_value("load_ass_data",True)
                                
        global_var.set_value("dataset", dataset)  
        if self.type.startswith("data_pca"):
            self.data_pca(dataset)
        if self.type.startswith("data_lstm"):
            self.data_lstm(dataset)
        if self.type.startswith("data_corr"):
            self.data_corr(dataset)
        if self.type.startswith("data_view"):
            self.data_view(dataset)            
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
        
    def data_corr(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行相关性分析"""
        
        batch_file_path = self.kwargs["batch_file_path"] 
            
        df_data_path = os.path.join(batch_file_path,"main_data.pkl")
        ass_train_path = os.path.join(batch_file_path,"ass_data_train.pkl")
        ass_valid_path = os.path.join(batch_file_path,"ass_data_valid.pkl")        
        # 加载主要序列数据和辅助数据
        with open(df_data_path, "rb") as fin:
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
                pickle.load(fin)   
        with open(ass_train_path, "rb") as fin:
            ass_data_train = pickle.load(fin)  
        with open(ass_valid_path, "rb") as fin:
            ass_data_valid = pickle.load(fin) 
        global_var.set_value("ass_data_train",ass_data_train)
        global_var.set_value("ass_data_valid",ass_data_valid)
        global_var.set_value("load_ass_data",True)

        output_chunk_length = self.optargs["forecast_horizon"]
        input_chunk_length = self.optargs["wave_period"] - output_chunk_length
        past_split = self.optargs["past_split"] 
        
        custom_dataset_valid = CustomSequentialDataset(
                    target_series=val_series_transformed,
                    past_covariates=past_convariates,
                    future_covariates=future_convariates,
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=output_chunk_length,
                    max_samples_per_ts=None,
                    use_static_covariates=True,
                    mode="valid"
                )            
        data_assis = StatDataAssis()
        col_list = dataset.col_def["col_list"] + ["label"]
        analysis_columns = ["label_ori","REV5","IMAX5","QTLUMA5","OBV5","CCI5","KMID","KLEN","KMID2","KUP","KUP2",
                            "KLOW","KLOW2","KSFT","KSFT2", 'STD5','QTLU5','CORD5','CNTD5','VSTD5','QTLUMA5','BETA5',
            'KURT5','SKEW5','CNTP5','CNTN5','SUMP5','CORR5','SUMPMA5','RANK5','RANKMA5']
        analysis_columns = ["price","QTLUMA5","CNTN5","SUMPMA5"]
        analysis_columns = ["price","QTLUMA5",'CCI5','ATR5','RVI','AOS']
        # 利用dataloader进行数据拼装
        val_loader = DataLoader(
                custom_dataset_valid,
                batch_size=1024,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_filter,
            )
        past_conv_index = past_split[0]
        past_columns = dataset.get_past_columns()
        past_columns = past_columns[past_conv_index[0]:past_conv_index[1]]       
        combine_columns = ["price"] + past_columns + dataset.get_target_column() 
        analysis_data = None
        print("total len：",len(val_loader))
        # 遍历数据集，并按照批次进行计算，汇总后取得平均值
        for index,batch_data in enumerate(val_loader):
            (past_target,past_covariates, historic_future_covariates,future_covariates,
                    static_covariates,scaler_future_past_covariate,target_class,target_info,price_target,future_target) = batch_data
            target_class = target_class[:,0,0]
            # if index>5:
            #     break
            index_filter = []
            # 筛选指定日期数据
            for i,ti in enumerate(target_info):
                future_start_datetime = ti["future_start_datetime"]
                if future_start_datetime<20220401 and future_start_datetime>=20220301 or True:
                    index_filter.append(i)    
            
            # 计算价格差的时候，把前一日期也包括进来
            price_array = np.array([ts["price_array"][-6:] for ts in target_info])  
            price_range = ((price_array[:,1:] - price_array[:,:-1])/price_array[:,:-1])*10
            price_range = np.expand_dims(price_range,-1)
            # 对价格归一化后进行比较
            price_array_scale = MinMaxScaler().fit_transform(price_array[:,1:].transpose(1,0)).transpose(1,0)
            price_array_scale = np.expand_dims(price_array_scale,-1)
            future_past_covariate = np.array([item[1] for item in scaler_future_past_covariate])
            future_past_covariate_item = future_past_covariate[...,past_conv_index[0]:past_conv_index[1]]  
            analysis_batch = np.concatenate([price_array_scale,future_past_covariate_item,future_target],-1)   
            analysis_batch = analysis_batch[index_filter]
            price_range = price_range[index_filter]
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
        
    def _batch_collate_filter(self,ori_batch):
        """
        重载方法，调整数据处理模式
        """
        
        batch = ori_batch
        aggregated = []
        first_sample = ori_batch[0]
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
        return tuple(aggregated)
    
    def data_corr_old(
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
        # data_assis.est_thredhold(train_ds)
                
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
                
     
    def data_view(self,dataset):
        from trader.data_viewer import DataViewer
        
        df_ref = dataset.df_all
        viz_target = global_var.get_value("viz_target")
        
        viewer = DataViewer(env_name="data_price")
        instrument = 603711
        instrument = 603817
        instrument = 399006
        date_range = (20220301,20220315)
        att_cols = ["QTLUMA5"]
        viewer.market_price(df_ref, date_range, instrument, dataset,viz_target=viz_target,att_cols=att_cols)    
        save_path = "custom/data/viz"   
        viewer.market_price_mpl(df_ref, date_range, instrument, dataset,save_path=save_path)   
        
        
        