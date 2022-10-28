from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.log import get_module_logger
from typing import Union, List, Tuple, Dict, Text, Optional
from inspect import getfullargspec
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting.data.encoders import TorchNormalizer,GroupNormalizer
from tft.timeseries_cus import TimeSeriesCusDataset
from tft.timeseries_crf import TimeSeriesCrfDataset
from tft.timeseries_numpy import TimeSeriesNumpyDataset

import bisect
import pandas as pd
import numpy as np

from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_filter import DataFilter

class TFTNumpyDataset(TFTDataset):
    """
    自定义数据集，直接从numpy文件中获取数据，并进行相关处理
    """

    def __init__(self, data_path=None,step_len = 30,pred_len = 5,aug_type="yes",low_threhold=-5,high_threhold=5,over_time=2,col_def={},**kwargs):
        super().__init__(col_def=col_def,step_len=step_len,pred_len=pred_len,**kwargs)
        
        self.future_covariate_col = col_def['future_covariate_col']
        self.past_covariate_col = col_def['past_covariate_col']
        self.static_covariate_col = col_def['static_covariate_col']    
        self.low_threhold = low_threhold
        self.high_threhold = high_threhold
        self.over_time = over_time        
        
        self.columns = self.get_seq_columns()
        self.training_cutoff = 0.7
        
        data = np.load(data_path,allow_pickle=True)
        # 根据条件决定是否筛选数据，取得均衡
        if aug_type=="yes":
            self.data = self.filter_balance_data(data)
        else:
            self.data = data
        # 对目标值进行归一化
        scaler = MinMaxScaler()
        target_index = self.get_target_column_index()
        self.data[:,:,target_index] = scaler.fit_transform(self.data[:,:,target_index])     
        # 对协变量值进行归一化 
        for index in self.get_past_column_index():
            if index==self.get_target_column_index():
                continue
            self.data[:,:,index] = scaler.fit_transform(self.data[:,:,index])   
    
    def filter_balance_data(self,data):
        """筛选出均衡的数据"""
        
        wave_period = self.step_len
        forecast_horizon = self.pred_len
        low_threhold = self.low_threhold
        high_threhold = self.high_threhold
        over_time = self.over_time
                        
        data_filter = DataFilter()
        target_index = self.get_target_column_index()
        low_data = data_filter.get_data_with_threhold(data,target_index,wave_threhold_type="less",threhold=low_threhold,
                                                      wave_period=wave_period,check_length=forecast_horizon,over_time=over_time)
        high_data = data_filter.get_data_with_threhold(data,target_index,wave_threhold_type="more",threhold=high_threhold,
                                                      wave_period=wave_period,check_length=forecast_horizon,over_time=over_time)       
        nor_size = (low_data.shape[0] + high_data.shape[0])//3
        nor_index = np.random.randint(1,data.shape[0],(nor_size,))
        # 参考高低涨幅数据量，取得普通数据量，合并为目标数据
        nor_data = data[nor_index,:,:]
        combine_data = np.concatenate((low_data,high_data,nor_data),axis=0)
        return combine_data
                    
    def get_custom_numpy_dataset(self,mode="train"):
        """
        直接使用numpy数据取得DataSet对象

        Parameters
        ----------
        numpy_data : numpy数据
        """     
        
        future_covariate_index = [self.columns.index(item) for item in self.future_covariate_col]
        past_covariate_index = self.get_past_column_index()
        static_covariate_index = [self.columns.index(item) for item in self.static_covariate_col]
        target_index = self.get_target_column_index()
        
        training_data,val_data = self.training_data_split()
        if mode=="train":
            data = training_data
        else:
            data = val_data
        
        return CustomNumpyDataset(
            data,
            self.step_len-self.pred_len,
            self.pred_len,
            future_covariate_index,
            past_covariate_index,
            static_covariate_index,    
            target_index        
        ) 
        

    def training_data_split(self):  
        """"拆分训练集和测试集"""
        
        time_index = self.get_time_column_index()
        time_begin = self.data[:,:,time_index].min()
        time_end = self.data[:,:,time_index].max()
        # 根据长度取得切分点
        sp_index = time_begin + (time_end - time_begin)*self.training_cutoff
        # 训练集使用时间序列最后一个时间点进行切分
        training_data = self.data[self.data[:,-1,time_index]<sp_index]
        # 测试集使用时间序列第一个时间点进行切分
        val_data = self.data[self.data[:,0,time_index]>sp_index]
        return training_data,val_data
    
        
        
    