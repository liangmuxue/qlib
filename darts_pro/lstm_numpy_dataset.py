from qlib.data.dataset.handler import DataHandler, DataHandlerLP

import numpy as np
import pandas as pd
from collections import Counter

from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_filter import DataFilter
from qlib import data

class LstmNumpyDataset(TFTDataset):
    """
    自定义数据集，直接从numpy文件中获取数据，并进行相关处理
    """

    def __init__(self, data_path=None,step_len = 30,pred_len = 5,data_type="date_range",instrument_pick=None,
                 aug_type="yes",model_type="tft",low_threhold=-5,high_threhold=5,over_time=2,col_def={},**kwargs):
        super().__init__(col_def=col_def,step_len=step_len,pred_len=pred_len,**kwargs)
        
        self.future_covariate_col = col_def['future_covariate_col']
        self.past_covariate_col = col_def['past_covariate_col']
        self.static_covariate_col = col_def['static_covariate_col']    
        self.low_threhold = low_threhold
        self.high_threhold = high_threhold
        self.over_time = over_time        
        self.aug_type = aug_type
        self.columns = self.get_seq_columns()
        self.training_cutoff = 0.8
        self.model_type = model_type
        self.data_type = data_type
        self.instrument_pick = instrument_pick
        
        self.data = np.load(data_path,allow_pickle=True)
        self.training_data,self.val_data = self.build_data_split()

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

    def get_seq_columns(self):  
        time_column = self.col_def["time_column"]
        target_column = self.col_def["target_column"]      
        future_columns = self.col_def["future_covariate_col"]  
        columns = [time_column] + future_columns + [target_column]
        return columns 
    
    def get_custom_numpy_dataset(self,mode="train"):
        """
        直接使用numpy数据取得DataSet对象

        Parameters
        ----------
        numpy_data : numpy数据
        """     
        
        future_covariate_index = [self.columns.index(item) for item in self.future_covariate_col]
        past_covariate_index = self.get_past_column_index()
        target_index = self.get_target_column_index()
        
        if mode=="train":
            data = self.training_data
        else:
            data = self.val_data
        
        return CustomNumpyDataset(
            data,
            self.step_len-self.pred_len,
            self.pred_len,
            future_covariate_index,
            past_covariate_index,
            static_covariate_index=[],    
            target_index=target_index,
            model_type=self.model_type   
        ) 
        
    def build_data_split(self):  
        """"生成训练集和测试集"""
        
        time_begin = self.data[:,:,-1].min()
        time_end = self.data[:,:,-1].max()
        # 根据长度取得切分点
        sp_index = time_begin + (time_end - time_begin)*self.training_cutoff
        # 训练集使用时间序列最后一个时间点进行切分
        training_data = self.data[self.data[:,-1,-1]<sp_index]
        # 测试集使用时间序列第一个时间点进行切分
        val_data = self.data[self.data[:,0,-1]>sp_index]
            
        # 归一化处理
        training_data,val_data = self.normolize(training_data,val_data)
        
        return training_data,val_data
    
    def normolize(self,training_data,val_data):
        """对目标值进行归一化"""

        target_scaler = self.get_scaler()
        target_index = self.get_target_column_index()
        training_data[:,:,target_index] = target_scaler.fit_transform(training_data[:,:,target_index])   
        val_data[:,:,target_index] = target_scaler.transform(val_data[:,:,target_index]) 
        
        def transfer_data(ori_data):
            source_data = np.expand_dims(ori_data.reshape(-1),axis=-1)
            target_data = scaler.fit_transform(source_data)      
            target_data = target_data.reshape(ori_data.shape)     
            return  target_data     
        
        # 对过去协变量值进行标准化(归一化)
        for index in self.get_past_column_index():
            scaler = self.get_scaler()
            training_data[:,:,index] = transfer_data(training_data[:,:,index])
            val_data[:,:,index] = transfer_data(val_data[:,:,index])

        # 对未来协变量值进行标准化(归一化)
        for index in self.get_future_column_index():
            scaler = self.get_scaler()
            training_data[:,:,index] = transfer_data(training_data[:,:,index])
            val_data[:,:,index] = transfer_data(val_data[:,:,index])
                        
        return training_data,val_data
        
    def view_datatime(self,training_data,val_data):
        t_time = training_data[:,:,-1]
        v_time = val_data[:,:,-1]
        print("training time max:{},val time min:{}".format(t_time.max(), v_time.min()))
        
    