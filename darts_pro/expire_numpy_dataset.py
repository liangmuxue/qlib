from qlib.data.dataset.handler import DataHandler, DataHandlerLP

import numpy as np
import pandas as pd
from collections import Counter

from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_filter import DataFilter
from qlib import data

class ExpNumpyDataset(TFTDataset):
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
        self.training_cutoff = 0.9
        self.model_type = model_type
        self.data_type = data_type
        self.instrument_pick = instrument_pick
        
        self.data = np.load(data_path,allow_pickle=True)
        self.training_data,self.val_data = self.build_data_split()

    def get_seq_columns(self):  
        time_column = self.col_def["time_column"]
        target_column = self.col_def["target_column"]      
        future_columns = self.col_def["future_covariate_col"]  
        columns = [time_column] + future_columns + [target_column]
        columns = [time_column] + ["Year","Month"] + [target_column]
        return columns  
        
    def get_custom_numpy_dataset(self,mode="train"):
        """
        直接使用numpy数据取得DataSet对象

        Parameters
        ----------
        numpy_data : numpy数据
        """     
        
        future_covariate_index = self.get_future_column_index()
        past_covariate_index = self.get_past_column_index()
        static_covariate_index = [0]
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
            static_covariate_index,    
            target_index,
            model_type=self.model_type   
        ) 
        
    def build_data_split(self):  
        """"生成训练集和测试集"""
        
        time_column_index = self.get_time_column_index()
        time_begin = self.data[:,:,time_column_index].min()
        time_end = self.data[:,:,time_column_index].max()
        # 使用时间范围筛选数据
        if self.data_type=="date_range":
            train_range = self.segments["train"]
            train_start = int(train_range[0].strftime('%Y%m%d'))
            train_end = int(train_range[1].strftime('%Y%m%d'))
            valid_range = self.segments["valid"]
            valid_start = int(valid_range[0].strftime('%Y%m%d'))
            valid_end = int(valid_range[1].strftime('%Y%m%d'))         
            training_data = self.data[(self.data[:,-1,-1]> train_start) & (self.data[:,-1,-1]<train_end)]
            val_data = self.data[(self.data[:,0,-1]> valid_start) & (self.data[:,0,-1]<valid_end)]
            # 选择指定股票
            if self.instrument_pick is not None:
                ins_index = self.get_group_column_index()
                training_data_array = None
                val_data_array = None
                for instrument in self.instrument_pick:
                    training_data_item = training_data[training_data[:,0,ins_index]==instrument,:,:]
                    val_data_item = val_data[val_data[:,0,ins_index]==instrument,:,:]
                    if training_data_array is None:
                        training_data_array = training_data_item
                        val_data_array = val_data_item
                    else:
                        training_data_array = np.concatenate((training_data_array,training_data_item),axis=0)
                        val_data_array = np.concatenate((val_data_array,val_data_item),axis=0)
                training_data = training_data_array
                val_data = val_data_array
        else:
            # 根据长度取得切分点
            sp_index = time_begin + (time_end - self.step_len - time_begin)*self.training_cutoff
            sp_index = 121
            # 训练集使用时间序列第一个时间点进行切分
            training_data = self.data[self.data[:,-1,time_column_index]<sp_index]
            # 测试集使用时间序列第一个时间点进行切分
            val_data = self.data[self.data[:,0,time_column_index]>sp_index]
            
        # 根据条件决定是否筛选数据，取得均衡
        if self.aug_type=="yes":
            # bins=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0, 1, 2, 3,4, 5, 6, 7,8, 9, 10]
            bins = [-10,-7,-4,-1, 1, 4,7,10]
            training_data = self.filter_balance_data_by_bins(training_data,bins=bins)
            # 测试集不进行筛选
            val_index = np.random.randint(1,val_data.shape[0],(training_data.shape[0],))
            # val_data = val_data[val_index,:,:]         
            val_data = val_data[:15000,:,:]   
            # val_data = self.filter_balance_data_by_bins(val_data,bins=bins)
            
        # 归一化处理
        training_data,val_data = self.normolize(training_data,val_data)
        return training_data,val_data
    
    def normolize(self,training_data,val_data):
        # 对目标值进行归一化

        target_scaler = self.get_scaler()
        target_index = self.get_target_column_index()
        training_data[:,:,target_index] = target_scaler.fit_transform(training_data[:,:,target_index])   
        val_data[:,:,target_index] = target_scaler.transform(val_data[:,:,target_index]) 
        
        def transfer_data(ori_data):
            source_data = np.expand_dims(ori_data.reshape(-1),axis=-1)
            target_data = scaler.fit_transform(source_data)      
            target_data = target_data.reshape(ori_data.shape)     
            return  target_data     
        
        # 对协变量值进行标准化(归一化)
        for index in self.get_future_column_index():
            scaler = self.get_scaler()
            target_index = self.get_target_column_index()
            training_data[:,:,index] = transfer_data(training_data[:,:,index])
            val_data[:,:,index] = transfer_data(val_data[:,:,index])
            
        return training_data,val_data
    
    def view_datatime(self,training_data,val_data):
        t_time = training_data[:,:,-1]
        v_time = val_data[:,:,-1]
        print("training time max:{},val time min:{}".format(t_time.max(), v_time.min()))
        
    