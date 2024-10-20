from qlib.data.dataset.handler import DataHandler, DataHandlerLP

import numpy as np
import pandas as pd
from collections import Counter

from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_filter import DataFilter
from qlib import data


class TFTNumpyDataset(TFTDataset):
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

    def filter_balance_data_by_bins(self,data,bins):
        """根据bins筛选出均衡的数据"""
        
        wave_period = self.step_len
        forecast_horizon = self.pred_len
        over_time = self.over_time
                        
        data_filter = DataFilter()
        target_index = self.get_target_column_index()
        target_data = data[:,wave_period-forecast_horizon:,target_index].reshape(-1)
        # 取得分箱统计数据，以及对应的数量最大最小值
        digitized = np.digitize(target_data, bins)
        counter_values = Counter(digitized)
        max_cnt = max(counter_values.values())
        min_cnt = min(counter_values.values())
        # 分箱进行筛选，并根据最小值进行截取
        rtn_data = None
        for index,item in enumerate(bins):
            if index == (len(bins) -1) :
                break
            low_threhold = bins[index]
            high_threhold = bins[index+1]
            combine_data = data_filter.get_combine_data_with_threhold(data,target_index,high_threhold=high_threhold,low_threhold=low_threhold,
                                                          wave_period=wave_period,check_length=forecast_horizon,over_time=over_time)
            combine_data = combine_data[:min_cnt]
            if rtn_data is None:
                rtn_data = combine_data
            else:
                rtn_data = np.concatenate((rtn_data,combine_data),axis=0)
        return rtn_data

    def enhance_data(self,data):              
        from cus_utils.common_compute import np_qcut,enhance_data_complex
        bins = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0, 1, 2, 3,4, 5, 6, 7,8, 9, 10]
        target_index = self.get_target_column_index()
        start = self.step_len - 1
        data_process =  np.reshape(data, (data.shape[0], data.shape[2]*data.shape[1]))    
        data_target = data[:,start:,target_index].reshape(-1)
        amplitude,y_res = enhance_data_complex(data_process,data_target,bins=bins)
        amplitude = np.reshape(amplitude, data_process.shape)
        amplitude[:,target_index]= y_res
        return amplitude
        
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
        
        time_begin = self.data[:,:,-1].min()
        time_end = self.data[:,:,-1].max()
        data = self.data
        # 选择指定股票
        if self.instrument_pick is not None:
            ins_index = self.get_group_column_index()
            data_array = None
            for instrument in self.instrument_pick:
                data_item = self.data[self.data[:,0,ins_index]==instrument,:,:]
                if data_array is None:
                    data_array = data_item
                else:
                    data_array = np.concatenate((data_array,data_item),axis=0)
            data = data_array
            
        # 使用时间范围筛选数据
        if self.data_type=="date_range":
            train_range = self.segments["train"]
            train_start = int(train_range[0].strftime('%Y%m%d'))
            train_end = int(train_range[1].strftime('%Y%m%d'))
            valid_range = self.segments["valid"]
            valid_start = int(valid_range[0].strftime('%Y%m%d'))
            valid_end = int(valid_range[1].strftime('%Y%m%d'))         

        else:
            # 根据长度取得切分点
            sp_index = time_begin + (time_end - time_begin)*self.training_cutoff
            # 训练集使用时间序列最后一个时间点进行切分
            training_data = data[data[:,-1,-1]<sp_index]
            # 测试集使用时间序列第一个时间点进行切分
            val_data = data[data[:,0,-1]>sp_index]
            
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
        training_data,val_data = self.normolize(data,(train_start,train_end),(valid_start,valid_end))
        
        return training_data,val_data
    
    def normolize(self,data,training_range=None,val_range=None):
        """对数据进行归一化"""
        
        train_start,train_end = training_range
        valid_start,valid_end = val_range
        training_data = data[(data[:,-1,-1]> train_start) & (data[:,-1,-1]<train_end)]
        val_data = data[(data[:,0,-1]> valid_start) & (data[:,0,-1]<valid_end)]
        
        # 目标数据归一化
        target_scaler = self.get_scaler()
        target_index = self.get_target_column_index()
        training_data[:,:,target_index] = target_scaler.fit_transform(training_data[:,:,target_index])   
        val_data[:,:,target_index] = target_scaler.transform(val_data[:,:,target_index]) 
                
        def transfer_data(column_index):
            scaler = self.get_scaler()
            scaler.fit(training_data[:,:,column_index])      
            t_data = scaler.transform(training_data[:,:,column_index])     
            v_data = scaler.transform(val_data[:,:,column_index]) 
            return  t_data, v_data    
        
        # 对过去协变量值进行标准化(归一化)
        for index in self.get_past_column_index():
            training_data[:,:,index],val_data[:,:,index] = transfer_data(index)

        # 对未来协变量值进行标准化(归一化)
        for index in self.get_future_column_index():
            training_data[:,:,index],val_data[:,:,index] = transfer_data(index)
                        
        return training_data,val_data
    

        
    