import numpy as np
import pandas as pd
import numba as nb
from numba import cuda
import _sysconfigdata_x86_64_conda_linux_gnu
from numba.core.types import none

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def check_time_ser_data(file_path):
    """检查原数据的时间长度"""
        
    df = pd.read_pickle(file_path)
    cnt = df.groupby("instrument").count()
    print(cnt)

def get_topN_dict(ori_dict,n):
    """取得字典前n条数据"""
    
    new_dict = {}
    for i,(k,v) in enumerate(ori_dict.items()):
        new_dict[k]=v
        if i>=n-1:
            break    
    return new_dict
    
class DataFilter():
    def __init__(self):
        self.wave_period = 30
        self.forecast_horizon = 5

    def data_clean(self,data,step_len,valid_range=None,group_column="instrument",time_column="time_idx"):
        """"清除序列长度不够的股票"""
        
        if isinstance(valid_range[0],str):
            valid_start = int(valid_range[0])
        else:
            valid_start = int(valid_range[0].strftime('%Y%m%d'))
        if isinstance(valid_range[1],str):
            valid_end = int(valid_range[1])
        else:
            valid_end = int(valid_range[1].strftime('%Y%m%d'))        
        thr_number = step_len
        # 清除训练中长度不够的股票数据
        df_train = data[(data["datetime_number"]<valid_start)]
        gcnt = df_train.groupby(group_column).count()
        # 全集需要至少2倍的预测长度
        index = gcnt[gcnt[time_column]>=thr_number*2].index
        data = data[data[group_column].isin(index)]
        # 还需要判断验证集是否符合长度要求
        df_val = data[(data["datetime_number"]>=valid_start)&(data["datetime_number"]<valid_end)]
        gcnt = df_val.groupby(group_column).count()
        index = gcnt[gcnt[time_column]>=thr_number].index
        data = data[data[group_column].isin(index)]
        
        return data

    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        print("shape in rolling window",shape)
        if shape[0]<1:
            return None        
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def filter_wave_data(self,data,target_column="ori_label",group_column="instrument",wave_period=30,forecast_horizon=5,wave_threhold=15,wave_threhold_type="more",over_time=2):
        """按照股票分组，寻找波动数据
           target_column: 使用的字段
           group_column: 分组字段
           wave_period:区间数据长度，在此区间内进行寻找
           forecast_horizon：预测长度，计算此范围内的波动数据
           wave_threhold: 比较的阈值
           wave_threhold_type: 阈值类型，超出还是不足
        """
        
        # @nb.njit(nogil=True)
        def handle(data,threhold,check_length,over_time):
            # 首先取得滚动后的目标2维数据，每行长度为wave_period
            rolling_data = self.rolling_window(data[target_column].values, wave_period)   
            if rolling_data is None:
                return None      
            # 查找每行,在后面几个数里,超过阈值的总个数
            if wave_threhold_type == "more":
                over_data = (rolling_data[:,wave_period-check_length:]>threhold).sum(axis=1)
            else:
                over_data = (rolling_data[:,wave_period-check_length:]<threhold).sum(axis=1)
            start_arr = np.where(over_data>=over_time)[0]
            if start_arr.shape[0]==0:
                return None
            # 根据坐标数据,以及wave_period长度，取得分段数据
            index_arr = np.array([np.arange(item,item+wave_period,dtype=np.int32) for item in start_arr])
            # target_data = [data.take(index) for index in index_arr]
            target_data = data.values.take(index_arr,axis=0)
            return target_data    
          
            
        # 按股票分组，病进行滚动操作，查找指定期限内，符合阈值条件的多个系列
        if group_column is None:
            target_data = handle(data,wave_threhold,forecast_horizon,over_time)
            return target_data
        target_data = data.groupby(group_column).apply(lambda data:handle(data,wave_threhold,forecast_horizon,over_time))
        print("do loop choice:{}".format(len(target_data)))
        target_data = target_data.dropna()
        target_data = np.concatenate(target_data.values, axis=0)
        return target_data
        
        # @nb.njit(nogil=True)
        # def combine_data(target_data):
        #     data_array = None
        #     for item in target_data:
        #         if item.shape[0]>0:
        #             if data_array is None:
        #                 data_array = item
        #             else:
        #                 data_array = np.concatenate((data_array,item),axis=0)
        #     return data_array
        #
        # return combine_data(target_data.values)

    def get_data_with_threhold(self,data,column_index,wave_threhold_type="more",threhold=5,wave_period=30,check_length=5,over_time=2):
        """筛选numpy数据，查找涨幅或跌幅超出的部分"""
        
        # 查找每行,在后面几个数里,超过阈值的总个数
        if wave_threhold_type == "more":
            over_data = (data[:,wave_period-check_length:,column_index]>threhold).sum(axis=1)
        else:
            over_data = (data[:,wave_period-check_length:,column_index]<threhold).sum(axis=1)
        index_arr = np.where(over_data>=over_time)[0]
        target_data = data.take(index_arr,axis=0)   
        return target_data

    def get_combine_data_with_threhold(self,data,column_index,low_threhold=-5,high_threhold=5,wave_period=30,check_length=5,over_time=1):
        """筛选numpy数据，查找涨幅或跌幅超出的部分"""
        
        # 查找每行,在后面几个数里,超过阈值的总个数
        over_data = ((data[:,wave_period-check_length:,column_index]>low_threhold) & (data[:,wave_period-check_length:,column_index]<high_threhold)).sum(axis=1)
        index_arr = np.where(over_data>=over_time)[0]
        target_data = data.take(index_arr,axis=0)   
        return target_data
            
if __name__ == "__main__":
    file_path = "/home/qdata/project/qlib/custom/data/aug/test_all_timeidx.pkl"
    check_time_ser_data(file_path)
    