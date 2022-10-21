import numpy as np
import pandas as pd
import numba as nb
from numba import cuda
import _sysconfigdata_x86_64_conda_linux_gnu
from numba.core.types import none

def check_time_ser_data(file_path):
    """检查原数据的时间长度"""
        
    df = pd.read_pickle(file_path)
    cnt = df.groupby("instrument").count()
    print(cnt)

class DataFilter():
    def __init__(self):
        self.wave_period = 30
        self.forecast_horizon = 5

    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
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
            # 查找每行,在后面几个数里,超过阈值的总个数
            if wave_threhold_type == "more":
                over_data = (rolling_data[:,wave_period-check_length:]>threhold).sum(axis=1)
            else:
                over_data = (rolling_data[:,wave_period-check_length:]<threhold).sum(axis=1)
            start_arr = np.where(over_data>=over_time)[0]
            if start_arr.shape[0]==0:
                return np.array([])
            # 根据坐标数据,以及wave_period长度，取得分段数据
            index_arr = np.array([np.arange(item,item+wave_period,dtype=np.int32) for item in start_arr])
            # target_data = [data.take(index) for index in index_arr]
            target_data = data.values.take(index_arr,axis=0)
            return target_data      
        
        # 进行滚动操作，查找指定期限内，符合阈值条件的多个系列
        # rtn = data[target_column].rolling(window=wave_period,axis=0, min_periods=wave_period).apply(handle,args=(wave_threhold,over_time,wave_period-forecast_horizon),raw=True).dropna()
        
        target_data = data.groupby(group_column).apply(lambda data:handle(data,wave_threhold,forecast_horizon,over_time))
        target_data_array = None
        for item in target_data:
            if item.shape[0]>0:
                if target_data_array is None:
                    target_data_array = item
                else:
                    target_data_array = np.concatenate((target_data_array,item),axis=0)
        return target_data_array

    
if __name__ == "__main__":
    file_path = "/home/qdata/project/qlib/custom/data/aug/test_all_timeidx.pkl"
    check_time_ser_data(file_path)
    