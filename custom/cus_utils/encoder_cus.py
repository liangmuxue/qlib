from typing import Callable, Dict, Iterable, List, Tuple, Union
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from sklearn.preprocessing import MinMaxScaler

MAX_RANGE = 1.85
MIN_RANGE = 0.08

def transform_slope_value_no_inverse(target_value):
    slope_value =  (target_value[:,1:] - target_value[:,:-1])/target_value[:,:-1]
    return slope_value

def transform_slope_value(target_value):
    slope_value = (target_value[:,1:] - target_value[:,:-1])/target_value[:,:-1]*10
    if isinstance(target_value, torch.Tensor):
        slope_value = torch.where(slope_value<1,slope_value,1.0)
        slope_value = torch.where(slope_value>-1,slope_value,-1.0)
    else:
        slope_value = np.where(slope_value<1,slope_value,1.0)
        slope_value = np.where(slope_value>-1,slope_value,-1.0)        
    return slope_value

def unverse_transform_slope_value(slope_value):
    target_range = slope_value/10
    return target_range

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def rolling_norm(arr,rolling_length,step_len):
    """滚动求时间段内差值的归一化数据"""
    
    # 前留1，后留预测时间段长度
    arr_padded = np.pad(arr,(1,step_len), mode='constant',constant_values=(0,0))
    # 滚动取得多段矩阵
    strides_data = rolling_window(arr_padded,rolling_length)    
    # 每个段内进行归一化
    scale_data = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(strides_data.transpose(1,0)).transpose(1,0)    
    # 求归一化后的差值
    round_vals = scale_data[:,-1] - scale_data[:,-step_len-1]
    # 整体再次归一化
    norm_vals = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(np.expand_dims(round_vals,-1)).squeeze(-1) 
    # 去掉之前补充的多余内容
    norm_vals = norm_vals[1:-step_len]
    round_vals = round_vals[1:-step_len]
    
    return norm_vals,round_vals
    
class StockNormalizer(object):
    """针对股票数据的标准化归一化工具类"""

    def __init__(self,norm_mode="price_range"):
        self.norm_mode = norm_mode
        self.ori_data = None
        self.keep_inverse_data = None

    def fit(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]):
        """数据准备"""
        
        # 根据固定的总体幅度，进行最大最小化
        self.max_value = data[0,0] * MAX_RANGE
        self.min_value = data[0,0] * MIN_RANGE  
             
        return self

    def transform(
        self,
        tar_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]:
        """使用之前数据的进行变换"""
        
        # 整体范围下的局部范围，,到不了0到1
        if isinstance(tar_data,np.ndarray):
            rtn = (self.max_value - tar_data)/(self.max_value - self.min_value) 
        else:
            rtn = (self.max_value - tar_data)/(self.max_value - self.min_value) 
        
        return rtn
        
    def fit_transform(
        self,
        tar_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]:
        
        rtn = self.fit(tar_data).transform(tar_data)
        return rtn        
        
    def inverse_transform(
        self,
        tar_data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    ) -> Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]:
        """反向归一化"""
        
        rtn = self.max_value - tar_data * (self.max_value - self.min_value)
        return rtn          
 
class MinMaxNormalizer(MinMaxScaler):

    def __init__(self,feature_range=(0.001,1)):
        super().__init__(feature_range=feature_range)

    def fit(self, data):
        
        super().fit(data)
        return self

    def transform(self,tar_data):
        return super().transform(tar_data)
        
    def fit_transform(self,tar_data):
        
        rtn = self.fit(tar_data).transform(tar_data)
        return rtn        
        
    def inverse_transform(self,tar_data):
        """反向归一化"""
        
        rtn = super().inverse_transform(tar_data)
        return rtn     
           
        