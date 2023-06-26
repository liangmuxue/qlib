from typing import Callable, Dict, Iterable, List, Tuple, Union
import pandas as pd
import numpy as np
import torch

MAX_RANGE = 1.85
MIN_RANGE = 0.08

def transform_slope_value(target_value):
    slope_value =  1 + (target_value[:,1:] - target_value[:,:-1])/target_value[:,:-1]*10
    slope_value = torch.where(slope_value>0,slope_value,0.0)
    slope_value = torch.where(slope_value<=2,slope_value,2.0)
    return slope_value
        
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
        
        