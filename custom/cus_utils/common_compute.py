from visdom import Visdom
import torch
import numpy as np
from torch import nn

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE,KMeansSMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

             
def mae_comp(input,target):
    loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    loss = loss_fn(input.float(), target.float())
    print(loss)
    return loss

def np_qcut(arr, q):
    """ 实现类似pandas的qcut功能"""

    res = np.zeros(arr.size)
    na_mask = np.isnan(arr)
    res[na_mask] = np.nan
    x = arr[~na_mask]
    sorted_x = np.sort(x)
    idx = np.linspace(0, 1, q+1) * (sorted_x.size - 1)
    pos = idx.astype(int)
    fraction = idx % 1
    a = sorted_x[pos]
    b = np.roll(sorted_x, shift=-1)[pos]
    bins = a + (b - a) * fraction
    bins[0] -= 1 
    
    res[~na_mask] = np.digitize(x, bins, right=True)
    return res

def enhance_data_complex(ori_data,target_data,mode="smote",bins=None):
    """综合数据增强，使用imblearn组件
    Parameters
    ----------
    ori_data : 需要增强的特征数据，numpy数组
    target_data : 需要增强的label数据，numpy数组
    mode : 增强模式
    bins : 数据间隔分组数
    Returns
    ----------
    增强后的数据
    """
    
    # 使用分箱范围数据进行数据分组
    digitized = np.digitize(target_data, bins)
    # print('Original dataset shape %s' % Counter(digitized))
    if mode=="smote":
        sm = KMeansSMOTE(random_state=42,cluster_balance_threshold=10)    
    # sm = SMOTE(random_state=42) 
    if mode=="adasyn":
        sm = ADASYN(random_state=42) 
    # 过采样数据补充
    amplitude, y_res = sm.fit_resample(ori_data, digitized)  
    # print('Resampled dataset shape %s' % Counter(y_res))
    amplitude = np.squeeze(amplitude,axis=1)     
    return amplitude,y_res

def enhance_data(ori_data,mode="smote",bins=None):
    """数据增强"""
    
    digitized = np.digitize(ori_data, bins)
    print('Original dataset shape %s' % Counter(digitized))
    amplitude = np.expand_dims(ori_data,axis=1)
    if mode=="smote":
        sm = BorderlineSMOTE(random_state=42,kind="borderline-1")    
    # sm = SMOTE(random_state=42) 
    if mode=="adasyn":
        sm = ADASYN(random_state=42) 
    amplitude, y_res = sm.fit_resample(amplitude, digitized)  
    print('Resampled dataset shape %s' % Counter(y_res))
    amplitude = np.squeeze(amplitude,axis=1)     
    return amplitude,y_res

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def compute_series_slope(series_data):
    """计算序列斜率,分段计算"""
    
    slope_arr = []
    for index in range(len(series_data)):
        if index==len(series_data)-1:
            break
        x = [1,2]
        y = series_data[index:index+2]
        slope, intercept = np.polyfit(x,y,1)
        slope_arr.append(slope)
        
    return slope_arr
        
if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    mae_comp(input,target)
    
       
    