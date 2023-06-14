from visdom import Visdom
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE,KMeansSMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH

def slope_classify_compute(target_ori,class1_len,threhold=0.1):
    """生成基于斜率的目标分类"""
    
    # 再次归一化，只衡量目标数据部分--取消
    # target = target_scale(target_ori)
    target = target_ori
    # 给每段计算斜率,由于刻度一致，因此就是相邻元素的差
    target_slope = target[1:,0]  - target[:-1,0]
    # 分为2个部分，分别分类
    split_arr = [target_slope[:class1_len],target_slope[class1_len:]]    
    result = [0,0]
    # 重点关注最后一段
    for index,item in enumerate(split_arr):
        # 每一段斜率都比较小，则类型为平稳
        if np.sum(np.abs(item)<threhold)==item.shape[0]:
            result[index] = SLOPE_SHAPE_SMOOTH
            continue
        # 持续上升
        if np.sum(item>0)==item.shape[0]:
            result[index] = SLOPE_SHAPE_RAISE
            continue        
        # 持续下降
        if np.sum(item<0)==item.shape[0]:
            result[index] = SLOPE_SHAPE_FALL
            continue 
        # 以上情况都不是，则为震荡
        result[index] = SLOPE_SHAPE_SHAKE
    return result


def slope_last_classify_compute(target,threhold=0.05):
    """生成基于斜率的目标分类"""
    
    # 给每段计算斜率,由于刻度一致，因此就是相邻元素的差,重点关注最后一段
    target_slope = np.array([target[-2,0]  - target[-3,0],target[-1,0]  - target[-2,0]])
    if np.sum(np.abs(target_slope)<threhold)==2:
        return SLOPE_SHAPE_SMOOTH
    if np.sum(target_slope>0)==2:
        return SLOPE_SHAPE_RAISE    
    if np.sum(target_slope<0)==2:
        return SLOPE_SHAPE_FALL
    if (target_slope[0]+target_slope[1])>0 and target_slope[1]>0:
        return SLOPE_SHAPE_RAISE 
    if (target_slope[0]+target_slope[1])>0 and target_slope[0]>0 \
            and target_slope[0]>2*abs(target_slope[1]):
        return SLOPE_SHAPE_RAISE     
    return SLOPE_SHAPE_SHAKE

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

def compute_price_range(price_arr):
    """根据价格，计算涨跌幅"""
    
    if isinstance(price_arr, list):
        price_arr = np.array(price_arr)
    price_arr_before = price_arr[:-1]
    price_arr_after = price_arr[1:]   
    slope_range = (price_arr_after - price_arr_before)/price_arr_before*100
    return slope_range

def target_scale(target_ori,range=0.1):
    """针对股市涨跌幅度，实现期间缩放"""
    
    # 把负数处理到正区间
    min_value = np.min(target_ori)
    if min_value<0:
        target = target_ori + abs(min_value)
        min_value = 0
    else:
        target = target_ori
    # 设定最大值为最大涨幅--即每天涨停的情况下的总涨幅    
    total_range = range * target.shape[0]
    max_value = min_value * (1+total_range)
    _range = max_value - min_value
    # 归一化，避免出现0值
    result = (target - min_value + 0.01)/_range
    return result

def comp_max_and_rate(np_arr):
    """计算最大值类别以及置信度"""
    
    arr = torch.tensor(np_arr)
    pred_class = F.softmax(arr,dim=-1)
    pred_class = torch.max(pred_class,dim=-1)    
    return pred_class[1].item(),pred_class[0].item()

if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    mae_comp(input,target)
    
       
    