import torch
import torch.nn.functional as F

import numpy as np

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE,KMeansSMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,get_simple_class

def slope_compute(target_ori):
    target = target_ori
    mask_idx = np.where(target<0.01)[0]
    target[mask_idx] = 0.01
    target_slope = (target[1:,:] - target[:-1,:])/target[:-1,:]
    return target_slope
        
def slope_classify_compute(target_ori,threhold=2):
    """生成基于斜率的目标分类"""
    
    target = target_ori
    target_slope = (target[1:,0]  - target[:-1,0])/target[:-1,0]
    if np.sum(abs(target_slope)<(threhold/100))==target_slope.shape[0]:
        return SLOPE_SHAPE_SMOOTH
    if np.sum(target_slope[:-1]<(threhold/100))==target_slope.shape[0]-1 and target_slope[-1]>(threhold/100):
        return SLOPE_SHAPE_SMOOTH    
    return SLOPE_SHAPE_SHAKE

def slope_classify_compute_batch(target,threhold=2,mode=1,num=3):
    """生成基于斜率的目标分类"""
    
    target_slope = (target[:,1:]  - target[:,:-1])/target[:,:-1]
    if mode==1:
        slope_index_bool = torch.abs(target_slope)<(threhold/100)
        slope_index_bool = torch.all(slope_index_bool,dim=-1)
    if mode==2:
        slope_index_bool = target_slope>(threhold/100)
        slope_index_bool = torch.sum(slope_index_bool,dim=1)>=num
    return slope_index_bool

def slope_last_classify_compute(target,threhold=0.05):
    """生成基于斜率的目标分类"""
    
    # 给每段计算斜率,由于刻度一致，因此就是相邻元素的差,重点关注最后一段
    target_slope = np.array([target[-2,0]  - target[-3,0],target[-1,0]  - target[-2,0]])
    if np.sum(target_slope>0)==2:
        return SLOPE_SHAPE_RAISE    
    if np.sum(target_slope<0)==2:
        return SLOPE_SHAPE_FALL
    return SLOPE_SHAPE_SMOOTH

def mae_comp(input,target):
    loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    loss = loss_fn(input.float(), target.float())
    return loss

def np_qcut(arr, q):
    """实现类似pandas的qcut功能"""

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

def normalization(data,res=0.001,mode="numpy",avoid_zero=True):
    if mode=="numpy":
        rtn = (data - np.min(data,axis=0) + res)/(np.max(data,axis=0)-np.min(data,axis=0) + res) 
    else:
        rtn = (data - torch.min(data,dim=0)[0] + res)/(torch.max(data,dim=0)[0]-torch.min(data,dim=0)[0] + res) 
    if avoid_zero:
        rtn = rtn + res  
    return rtn

def price_range_normalization(data,res=0.001,mode="numpy",avoid_zero=True):
    """针对股市涨跌幅度，进行统一的归一化"""
    
    MAX_RANGE = 1.5
    MIN_RANGE = 0.8
    
    # 根据固定的总体幅度，进行最大最小化
    max_value = data[:,0] * (1+MAX_RANGE)
    min_value = data[:,0] * (1-MIN_RANGE)
    
    if mode=="numpy":
        rtn = (data - np.min(data,axis=0))/(max_value-min_value) 
    else:
        rtn = (data - torch.min(data,dim=0)[0])/(torch.max(data,dim=0)[0]-torch.min(data,dim=0)[0]) 
    if avoid_zero:
        rtn = rtn + res  
    return rtn

def price_range_inverse_normalization(data,res=0.001,mode="numpy",avoid_zero=True):
    """针对股市涨跌幅度，进行反向的归一化"""
    
    MAX_RANGE = 0.85
    MIN_RANGE = 0.08
    
    # 根据固定的总体幅度，进行最大最小化
    max_value = data[:,0] * (1+MAX_RANGE)
    min_value = data[:,0] * MIN_RANGE
    
    if mode=="numpy":
        rtn = (data - np.min(data,axis=0))/(max_value-min_value) 
    else:
        rtn = (data - torch.min(data,dim=0)[0])/(torch.max(data,dim=0)[0]-torch.min(data,dim=0)[0]) 
    if avoid_zero:
        rtn = rtn + res  
    return rtn

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

def compute_price_class(price_array,mode="max_range"):   
    cur_price = price_array[0]
    if mode=="max_range":
        max_value = np.max(price_array)
        min_value = np.min(price_array)
        if max_value - cur_price > cur_price - min_value:
            raise_range = (max_value - cur_price)/cur_price*100
        else:
            raise_range = (min_value - cur_price)/cur_price*100         
    if mode=="first_last":
        raise_range = (price_array[-1] - cur_price)/cur_price*100             
    if mode=="fast":
        raise_range = (price_array[2] - cur_price)/cur_price*100                
    p_taraget_class = get_simple_class(raise_range)     
    return p_taraget_class

def compute_price_class_batch(price_array,mode="first_last"):   
    cur_price = price_array[:,0]        
    if mode=="first_last":
        raise_range = (price_array[:,-1] - cur_price)/cur_price*100             
    if mode=="fast":
        raise_range = (price_array[:,-3] - cur_price)/cur_price*100                
    p_taraget_class = np.array([get_simple_class(item) for item in raise_range])
    return p_taraget_class,raise_range

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

def comp_max_and_rate(np_arr,threhold=-1):
    """计算最大值类别以及置信度"""
    
    arr = torch.tensor(np_arr)
    pred_class = F.softmax(arr,dim=-1)
    pred_class = torch.max(pred_class,dim=-1)    
    if threhold!=-1:
        rtn = torch.where(pred_class[0]>threhold)[0]
    else:
        rtn = pred_class[1]
    rtn = rtn.numpy()
    return rtn

if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    mae_comp(input,target)
    
       
    