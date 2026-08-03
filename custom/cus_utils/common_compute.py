import torch
import torch.nn.functional as F
from torchmetrics.regression import ConcordanceCorrCoef

import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE,KMeansSMOTE
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,get_simple_class
from pip._internal.models import candidate


def intersect1d_preserve_order(ar1, ar2, return_unique: bool = True):
    """
    两个数组求交集，保持 ar1 内部出现顺序
    :param ar1: 基准数组（顺序以此为准）
    :param ar2: 对比数组
    :param return_unique: True 返回不重复交集；False 保留ar1内所有匹配重复元素
    :return: 交集数组
    """
    common = np.intersect1d(ar1, ar2)
    mask = np.isin(ar1, common)
    filtered = ar1[mask]
    if return_unique:
        _, idx = np.unique(filtered, return_index=True)
        filtered = filtered[np.sort(idx)]
    return filtered

def check_iqr_outler(data,range=3):
    """四分位排查异常值"""
    
    col = "target"
    df = pd.DataFrame(data,columns = [col] )
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - range * IQR
    upper_bound = Q3 + range * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    return outliers

def standardize_dict_tensor(original_dict):
    """
    对字典中所有PyTorch张量进行全局标准化，并映射回原字典
    :param original_dict: 输入字典，value为torch.Tensor
    :return: 标准化后的新字典（结构、key、形状完全一致）
    """
    # ---------------------- 步骤1：保存每个tensor的形状和长度 ----------------------
    tensor_info = []  # 存储 (key, 原始形状, 元素总数)
    all_tensors = []  # 存储所有展平后的tensor
    
    for key, tensor in original_dict.items():
        # 保存原始信息
        shape = tensor.shape
        numel = tensor.numel()  # 总元素个数
        tensor_info.append((key, shape, numel))
        
        # 展平并收集
        all_tensors.append(tensor.reshape(-1))

    # ---------------------- 步骤2：拼接所有张量 ----------------------
    concatenated = torch.cat(all_tensors)

    # ---------------------- 步骤3：全局标准化（均值0，方差1） ----------------------
    mean = concatenated.mean()
    std = concatenated.std()
    
    # 防止标准差为0
    std = std if std > 1e-6 else torch.tensor(1.0)
    standardized = (concatenated - mean) / std

    # ---------------------- 步骤4：按原始长度切分并映射回字典 ----------------------
    result_dict = {}
    start_idx = 0
    
    for key, shape, numel in tensor_info:
        # 截取当前key对应的标准化数据
        end_idx = start_idx + numel
        part = standardized[start_idx:end_idx]
        
        # 恢复原始形状
        part = part.reshape(shape)
        result_dict[key] = part
        
        # 更新起始索引
        start_idx = end_idx

    return result_dict

def map_to_neg1_pos1_torch(tensor: torch.Tensor) -> torch.Tensor:
    """
    PyTorch 实现：将张量映射到 [-1, 1] 区间，保留原最大正数/最小负数的比值
    支持：CPU/GPU张量、任意维度、自动求导
    :param tensor: 输入张量 (可包含正数、负数、0)
    :return: 归一化到 [-1, 1] 的张量
    """
    # 克隆避免修改原张量
    x = tensor.clone()
    
    # 1. 全局最大/最小值（支持任意维度张量）
    max_val = x.max()
    min_val = x.min()
    
    # 2. 计算缩放尺度：取正负极值绝对值的最大值
    scale = torch.max(torch.abs(max_val), torch.abs(min_val))
    
    # 3. 安全处理：全0张量避免除以0
    scale = torch.where(scale == 0, torch.tensor(1.0, device=x.device), scale)
    
    # 4. 核心映射公式
    mapped_tensor = x / scale
    
    return mapped_tensor

def map_to_neg1_pos1(arr):
    """
    将数组映射到 [-1, 1] 区间，保留正负极值的原始比值
    :param arr: 输入数组（可包含正数、负数、0）
    :return: 映射后的数组
    """
    # 转换为 numpy 数组（方便计算）
    arr = np.asarray(arr, dtype=np.float32)
    
    # 1. 获取最大正数、最小负数
    max_val = np.max(arr)
    min_val = np.min(arr)
    
    # 2. 计算缩放尺度：取正负极值绝对值的更大值
    scale = max(abs(max_val), abs(min_val))
    
    # 3. 安全判断：避免除以 0（全 0 数组）
    if scale == 0:
        return np.zeros_like(arr)
    
    # 4. 核心映射公式
    mapped_arr = arr / scale
    
    return mapped_arr

def weighted_signed_score_3d(score_3d, feature_weights=None):
    """
    3维数据仅对最后一维做加权求和（时间步无加权）
    Args:
        score_3d: 3维张量，shape=(batch_size, time_steps, feature_dim)
        feature_weights: 最后一维的权重，shape=(feature_dim,) 或 list/tuple
                         若为None，默认等权重求和（即普通sum）
    Returns:
        output_2d: 2维张量，shape=(batch_size, time_steps)
    """
    batch_size, time_steps, feature_dim = score_3d.shape
    
    # 1. 初始化最后一维的权重（默认等权重）
    if feature_weights is None:
        feature_weights = torch.ones(feature_dim, dtype=score_3d.dtype)
    else:
        # 转为张量并确保形状匹配
        feature_weights = torch.tensor(feature_weights, dtype=score_3d.dtype)
        if feature_weights.shape != (feature_dim,):
            raise ValueError(f"feature_weights形状需为({feature_dim},)，当前是{feature_weights.shape}")
    
    # 2. 扩展权重维度，适配广播（关键！）
    # feature_weights: (feature_dim,) → (1, 1, feature_dim)
    # 这样能和3维输入 (batch, time, feature) 逐元素相乘
    feature_weights = feature_weights.unsqueeze(0).unsqueeze(0)
    feature_weights = feature_weights.to(score_3d.device)
    # 3. 最后一维加权求和（保留正负特征）
    # 先逐元素相乘，再对最后一维求和
    output_2d = (score_3d * feature_weights).sum(dim=-1)
    
    return output_2d

def get_outlier_bounds(df, column,range=2.0):
    """获取异常值的上下边界（IQR法）"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - range * iqr
    upper_bound = q3 + range * iqr
    return lower_bound, upper_bound

def process_outliers_multi_cols(
    df, 
    cols,  # 需要处理的字段列表
    range=2.0,
    method='median_fill',  # 处理方式：median_fill/ truncate/ drop
    detect_method='iqr'    # 异常值识别方式：iqr/ std
):
    """
    批量处理多个字段的异常值
    :param df: 原始DataFrame
    :param cols: 待处理字段列表，如 ['销售额', '订单量', '利润']
    :param method: 处理策略
                   - median_fill: 中位数填充（默认）
                   - truncate: 边界值截断
                   - drop: 删除含异常值的行
    :param detect_method: 异常值识别方式
                          - iqr: 四分位数法（默认，推荐）
                          - std: 3σ标准差法
    :return: 处理后的DataFrame
    """
    df_processed = df.copy()  # 避免修改原数据
    
    # 遍历每个需要处理的字段
    for col in cols:
        # 步骤1：识别异常值，获取上下边界
        if detect_method == 'iqr':
            q1 = df_processed[col].quantile(0.25)
            q3 = df_processed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - range * iqr
            upper_bound = q3 + range * iqr
        elif detect_method == 'std':
            mean = df_processed[col].mean()
            std = df_processed[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            raise ValueError("detect_method仅支持 'iqr' 或 'std'")
        
        # 步骤2：标记异常值
        outlier_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        
        # 步骤3：根据策略处理异常值
        if method == 'median_fill':
            median_val = df_processed[col].median()
            df_processed.loc[outlier_mask, col] = median_val
            print(f"字段【{col}】：用中位数{median_val:.2f}填充了{outlier_mask.sum()}个异常值")
        
        elif method == 'truncate':
            # 低于下限→下限，高于上限→上限
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            print(f"字段【{col}】：截断了{outlier_mask.sum()}个异常值（下限{lower_bound:.2f}，上限{upper_bound:.2f}）")
        
        elif method == 'drop':
            # 删除含异常值的行（注意：多字段时会删除任一字段有异常值的行）
            df_processed = df_processed[~outlier_mask]
            print(f"字段【{col}】：删除了{outlier_mask.sum()}个含异常值的行，剩余行数：{len(df_processed)}")
        
        else:
            raise ValueError("method仅支持 'median_fill' / 'truncate' / 'drop'")
    
    return df_processed


def to_2d_tuple_index(idx_1d,shape_of_2d):
    """
    将二维张量的一维索引（展平后的索引）转为二元组索引
    或把分开的行/列索引转为二元组索引
    
    Args:
        idx_1d: 可以是两种形式：
                1. 展平后的一维索引（如 tensor([2,5,7])）
                2. (行索引张量, 列索引张量) 元组
    
    Returns:
        二元组 (rows, cols)，可直接用于二维张量索引
    """
    # 情况1：输入是展平后的一维索引（比如 torch.tensor([3,5,8])）
    if isinstance(idx_1d, torch.Tensor) and idx_1d.dim() == 1:
        # 先获取二维张量的形状（这里需要你替换成自己的张量shape）
        # 假设你的二维张量是 arr，先执行 h, w = arr.shape
        # 这里先留空，你替换成实际形状即可
        h, w = shape_of_2d  # ！！！替换成你的二维张量形状，比如 4,5
        if h is None or w is None:
            raise ValueError("请先设置二维张量的形状 h, w = arr.shape")
        
        rows = idx_1d // w  # 计算行索引
        cols = idx_1d % w   # 计算列索引
        return (rows, cols)
    
    # 情况2：输入是分开的行/列索引（比如 (torch.tensor([0,1]), torch.tensor([2,3]))）
    elif isinstance(idx_1d, tuple) and len(idx_1d) == 2:
        rows, cols = idx_1d
        # 确保是张量格式，方便直接索引
        if not isinstance(rows, torch.Tensor):
            rows = torch.tensor(rows)
        if not isinstance(cols, torch.Tensor):
            cols = torch.tensor(cols)
        return (rows, cols)
    
    else:
        raise TypeError("输入格式错误！请传入一维索引张量 或 (行索引, 列索引) 元组")

def torch_intersect_indices(a, b):
    """
    兼容所有 PyTorch 版本
    求两个一维张量的交集 + 各自原索引
    返回：交集元素, a中的索引, b中的索引
    """
    a = a.flatten()
    b = b.flatten()

    # 找出在 a 里同时也在 b 里的元素 & 索引
    mask_a = torch.zeros_like(a, dtype=torch.bool)
    indices_a = []
    for i, val in enumerate(a):
        if (b == val).any():
            mask_a[i] = True
            indices_a.append(i)

    # 找出在 b 里同时也在 a 里的元素 & 索引
    mask_b = torch.zeros_like(b, dtype=torch.bool)
    indices_b = []
    for i, val in enumerate(b):
        if (a == val).any():
            mask_b[i] = True
            indices_b.append(i)

    # 交集元素（去重）
    intersect = torch.unique(a[mask_a]).to(a.device)

    # 转成 tensor
    indices_a = torch.tensor(indices_a).to(a.device)
    indices_b = torch.tensor(indices_b).to(a.device)

    return intersect, indices_a, indices_b
    
def tensor_intersect(t1, t2):
        indices = torch.zeros_like(t1, dtype = torch.bool, device = t1.device)
        for elem in t2:
            indices = indices | (t1 == elem)  
            intersection = t1[indices]  
        return intersection
    
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

def scale_value(value, src_range_min, src_range_max, dst_range_min, dst_range_max):
    """数值从源范围缩放到目标范围"""
    
    return (value - src_range_min) * (dst_range_max - dst_range_min) / (src_range_max - src_range_min) + dst_range_min

def normalization(data,res=1e-5,mode="numpy",avoid_zero=True,axis=0):
    if mode=="numpy":
        if len(data.shape)==1:
            sub = data - np.min(data) 
            max_min = np.max(data)-np.min(data) + res
            rtn = sub/max_min     
        else:   
            sub = data.transpose(1,0) - np.min(data,axis=axis)
            rtn = sub/(np.max(data,axis=axis)-np.min(data,axis=axis) + res) 
            rtn = rtn.transpose(1,0)
    else:
        if len(data.shape)==1:
            sub = data - torch.min(data) 
            max_min = torch.max(data)-torch.min(data) + res
            rtn = sub/max_min
        else:
            sub = data.transpose(1,0) - torch.min(data,dim=axis)[0]
            max_min = torch.max(data,dim=axis)[0] - torch.min(data,dim=axis)[0] + res
            rtn = sub/max_min
            rtn = rtn.transpose(1,0)
    if avoid_zero:
        rtn = rtn + res  
    return rtn

def normalization_axis(data,res=1e-5,avoid_zero=True,axis=0):
    if isinstance(data,torch.Tensor):
        sub = data - torch.unsqueeze(torch.min(data,dim=axis)[0],dim=axis)
        div = torch.unsqueeze((torch.max(data,axis=axis)[0]-torch.min(data,dim=axis)[0]),dim=axis)
        div[div==0] = res
        rtn = sub/div
    else:
        sub = data - np.expand_dims(np.min(data,axis=axis),axis=axis)
        div = np.expand_dims((np.max(data,axis=axis)-np.min(data,axis=axis)),axis=axis)
        div[div==0] = res
        rtn = sub/div        
    if avoid_zero:
        rtn = rtn + res  
    return rtn

def normalization_standard(x, dim=-1, eps=1e-8, weight=None, bias=None, clamp_range=None):
    """
    自定义标准化函数（支持加权均值、维度指定、值域截断）
    :param x: 输入张量 (batch_size, ..., n_features/n_candidates)
    :param dim: 计算均值/方差的维度（排序任务中设为-1，对每个样本的候选者维度标准化）
    :param eps: 避免除零的极小值
    :param weight: 可选的缩放权重（可学习，类似BatchNorm的gamma）
    :param bias: 可选的偏移权重（可学习，类似BatchNorm的beta）
    :param clamp_range: 标准化后的值域截断（如(0.01, 0.99)，避免极端值）
    :return: 标准化后的张量
    """
    # 1. 计算自定义均值（支持加权，无权重则为普通均值）
    if weight is not None:
        # 加权均值：适用于对重要特征/候选者赋予更高权重
        weighted_x = x * weight
        mu = torch.sum(weighted_x, dim=dim, keepdim=True) / torch.sum(weight, dim=dim, keepdim=True)
    else:
        # 普通均值（按指定维度）
        mu = torch.mean(x, dim=dim, keepdim=True)
    
    # 2. 计算自定义标准差（无偏方差）
    var = torch.var(x, dim=dim, keepdim=True, unbiased=True)
    sigma = torch.sqrt(var + eps)
    
    # 3. 标准化核心计算
    x_norm = (x - mu) / sigma
    
    # 4. 可选：缩放+偏移（模拟BatchNorm的可学习参数）
    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    
    # 5. 可选：值域截断（避免极端值，增强稳定性）
    if clamp_range is not None:
        x_norm = torch.clamp(x_norm, min=clamp_range[0], max=clamp_range[1])
    
    return x_norm

def normalization_except_outlier(x):
    """归一化并可以兼顾处理离群值"""
    
    rtn = (x - np.median(x,axis=0)) / (np.percentile(x, 75,axis=0) - np.percentile(x, 25,axis=0))
    return rtn

def interquartile_range(array,bound_ratio=1.2):
    p_low, p_up = np.percentile(array, 10), np.percentile(array, 90)
    # 取得上下区间范围数值
    bound = (p_up - p_low) * bound_ratio
    lower_bound, upper_bound = p_low - bound, p_up + bound
    
    # 对于超出范围的使用区间最大最小值代替，注意代替的时候需要加减随机数，避免多个相等的数值
    lower_index = np.where(array<lower_bound)[0]
    eps_lower = lower_bound * np.random.random(lower_index.shape[0])/10
    array[lower_index] = lower_bound - eps_lower
    
    upper_index = np.where(array>upper_bound)[0]
    eps_upper = upper_bound * np.random.random(upper_index.shape[0])/10
    array[upper_index] = lower_bound + eps_upper
        
    return array

def batch_normalization(data,res=1e-5):
    if isinstance(data, torch.Tensor):
        rtn = (data - torch.min(data))/(torch.max(data)-torch.min(data) + res) 
    else:
        rtn = (data - np.min(data))/(np.max(data)-np.min(data) + res)         
    return rtn + res

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
    price_arr_before = price_arr[:,:-1]
    price_arr_after = price_arr[:,1:]   
    slope_range = (price_arr_after - price_arr_before)/price_arr_before*100
    return slope_range

def compute_price_class(price_array,mode="max_range"):   
    cur_price = price_array[0]
    if mode=="max_range":
        max_value = np.max(price_array)
        min_value = np.min(price_array)
        if price_array[-1] - cur_price > 0:
            raise_range = (max_value - cur_price)/cur_price*100
        else:
            raise_range = (min_value - cur_price)/cur_price*100         
    if mode=="first_last":
        raise_range = (price_array[-1] - cur_price)/cur_price*100             
    if mode=="fast":
        raise_range = (price_array[3] - cur_price)/cur_price*100     
    if mode=="very_fast":
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

def adjude_seq_eps(seq_data:torch.Tensor,eps=1e-5):
    """调整序列值，避免序列中的所有值均相同"""
    
    result = []
    for i in range(seq_data.shape[0]):
        item = seq_data[i]
        if torch.unique(item).shape[0]==1:
            item[0] = item[0] + eps
        result.append(item)
    
    return torch.stack(result)    

def apply_along_axis(function, axis, x,y):
    return torch.stack([
        function(x_i,y) for x_i in torch.unbind(x, dim=axis)
        ], dim=axis)
    
def pairwise_compare(m,n,distance_func=None):
    """根据自定义距离函数，进行m比n"""
    
    result_list = []
    index = 1
    for item in m:
        # 把单条数据复制为和目标同样形状，进行批量比较
        item_metric = item.unsqueeze(0)
        item_metric = item_metric.repeat(n.shape[0],1)
        v = distance_func(item_metric,n)
        result_list.append(v)
        # print("apply:",index)
        index+=1
    return torch.stack(result_list).squeeze(-1)

def pairwise_distances(metirx,distance_func=None,make_symmetric=False,reduction="mean"):
    """根据自定义距离函数，生成配对距离矩阵"""
    
    result_list = []
    index = 1
    size = metirx.shape[0]
    for i in range(size):
        # 如果超过2维，则分别计算
        if len(metirx.shape)==3:
            v_array = []
            for j in range(metirx.shape[2]):
                metirx_t = torch.cat([metirx[i:,:,j],metirx[:i,:,j]],dim=0)
                v = distance_func(metirx[:,:,j],metirx_t)
                v_array.append(v)
            v_array = torch.stack(v_array)
            if reduction=="mean":
                v = torch.mean(v_array,dim=0)
            if reduction=="max":
                v = torch.max(v_array,dim=0)[0]          
            if reduction=="min":
                v = torch.min(v_array,dim=0)[0]                     
            result_list.append(v)
        else:       
            # 滚动比较,忽略自比较数据
            metirx_t = torch.cat([metirx[i:,:],metirx[:i,:]],dim=0)
            v = distance_func(metirx,metirx_t)
            result_list.append(v)
            # if index%100==0:
            #     print("apply:",index)
            index+=1
    dis_met = torch.stack(result_list)
    result_list = []
    # 构造为对角矩阵
    for i in range(size):
        roll_vector = torch.cat([dis_met[size-i:,i],dis_met[:size-i,i]],dim=0)
        result_list.append(roll_vector)
    dis_met = torch.stack(result_list)
    # 对角线置零
    dis_met[dis_met<1e-6] = 0
    dis_met = torch.round(dis_met,decimals=5)
    # 根据配置，决定个是否进行对称拷贝
    if make_symmetric:
        # 首先转换成上三角形，然后向下拷贝
        dis_met = torch.triu(dis_met)
        dis_met += dis_met.T - torch.diag(dis_met.diagonal())
    return dis_met

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def intersect2d(A,B):
    ret = []
    for i in range(A.shape[0]):
        res = np.intersect1d(A[i],B[i])
        ret.append(res)
    return np.array(ret)

def build_symmetric_adj(arr,distance_func=None,device=None):
    """根据原始数据，生成symmetric邻接矩阵以及拉普拉斯矩阵"""
    
    # 使用配对比较方式，生成距离矩阵
    if not isinstance(arr, torch.Tensor):
        arr = torch.Tensor(arr).to(device)
    adj_matrix = pairwise_distances(arr,distance_func=distance_func)
    # 转换为稀疏矩阵
    adj_matrix = csr_matrix(adj_matrix.cpu().numpy())
    # 对称性变换
    adj_matrix = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)
    # 与度矩阵结合
    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
    # 归一化并生成拉普拉斯矩阵
    adj_matrix = matirx_normalize(adj_matrix)
    adj_matrix = sparse_mx_to_torch_sparse_tensor(adj_matrix)
    return adj_matrix

def matirx_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def ccc_distance_torch(x,y):
    flag_numpy = 0
    if isinstance(x,np.ndarray):
        flag_numpy = 1
        x = torch.Tensor(x)
        y = torch.Tensor(y)
    if len(x.shape)>1:
        x = x.transpose(1,0)
        y = y.transpose(1,0)
        concordance = ConcordanceCorrCoef(num_outputs=x.shape[-1])
    else:
        concordance = ConcordanceCorrCoef()
    dis = 1 - concordance(x, y)
    if flag_numpy==1:
        dis = dis.cpu().numpy()
    return dis 
    
def batch_cov(points):
    points = points.permute(0,2,1)
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)

def batch_cov_comp(x,y):
    """实现对给定2个变量的协方差矩阵的计算，变量shape为：(batch_size,样本数,样本时间长度)"""
    
    # 在最后一个维度合并,并计算
    points = torch.concat((x,y),dim=-1)
    bcov = batch_cov(points)
    return bcov 

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def corr_compute(source,target):
    corr_tensor = torch.concat([source,target],dim=0)
    corr = torch.corrcoef(corr_tensor)
    corr_real = corr[source.shape[0]:,:source.shape[0]]
    return corr_real

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.sum((np.expand_dims(array,1) - value)**2,axis=2).argmin(axis=0)
    return idx

def eps_rebuild(data):
    """Eps for Zero data"""
    
    if isinstance(data,np.ndarray):
        eps_ori = np.random.uniform(low=1e-4,high=1e-3,size=data.shape)
        data = np.where(data==0,eps_ori,data)
    else:
        eps_ori = torch.ones(data.shape).uniform_(1e-4, 1e-3).to(data.device)
        data = torch.where(data==0,eps_ori,data)      
    return data

def same_value_eps(data):
    """Eps for Same Value problem"""
    
    eps = 1e-4
    if isinstance(data,np.ndarray):
        for i in range(data.shape[0]):
            eps_adju = np.random.uniform(low=eps,high=eps*10,size=data.shape[1])
            item = data[i]
            if np.unique(item).shape[0]==1:
                data[i] = data[i] + eps_adju
    else:
        eps_ori = torch.ones(data.shape).uniform_(1e-4, 1e-3).to(data.device)
        data_match = data - data[0]
        data = torch.where(data_match==0,eps_ori,data)      
    return data

def get_trunck_index(total_size,batch_size):
    return total_size//batch_size


def compute_average_precision(candidate,target,topk=10):
    
    top_cls = np.argsort(-candidate)[:topk]
    top_ret = []
    target_sort = np.argsort(-target)[:topk]
    for i in range(topk):
        if i>=top_cls.shape[0]:
            continue
        result = np.where(target_sort==top_cls[i])[0]
        if len(result)>0:
            top_ret.append(result[0])    
    top_ret = np.sort(np.array(top_ret))
    top_ret = top_ret + 1
    total_score = 0
    for i in range(1,top_ret.shape[0]+1):
        total_score += i/top_ret[i-1]
    return total_score

def softmax(x):
    """ Softmax function """
    
    x -= np.max(x, axis = 1, keepdims = True) 
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

def slope_angle(lines):
    """取得斜率角度正弦值"""
    
def round_to_tick(value, tick_size):
    """实现最小变动单位"""
    
    return round(value / tick_size) * tick_size

def scale_multiple_series(data):
    
    result = []
    for ser in data:
        ser_scale = MinMaxScaler().fit_transform(np.expand_dims(ser,-1)).squeeze(-1)
        result.append(ser_scale)
        
    result = np.stack(result)
    result_main = np.mean(result,0)
    result_main = MinMaxScaler().fit_transform(np.expand_dims(result_main,-1)).squeeze(-1)
    return result,result_main

def all_elements_same(tensor, eps=1e-5):
    """
    判断张量内所有元素是否为同一个值
    :param tensor: 输入PyTorch张量
    :param eps: 浮点数精度容错（整数设为0）
    :return: bool，True表示所有元素相同
    """
    if tensor.numel() == 1:  # 只有1个元素，默认相同
        return True
    
    # 取第一个元素作为基准
    base = tensor.flatten()[0]
    
    # 浮点数用绝对值差判断，整数直接相等判断
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        return torch.all(torch.abs(tensor - base) < eps).item()
    else:
        return torch.all(tensor == base).item()

def check_nan(tensor, name="tensor"):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f" {name} has NaN)")
        print(f"  mean={tensor.mean().item():.4f}, max={tensor.max().item():.4f}, min={tensor.min().item():.4f}")   
        return True
    return False

def check_param_nan(params):
    for key in params:
        param = params[key]
        try:
            if param is not None and torch.any(torch.isnan(param.data)):
                return key
        except Exception as e:
            print("eee:{}".format(e))
    return None
 
def is_same_elements(tensor1, tensor2,eps=1e-5):
    diff_num = torch.sum(torch.abs(tensor1 - tensor2)>eps)
    if diff_num>0:
        return False
    return True
    
    
def linear_map(arr, new_min, new_max):
    """线性映射到新范围"""
    
    arr_min, arr_max = arr.min(), arr.max()
    # 避免除零错误
    if arr_max - arr_min == 0:
        return np.full_like(arr, (new_min + new_max) / 2)
    # 线性映射公式
    return (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min

def row_topk_coords(matrix, k):
    """
    返回每行中最大的 k 个元素的坐标 (行数*k, 2)
    """
    values, col_indices = torch.topk(matrix, k, dim=1)  # col_indices shape: (行数, k)
    rows = torch.arange(matrix.size(0)).unsqueeze(1).expand_as(col_indices).to(matrix.device)  # 每行重复 k 次
    coords = torch.stack([rows, col_indices], dim=-1)  # shape: (行数, k, 2)
    coords = coords.view(-1, 2)  # 展平为 (行数*k, 2)
    return coords,col_indices,values

def min_max_norm(data,range=[0,1]):
    """根据原数值和范围，计算归一化数值"""
    
    min_value,max_value = range
    if data>max_value:
        max_value = data
    if data<min_value:
        min_value = data        
    mean_data = (data-min_value)/(max_value-min_value)
    return mean_data

def min_max_norm_reverse(data,range=[0,1]):
    """根据归一化数值和归一化范围，反向计算出原数值"""
    
    min_value,max_value = range
    if data>max_value:
        max_value = data
    if data<min_value:
        min_value = data        
    ori_data = data * (max_value-min_value) + min_value
    return ori_data

def build_random_mul_data(size,mul_range=[0,1]):
    """生成指定范围内的随机乘数数组"""
    
    low,high = mul_range
    rand_tensor = low + (high - low) * torch.rand(size)
    return rand_tensor

if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 2)
    target = torch.randn(2, 2)
    # mae_comp(input,target)
    # find_nearest(input.numpy(),target.numpy())
    print(round_to_tick(3128,5))
    
       
    