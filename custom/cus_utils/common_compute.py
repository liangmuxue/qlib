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
        data = torch.where(data==0,eps_ori,data)      
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

if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 2)
    target = torch.randn(2, 2)
    # mae_comp(input,target)
    find_nearest(input.numpy(),target.numpy())
    
       
    