import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

### Functions for correlating matrix to a column
# O - (n,t) array of observations: n traces with t samples each
# P - column of n predictions

# initial version, copied from my Matlab code
def ColumnWiseCorrcoef(O, P):
    n = P.size
    DO = O - (np.sum(O, 0) / np.double(n))
    DP = P - (np.sum(P) / np.double(n))
    return np.dot(DP, DO) / np.sqrt(np.sum(DO ** 2, 0) * np.sum(DP ** 2))

# the slow naive version using the built-in function
def ColumnWiseCorrcoefNaive(O, P):
    return np.corrcoef(P,O.T)[0,1:O[0].size+1]

# improvement over the initial one from Daniel at stackoverflow.com
# note that it modifies P  (however, the gain in performance from it appears to be insignificant)
def newColumnWiseCorrcoef(O, P):
    n = P.size
    DO = O - (np.einsum('ij->j',O) / np.double(n))
    P -= (np.einsum('i->',P) / np.double(n))
    tmp = np.einsum('ij,ij->j',DO,DO)
    tmp *= np.einsum('i,i->',P,P)          #Dot or vdot doesnt really change much.
    return np.dot(P, DO) / np.sqrt(tmp)


### Functions for correlating matrix to a matrix
# O - (n,t) array of observations: n traces with t samples each
# P - (n,m) array of n predictions for each of the m candidates
# C - (optional) pre-allocated (m,t) array for correlation traces of length t for each of the m candidates

# Naively using an outer loop with the function from above, as a reference for comparing performance
def loopedNewColumnWiseCorrcoef(O, P, C):
    for i in range(0,256):
        C[i] = newColumnWiseCorrcoef(O, P[:,i])

# this one has the naive loop over columns of P internally
def AlmightyCorrcoefNaive(O, P, C):
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.sum(O, 0) / np.double(n)) # compute O - mean(O); note that mean(O) will be appleid row-wise to O
    DP = P - (np.sum(P, 0) / np.double(n)) # compute P - mean(P)

    for i in np.arange(0, m):
        tmp = np.sum(DO ** 2, 0)
        tmp *= np.sum(DP[:,i] ** 2)
        C[:,i] = np.dot(DP[:,i], DO) / np.sqrt(tmp)

# here the loop is avoided by matrix operations
# returns (m,t) correaltion matrix of m traces t samples each
def AlmightyCorrcoef(O, P):
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.sum(O, 0) / np.double(n)) # compute O - mean(O)
    DP = P - (np.sum(P, 0) / np.double(n)) # compute P - mean(P)
    # note that mean row will be appleid row-wise to original matrices

    cov = np.einsum("nt,nm->tm", DO, DP)

    varO = np.sum(DO ** 2, 0)
    varP = np.sum(DP ** 2, 0)
    tmp = np.outer(varO, varP)

    return cov / np.sqrt(tmp)

# Here the einsum is applied to speed up the computations
# O - (n,t) array of n traces with t samples each
# P - (n,m) array of n predictions for each of the m candidates
# returns (m,t) correaltion matrix of m traces t samples each
def AlmightyCorrcoefEinsum(O, P):
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.einsum("nt->t", O) / np.double(n)) # compute O - mean(O)
    DP = P - (np.einsum("nm->m", P) / np.double(n)) # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO)

    varP = np.einsum("nm,nm->m", DP, DP)
    varO = np.einsum("nt,nt->t", DO, DO)
    tmp = np.einsum("m,t->mt", varP, varO)

    return cov / np.sqrt(tmp)

# same, but with einsum optimization
def AlmightyCorrcoefEinsumOptimized(O, P):
    (n, t) = O.shape      # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (np.einsum("nt->t", O, optimize='optimal') / np.double(n)) # compute O - mean(O)
    DP = P - (np.einsum("nm->m", P, optimize='optimal') / np.double(n)) # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize='optimal')

    varP = np.einsum("nm,nm->m", DP, DP, optimize='optimal')
    varO = np.einsum("nt,nt->t", DO, DO, optimize='optimal')
    tmp = np.einsum("m,t->mt", varP, varO, optimize='optimal')

    return cov / np.sqrt(tmp)

# check computation correctness
def testCorrectness():
    
    O = np.random.rand(int(1E3), int(1E2))
    P = np.random.rand(int(1E3), 256)

    C = AlmightyCorrcoefEinsumOptimized(O,P)
    firstRow = ColumnWiseCorrcoef(O,P[:, 0])
    secondRow = ColumnWiseCorrcoef(O,P[:,1])

    firstRowOk = np.allclose(C[0], firstRow)
    secondRowOk = np.allclose(C[1], secondRow)

    if firstRowOk and secondRowOk:
        print("Test passed")
    else:
        print("Test failed")

def testCorrectnessBis():
    
    O = np.random.rand(int(1E3), int(1E2))
    P = np.random.rand(int(1E3), 256)
    C = np.zeros((256, int(1E2)))

    loopedNewColumnWiseCorrcoef(O, P, C)
    Z = AlmightyCorrcoefEinsumOptimized(O,P)

    if np.allclose(C,Z):
        print("Test passed")
    else:
        print("Test failed")

def analyze_model_complexity(model, input_shape, threshold=1e6):
    """
    分析模型复杂度
    """
    import torch
    from torchsummary import summary
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 粗略估计计算量（FLOPs）
    # 这里简化处理，实际可以使用thop等库
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    issues = []
    
    # 检查参数量是否过多
    if total_params > threshold:
        issues.append(f"模型参数量过多: {total_params:,} > {threshold:,}")
    
    # 检查层数
    num_layers = len(list(model.children()))
    if num_layers > 50:  # 假设阈值
        issues.append(f"模型层数过多: {num_layers}层")
    
    # 检查过大的层
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.numel() > 1e6:  # 单个参数超过100万
                issues.append(f"层 {name} 参数过多: {param.numel():,}")
    
    return issues, total_params

def calculate_comprehensive_rank_from_scores(
    scores_df, 
    weights=None, 
    normalization_method='minmax',
    direction=None,
    handle_missing='mean'
):
    """
    通过单项得分计算综合排名
    
    参数：
    ----------
    scores_df : pandas DataFrame
        包含各单项得分的DataFrame，每行是一个样本，每列是一个指标
    weights : list or dict, optional
        权重列表或字典。如果是列表，长度必须与列数相同；
        如果是字典，键必须是列名。默认为等权重
    normalization_method : str, optional
        标准化方法，可选：
        - 'minmax': Min-Max归一化 (0-1)
        - 'zscore': Z-Score标准化
        - 'none': 不进行标准化
    direction : dict, optional
        指标方向字典，键为列名，值为'positive'或'negative'
        正向指标：值越大越好
        负向指标：值越小越好
    handle_missing : str, optional
        缺失值处理方法，可选：
        - 'mean': 用列均值填充
        - 'median': 用列中位数填充
        - 'drop': 删除有缺失值的行
    
    返回：
    -------
    result_df : pandas DataFrame
        包含综合得分和排名的DataFrame
    normalized_scores : pandas DataFrame
        标准化后的得分
    """
    
    # 创建副本避免修改原始数据
    df = scores_df.copy()
    
    # 1. 处理缺失值
    if df.isnull().any().any():
        if handle_missing == 'mean':
            df = df.fillna(df.mean())
        elif handle_missing == 'median':
            df = df.fillna(df.median())
        elif handle_missing == 'drop':
            df = df.dropna()
        else:
            raise ValueError("handle_missing参数必须是'mean', 'median'或'drop'")
    
    # 2. 标准化处理
    normalized_df = df.copy()
    
    if normalization_method == 'minmax':
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_values, 
                                     index=df.index, 
                                     columns=df.columns)
        
    elif normalization_method == 'zscore':
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(df)
        normalized_df = pd.DataFrame(normalized_values, 
                                     index=df.index, 
                                     columns=df.columns)
        
    elif normalization_method == 'none':
        # 不进行标准化，但确保所有指标为正向
        pass
    else:
        raise ValueError("normalization_method必须是'minmax', 'zscore'或'none'")
    
    # 3. 处理指标方向
    if direction is not None:
        for col, dir_type in direction.items():
            if col in normalized_df.columns:
                if dir_type == 'negative':
                    # 对于负向指标，反转标准化值
                    normalized_df[col] = 1 - normalized_df[col] if normalization_method == 'minmax' else -normalized_df[col]
    
    # 4. 设置权重
    if weights is None:
        # 等权重
        weights_array = np.ones(len(df.columns)) / len(df.columns)
    elif isinstance(weights, dict):
        # 字典权重
        weights_array = np.array([weights.get(col, 0) for col in df.columns])
        # 归一化权重
        weights_array = weights_array / weights_array.sum()
    elif isinstance(weights, list):
        # 列表权重
        if len(weights) != len(df.columns):
            raise ValueError("权重列表长度必须与指标数相同")
        weights_array = np.array(weights)
        # 归一化权重
        weights_array = weights_array / weights_array.sum()
    else:
        raise TypeError("权重必须是list、dict或None")
    
    # 5. 计算综合得分
    # 确保数据类型一致
    normalized_values = normalized_df.values.astype(float)
    comprehensive_scores = np.dot(normalized_values, weights_array)
    
    # 6. 计算排名（得分越高，排名越靠前）
    # rankdata默认是值越小排名越靠前，所以我们用负值
    comprehensive_ranks = rankdata(-comprehensive_scores, method='min')
    
    # 7. 创建结果DataFrame
    result_df = pd.DataFrame({
        '样本名称': df.index.tolist(),
        '综合得分': comprehensive_scores,
        '综合排名': comprehensive_ranks
    })
    
    # 添加权重信息
    for i, col in enumerate(df.columns):
        result_df[f'权重_{col}'] = weights_array[i]
    
    # 按排名排序
    result_df = result_df.sort_values('综合排名').reset_index(drop=True)
    
    return result_df, normalized_df

if __name__ == '__main__':

    import timeit
    import sys

    # system information
    print("Python: " + sys.version)
    print("Numpy : " + np.version.version)

    a = np.random.random((100, 1000))
    b = a # np.random.random((100, 1000))
    corrmat = np.zeros((a.shape[1], b.shape[1]))
    corrmat = AlmightyCorrcoefEinsumOptimized(b, a)
    
    print(corrmat.shape)
    
    
    
