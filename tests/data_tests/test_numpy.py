import pandas as pd
import sys, os
import numpy as np
import torch
from torchvision.transforms import *
import cv2
from torch import nn
from sklearn.metrics import precision_recall_curve
import datetime

def test_nozero():
    L = np.arange(18).reshape((2, 3, 3))
    L[:,:, 1] = 0; L[:, [0, 1],:] = 0
    print("l is:", L)
    t = (L != 0).any(axis=(0, 2))
    print("t is:", t)
    a = L[:, t,:]
    print("a is:", a)


def test_argwhere():
    Z = np.zeros((6, 6, 3), dtype=int)
    s0 = np.array([1, 0, 1])
    s1 = np.array([0, 1, 0])
    Z[1, 2] = s0
    Z[1, 5, 1] = 1
    Z[1, 5, 2] = 1
    Z[4, 5] = s1
    print("Z is:", Z)
    # z1 = Z[:,:,0]
    # print("z1:",z1)
    w = np.argwhere(Z) 
    print("w is:", w)
    S = np.row_stack((s0, s1))
    out = np.where((np.in1d(Z, S).reshape(Z.shape)).all(2))
    # print("S is:",S)
    print("out is:", out)
    
def test_copy():    
    ori_data = np.ones([1920,1080,3])
    for i in range(100):
        dt = datetime.datetime.now() 
        print("copy begin:{},shape:{}".format(dt,ori_data.shape))
        tar_data = np.copy(ori_data)
        rel_dt = datetime.datetime.now() - dt
        print("copy end:{},shape:{}".format(rel_dt,tar_data.shape))

def test_copy_torch():    
    ori_data = np.ones([1920,1080,3])
    for i in range(100):
        dt = datetime.datetime.now() 
        print("copy begin:{},shape:{}".format(dt,ori_data.shape))
        torch_data = torch.from_numpy(ori_data)
        tar_data = torch_data.clone().numpy()
        rel_dt = datetime.datetime.now() - dt
        print("copy end:{},shape:{}".format(rel_dt,tar_data.shape))
            
def test_condition():
    Z = np.zeros((6, 6, 3), dtype=int)
    s0 = np.array([1, 0, 18])
    s1 = np.array([0, 1, 0])
    Z[1, 2] = s0
    Z[1, 5, 1] = 18
    Z[1, 5, 2] = 18
    Z[4, 5] = s1
    print("Z is:", Z)
    ind = np.any(Z > 0, axis=1)
    print("ind:", ind)
    ind = np.where(Z[:,:, 2] == 18)
    print("ind:", ind)      


def test_condition2():
    x = np.array([[1, 0, 18], [2, 3, 4], [3, 4, 5]])
    mask = x[:, 2] == 5
    print("mask:", mask)
    print(x[mask,:])

def test_condition3():
    s0 = np.array([[[1, 0, 18],[255,255,255]],[[1, 0, 18],[249,255,255]]])
    print("s0 shape:", s0.shape)
    cond = np.all(s0 > 230, axis=2)
    print("cond:", cond)
    ind = s0[cond]
    print("ind:", ind)
    

def test_variable():
    Z = np.zeros((6, 6, 3), dtype=int)
    s0 = np.array([[1, 0, 18], [2, 3, 7], [11, 10, 118], [12, 13, 17], [21, 20, 128], [22, 23, 27]])
    s1 = np.array([0, 1, 0])
    Z[:, 1,:] = s0
    Z[4, 5] = s1
    print("Z is:", Z)   


def test_index():
    x = np.array([[1, 0, 18], [2, 3, 7]])
    y = x[:, 0:2]
    # print("x is:", x)   
    # print("y is:", y)  
    
    list = ['MA5','QTLU5','CCI5','OBV5','CNTD5']  
    print(list[3:5])

    
def test_index2(): 
    Z = np.zeros((6, 6, 3), dtype=int)
    s0 = np.array([[1, 0, 18], [2, 3, 7], [11, 10, 118], [12, 13, 17], [21, 20, 128], [22, 23, 27]])
    Z[:, 1,:] = s0
    idx = np.array([[1, 1], [0, 0], [1, 0], [0, 0], [1, 0], [1, 1]])
    # z1 = Z[:,idx]
    z1 = np.take(Z, idx)
    print("Z is:", Z)   
    print("z1 is:", z1)    

    
def test_index3(): 
    np.random.seed(77)
    data = np.random.randint(low=0, high=10, size=(5, 7, 3))
    indices = np.array([[0, 1, 6, 4, 5], [1, 1, 2, 0, 1]])
    t = np.arange(data.shape[0])
    out = data[t, indices[0], indices[1]]
    print("data is:", data)   
    print("out is:", out)   

       
def test_condi_remove():
    nparray = np.arange(40).reshape((8, 5))
    print("Given numpy array:\n", nparray)    
    wh = np.where((nparray >= 5) & (nparray <= 20))
    nparray1 = np.delete(nparray, wh[0], axis=0)
    print("After deletion of rows containing numbers between 5 and 20: \n", nparray1)
    nparray2 = np.delete(nparray, np.where((nparray[:, 0] >= 25) & (nparray[:, 0] <= 35))[0], axis=0)
    print("After deletion of rows whose first element is between 25 and 35:\n", nparray2)

    
def test_group():
    x = np.array([[ 569 , 897],
     [ 570 , 898],
     [ 570 , 900],
     [ 571 , 901],
     [ 571 , 905],
     [ 572  , 906]])
    # t = np.flatnonzero(x[1:,0] != x[:-1,0])+1
    # print(t)
    # out = np.split(x, t,axis=0)
    # print("out:",out)
    t2 = np.unique(x[:, 0], return_counts=True)
    print(t2)
    t2 = t2[1][1:]
    print(t2)
    out2 = np.split(x, t2, axis=0)
    print("out2:", out2)
    
    def max_len(list):
        longest_list = max(len(elem) for elem in list)
        return longest_list
    
    out_max_len = max_len(out2)
    print("out_max_len:", out_max_len)
    # out_con = np.concatenate(out2, axis=1)
    # print("out_con:",out_con)
    res = []
    for item in out2:
        re = np.pad(item, ((0, out_max_len - item.shape[0]), (0, 0)), 'constant', constant_values=(0))
        res.append(re)
    nres = np.array(res)
    print(nres)

    
def test_each():
    s = np.array([[-1.56113514, 4.51759732],
       [-2.80261623, 5.068371  ],
       [ 0.7792729 , 6.0169462 ],
       [-1.35672858, 3.52517478],
       [-1.92074891, 5.79966161],
       [-2.79340321, 4.73430001],
       [-2.79655868, 5.05361163],
       [-2.13637747, 5.39255837],
       [ 0.17341809, 3.60918261],
       [-1.22712921, 4.95327158]])
    v = s[:, 0] ** 2
    out = np.exp((-v / 200) - 0.5 * (s[:, 1] + 0.05 * v - 5) ** 2)
    print(out)
    v = s[:, 0]
    out = np.subtract(out,v)
    print(out)

def test_split_compute():
    arr = np.array([
        [[569 , 897],
         [0, 0]],
    
         [[570 , 898],
          [570 , 900]],
        
         [[571, 901],
          [571 , 905]],
        
         [[572 , 906],
          [  0 , 0]]
    ])
    length = arr.shape[0]
    for i in range(2):
        for j in range(i + 1, 2):
            du1 = (arr[:, i, 0] - arr[:, j, 0]) ** 2
            sum = (arr[:, i, 0] - arr[:, j, 0]) ** 2 + (arr[:, i, 1] - arr[:, j, 1]) ** 2
            sq = np.sqrt(sum)     
    
            if sq < 6:
                print("lll")

                
def test_temp():
    # arr = [[(130, 37, 0.9547701, 0)], [(106, 41, 0.960853, 1), (130, 43, 0.9493217, 2)], [(110, 41, 0.9765356, 3), (124, 43, 0.87878436, 4)], [(114, 48, 0.9471407, 5), (121, 51, 1.0362356, 6)], [(110, 49, 0.21988666, 7), (120, 59, 0.9178307, 8)], [(101, 40, 0.9474892, 9), (135, 44, 0.9179209, 10)], [(98, 47, 0.7709754, 11), (138, 52, 0.8963235, 12)], [(97, 49, 0.23131275, 13), (139, 60, 0.822693, 14), (152, 0, 0.118390195, 15)], [(109, 55, 0.9592643, 16), (125, 61, 0.92577446, 17)], [(109, 66, 0.92471373, 18), (124, 73, 0.98474723, 19), (159, 79, 0.12637408, 20)], [(108, 77, 0.92193204, 21), (125, 82, 0.8574085, 22), (182, 120, 0.10133813, 23), (192, 121, 0.10296779, 24)], [(102, 55, 0.8907591, 25), (132, 61, 0.8883806, 26)], [(103, 65, 0.83924854, 27), (131, 73, 0.9766074, 28), (158, 80, 0.19179878, 29)], [(105, 73, 0.7987924, 30), (131, 84, 0.9129152, 31), (158, 96, 0.12822857, 32), (159, 87, 0.11083252, 33), (192, 123, 0.11102441, 34)], [(129, 36, 0.9734128, 35)], [(132, 36, 0.94683576, 36)], [(109, 35, 0.9049975, 37), (127, 37, 0.88041323, 38)], [(105, 35, 0.7279416, 39), (133, 37, 0.68659896, 40)]]               
    # for index,item in enumerate(arr):
    #     if len(item)>2:
    #         print("index:{},data:{}".format(index,item))
    print(np.random.randint(0,2))
        
        
def test_pd_ser():
    
    dict_data1 = {
        "Beijing":1000,
        "Shanghai":2000,
        "Shenzhen":500
    }
    
    data1 = pd.Series(dict_data1)
    data1.name = "City Data"
    data1.index.name = "City Name"
    print(data1)
 
def test_pd_df():
    index_arrays = [[1, 1, 2, 2], ['男', '女', '男', '女']]
    columns_arrays = [['2020', '2020', '2021', '2021'],
                      ['上半年', '下半年', '上半年', '下半年',]]
    index = pd.MultiIndex.from_arrays(index_arrays,
                                      names=('班级', '性别'))
    print("index:",index)
    columns = pd.MultiIndex.from_arrays(columns_arrays,
                                        names=('年份', '学期'))
    print("columns:",columns)
    df = pd.DataFrame([(88,99,88,99),(77,88,97,98),
                       (67,89,54,78),(34,67,89,54)],
                      columns=columns, index=index)    
    print("df:",df)    
 
def test_pd_trans():
    np.random.seed(2015)
    
    df = pd.DataFrame({'account': ['foo', 'bar', 'baz'] * 3,
                       'val': np.random.choice([np.nan, 1],size=9)})
    print (df)
    
    df['val1'] = df.groupby('account')['val'].transform('last')
    df['val2'] = df.groupby('account')['val'].transform('nth', -1)
    print (df)       
   
def test_pd_index():
    # Creating index for multi-index dataframe
    tuples = [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
    index = pd.MultiIndex.from_tuples(tuples)
    # Value corresponding to the index
    data = [2, 4, 6, 8]
    # Creating dataframe using 'data' and 'index'
    df = pd.DataFrame(data = data, index = index, columns = ['value'])
    print(df)
    reset_df = df.reset_index()
    print(reset_df)

                  
def test_norm():
 
    a = np.array([[0.1,0.4,0.3],[0.2,0.5,0.6]]).transpose(1,0)
    rtn = (a-np.min(a,axis=0))/(np.max(a,axis=0)-np.min(a,axis=0)) 
    rtn = rtn + 0.001
    print(rtn)
   
def test_torch_vision():
    test_data = cv2.imread("/home/bavon/face_test/capture_lmx_gfc3.jpg")
    # test_data = np.ones((256,456,3), np.float32)
    # test_data = np.array([[[0.5,0.5,0.5]]])
    transform_te_t = Compose(
         [ToTensor()]
    )
    # temp_data =  transform_te_t(test_data)  
    # print("temp_data",temp_data)
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_te = Compose(
         [ToTensor(), normalize]
    )
    targ_data =  transform_te(test_data)     
    print("targ_data",targ_data)
 
def test_np_norm():
    test_data = cv2.imread("/home/bavon/face_test/capture_lmx_gfc3.jpg")
    test_data = test_data/255
    # print("test_data",test_data)
    # test_data = np.array([[[0.5,0.5,0.5]]])
    # test_data = np.ones((256,456,3), np.float32)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    targ_data =  test_data - mean
    targ_data =  targ_data /std
    targ_data= targ_data.transpose(2,0,1)
    print("targ_data",targ_data)
 
def np_comp():
    a = np.array([0.4, 8, 0.1])
    c = np.array([0.5, 9, 0.05])
    max_value = np.maximum(a, c)
    # print("clamp",np.clip(a, 3, 9))  
    print("max is:",max_value)
         
    
def torch_comp():
    a = torch.tensor([0.4, 8, 0.1])
    b = torch.clamp(a, 3, 9)
    c = torch.tensor([0.5, 9, 0.05])
    max_value = torch.max(a, c)
    # print("clamp",b)
    print("max:",max_value)

def test_concat():
    a = torch.tensor([[0.4, 8, 0.1],[0.5, 9, 0.2]])
    b = [torch.cat(item) for item in a]
  
def test_tensor_sum():
    a = torch.tensor([[0.4, 8, 0.1],[0.5, 9, 0.2]])
    b = a.sum(-1)
    print(b)
    
def test_linear():
    v = torch.ones(98, 57).cuda()
    lin = nn.Linear(57, 6)   
    out = lin(v)     
    print(out)
    
def test_inrange():   
    scope = 5.5 
    CLASS_VALUES={"0":[10,10000],"1":[5,10],"2":[0,5],"3":[-5,0],"4":[-10,-5],"5":[-10000,-10]} 
    if scope in range(5,10):
        print("in")
    p = [k for k, v in CLASS_VALUES.items() if (scope>=v[0] and scope<v[1])]
    print(p)
    
def test_tensor_list():    
    arr = ["2"]
    tensor=torch.Tensor(list(map(int,arr)))
    print(tensor)

def test_reshape():
    A = np.reshape(np.arange(24),(4,3,2)) ##��������������0��23����������4,3,2��������
    print(A.shape)
    B = np.reshape(A,(-1,2))
    print(B.shape)

def test_meshgrid():
    xy = np.mgrid[-5:5.1:0.5, -5:5.1:0.5].reshape(2,-1).T
    X, Y = np.mgrid[-5:5:21j, -5:5:21j]
    print(xy)   

def test_linespace_2d():
    len = 3
    array_ori = np.array([3,100,300])
    arr_2d = [np.arange(item,item+len,dtype=np.int32) for item in array_ori]
    np_arr_2d = np.array(arr_2d)
    # arr = np.linspace(0, 10,10,dtype=np.int32)
    print(arr_2d)

def test_corr():
    x_simple = np.array([-1.8430, -1.9625, -2.1838, -1.8075, -1.8022])
    y_simple = np.array([0.2252, 0.2269, 0.2403, 0.2496, 0.2595])
    t = np.stack((x_simple,y_simple))
    my_rho = np.corrcoef(t)
    # print(cov) 
    print(my_rho)      
    
def test_corr_tensor():
    from torchmetrics import PearsonCorrCoef
    target = torch.tensor([[0.2205, 0.2276, 0.2315, 0.2313, 0.2374],
        [0.1232, 0.1220, 0.1231, 0.1224, 0.1234],
        [0.0429, 0.0427, 0.0443, 0.0467, 0.0480],
        [0.0213, 0.0192, 0.0168, 0.0155, 0.0133]])
    target = torch.tensor([[-1.8430, -1.9625, -2.1838, -1.8075, -1.8022]])
    preds = torch.tensor([[ 0.9404,  0.9669,  0.8062,  0.8755,  0.7477],
        [ 0.2350, -1.0773, -0.7159, -0.6425, -0.7188],
        [-1.1237, -1.2764, -1.3003, -1.1781, -1.6030],
        [-1.0591, -1.4384, -1.2628, -2.0782, -1.6555]])
    preds = torch.tensor([[0.2252, 0.2269, 0.2403, 0.2496, 0.2595]])
    pearson = PearsonCorrCoef(num_outputs=5)
    ret = pearson(preds, target)
    print(ret)

def test_slope():
    import numpy as np
    x = [1, 3]
    y = [1,0.5]
    slope, intercept = np.polyfit(x,y,1)
    print(slope)

def concordance_correlation_coefficient(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

def concordance_correlation_coefficient_torch(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    
    corr_tensor = torch.stack((y_pred,y_true),dim=0)
    cor = torch.corrcoef(corr_tensor)[0][1]
    
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    
    sd_true = torch.std(y_true)
    sd_pred = torch.std(y_pred)
    
    numerator = 2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

def test_ccc():
    n_samples=1000
    y_true = np.arange(n_samples)
    y_pred = y_true + 1
    y_true = np.array([-1.8430, -2.9625, -2.1838, -1.8075, -1.8022])
    y_pred = np.array([0.2252, 0.2269, 0.2403, 0.2496, 0.2595])    
    y_pred = np.array([-1.8, -1.9625, -2.1838, -1.8075, -1.8022])   
    c = concordance_correlation_coefficient(y_true,y_pred)  
    c_torch = concordance_correlation_coefficient_torch(torch.tensor(y_true),torch.tensor(y_pred))    
    print("c is:{},and c_torch:{}".format(c,c_torch))

def test_pr():
    y_true = [0.2, 0.3, 0.5, 0.7]
    y_score = [0.1, 0.4, 0.35, 0.8]
     
     
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    print(precision)
    print(recall)
    print(thresholds)

def test_scaler():
    from sklearn.preprocessing import MinMaxScaler
    
    X = np.array([[1.0,2.0,3.0],[5.0,6.0,7.0]])
    X = X.transpose(1,0)
    scaler = MinMaxScaler()
    scaler.fit(X)             
    target = scaler.transform(X)      
    print(target)

def test_compute():
    vector_1d = np.array([3,4,5])
    numbers_1_to_12 = np.arange(start = 1, stop = 13)
    matrix_2d_ordered = numbers_1_to_12.reshape((3,4))
    print(matrix_2d_ordered)
    result = np.subtract(matrix_2d_ordered.transpose(1,0),vector_1d).transpose(1,0)
    print(result)
               
if __name__ == "__main__":
    # test_argwhere()
    # test_condition3()
    # test_condi_remove()
    # test_group()
    # test_corr()
    # test_corr_tensor()
    # test_slope()
    # test_each()
    # test_split_compute()
    # test_variable()
    # test_index()
    # test_copy()
    test_copy_torch()
    # test_compute()
    # test_temp()
    # test_pd_ser()
    # test_torch_vision()
    # torch_comp()
    # np_comp()  
    # test_np_norm()
    # test_concat()
    # test_tensor_sum()
    # test_linear()
    # test_inrange()
    # test_tensor_list()
    # test_reshape()
    # test_meshgrid()
    # test_linespace_2d()
    # test_pd_df()
    # test_pd_trans()
    # test_pd_index()
    # test_norm()
    # test_ccc()
    # test_pr()
    # test_scaler()
    
    
    