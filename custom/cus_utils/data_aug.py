import torch
import numpy as np
import pandas as pd
import random
import pickle

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    val_low = data_ser.quantile(0.25) - iqr
    val_up = data_ser.quantile(0.75) + iqr
    rule_low = (data_ser < val_low)
    rule_up = (data_ser > val_up)
    return (rule_low, rule_up), (val_low, val_up)
    
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认box_plot(scale=3)进行清洗
    param data: 接收pandas数据格式
    param col_name: pandas列名
    param scale: 尺度
    """
    
    data_n = data.copy()
    data_serier = data_n[col_name]
    rule, value = box_plot_outliers(data_serier, box_scale=scale)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]
    print("Delete number is:{}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is:{}".format(data_n.shape[0]))
    index_low = np.arange(data_serier.shape[0])[rule[0]]
    outliers = data_serier.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_serier.shape[0])[rule[1]]
    outliers = data_serier.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    # sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n      

def aug_data_view(file_path):
    data = np.load(file_path,allow_pickle=True)
    print("shape",data.shape)

def aug_pd_data_view(file_path):
    df = pd.read_pickle(file_path)
    print("df",df)
    
def aug_data_to_pd(file_path,tar_file_path,columns):
    data = np.load(file_path,allow_pickle=True)
    data = np.reshape(data,(-1,len(columns)))
    pd_data = pd.DataFrame(data,columns=columns)
    pd_data.to_pickle(tar_file_path)
    print("save ok")
        
def aug_data_process(file_path,train_path=None,test_path=None,sp_rate=0.7):
    """再次加工数据"""
    
    data = np.load(file_path,allow_pickle=True)
    # 累加时间部分的后五项数据，形成15+1=16的长度 
    data = np.concatenate((data[:,0:15,:],np.expand_dims(data[:,15:,:].sum(axis=1),axis=1)),axis=1)
    train_len = data.shape[0] * sp_rate
    # 切分并保存
    train_data = data[:train_len,:,:]   
    test_data = data[train_len:,:,:]   
    np.save(train_path,train_data)
    np.save(test_path,test_data)

def compare_dataset_consistence():
    """比较验证数据与推理数据的一致性"""
    
    val_file_path = "custom/data/results/data_compare_val_20220616.pkl"
    pred_file_path = "custom/data/results/data_compare_pred_20220616.pkl"
    with open(val_file_path, "rb") as fin:
        result_data_val = pickle.load(fin)           
    with open(pred_file_path, "rb") as fin:
        result_data_pred = pickle.load(fin) 

    names = ["target_info","past_target_total", "past_covariate_total", "historic_future_covariates_total","future_covariates_total","static_covariate_total"
               ,"covariate_future_total","past_future_round_targets","index_round_targets"]
    eps = 1e-3
    for i in range(1,len(result_data_val)):
        val_item = result_data_val[i]
        pred_item = result_data_pred[i]
        diff = np.abs(val_item - pred_item)
        compare_rs = np.where(diff>eps)
        if np.sum(compare_rs)>1:
            print("{} difference:{}".format(names[i],compare_rs))
        
if __name__ == "__main__":
    file_path = "custom/data/aug/test_100.npy"
    pd_file_path = "custom/data/aug/test_100.pkl"
    train_path = "custom/data/aug/test100_all_train.npy"
    test_path = "custom/data/aug/test100_all_test.npy"
    # aug_data_process(file_path,train_path=train_path,test_path=test_path)
    # aug_data_view(file_path)
    # aug_data_to_pd(file_path,pd_file_path,['datetime','instrument','dayofweek','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label'])\
    # aug_pd_data_view(pd_file_path)
    compare_dataset_consistence()
    
       
    
