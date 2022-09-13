import torch
import numpy as np
import pandas as pd

def aug_data_view(file_path):
    data = np.load(file_path,allow_pickle=True)
    print("shape",data.shape)
    
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
    
if __name__ == "__main__":
    file_path = "custom/data/aug/test100_all.npy"
    train_path = "custom/data/aug/test100_all_train.npy"
    test_path = "custom/data/aug/test100_all_test.npy"
    # aug_data_process(file_path,train_path=train_path,test_path=test_path)
    aug_data_view(file_path)
    
       
    