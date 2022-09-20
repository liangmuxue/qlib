import numpy as np
import pandas as pd
from cus_utils.data_aug import outliers_proc,box_plot_outliers

def test_outliers():
    file_path = "/home/qdata/project/qlib/custom/data/aug/test100.npy"
    data = np.load(file_path,allow_pickle=True)
    data = np.reshape(data,(-1,data.shape[2]))
    columns = ['datetime_number','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label']
    df_data = pd.DataFrame(data,columns=columns)
    data_n = df_data.copy()
    col_name = 'VSTD5'
    data_serier = data_n[col_name]
    rule, value = box_plot_outliers(data_serier, box_scale=3)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]
    print("Delete number is:{}".format(len(index)))
    print("data_n begein",data_n[col_name].describe())
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("data_n end",data_n[col_name].describe())

def test_outliers_proc():
    file_path = "/home/qdata/project/qlib/custom/data/aug/test100.npy"
    data = np.load(file_path,allow_pickle=True)
    data = np.reshape(data,(-1,data.shape[2]))    
    columns = ['datetime_number','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label']
    df_data = pd.DataFrame(data,columns=columns)
    col_name = 'VSTD5'
    print("data_n begein",df_data[col_name].describe())
    data = outliers_proc(df_data,col_name)
    print("data_n end",data[col_name].describe())
        
if __name__ == "__main__":
    # test_outliers()
    test_outliers_proc()