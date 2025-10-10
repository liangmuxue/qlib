import os
import torch
import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime

from cus_utils.db_accessor import DbAccessor
from pickle import TRUE

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
                
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
    
    val_file_path = "custom/data/results/data_compare_val_20250527.pkl"
    val_file_path = "custom/data/results/data_compare_predresult_20250407.pkl"
    pred_file_path = "custom/data/results/data_compare_pred_20250407.pkl"
    with open(val_file_path, "rb") as fin:
        result_data_val = pickle.load(fin)           
    with open(pred_file_path, "rb") as fin:
        result_data_pred = pickle.load(fin) 

    names = ["target_info","past_target_total", "past_covariate_total", "historic_future_covariates_total","future_covariates_total","static_covariate_total"
               ,"past_future_round_targets","index_round_targets"]
    eps = 1e-3
    for i in range(1,len(result_data_val)):
        val_item = result_data_val[i]
        pred_item = result_data_pred[i]
        diff = np.abs(val_item - pred_item)
        compare_rs = np.where(diff>eps)
        # if names[i]=="past_future_round_targets":
        #     print("ggg")
        if np.sum(compare_rs)>1:
            print("{} difference:{}".format(names[i],compare_rs))

def compare_clean_data_and_continus_data(match_date=None):
    """比较用于训练的数据和主连数据是否一致及完备"""
    
    dbaccessor = DbAccessor({})
    ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
    ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
    compare_results = []
    for row in ins_data.values:
        symbol = row[0] 
        sql = "select c.date,v.exchange_id from dominant_continues_data_cross c left join trading_variety v on " \
            "c.code=v.code where c.code='{}' order by date desc limit 1".format(symbol)
        result = dbaccessor.do_query(sql)
        if len(result)==0:
            date = 0
            exchange_id = 0
        else:
            date = int(result[0][0].strftime("%Y%m%d"))
            exchange_id = int(result[0][1])
        compare_results.append([symbol,date,exchange_id])
    # print(compare_results)
    compare_results = pd.DataFrame(np.array(compare_results),columns=['code','date','exchange_id'])
    compare_results['date'] = compare_results['date'].astype(int)
    compare_results['exchange_id'] = compare_results['exchange_id'].astype(int)
    lack_data = compare_results[compare_results['date']<match_date]
    print("lack data:{}".format(lack_data))
    print("min date:{}".format(lack_data['date'].min()))
    return lack_data

def compare_clean_data_and_1min_cross_data(match_date=None):
    """比较用于训练的数据和1分钟主力合约交错数据是否一致及完备"""
    
    dbaccessor = DbAccessor({})
    from data_extract.akshare_futures_extractor import AkFuturesExtractor
    extractor = AkFuturesExtractor(savepath="/home/qdata/futures_data")   
    
    ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
    ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
    compare_results = []
    for row in ins_data.values:
        symbol = row[0] 
        main_contract_name = extractor.get_main_contract_name(symbol, str(match_date), use_1min_data=True)
        item_save_path = os.path.join(extractor.get_1min_save_path(),"{}.csv".format(main_contract_name))
        # 检查分钟数据（交错模式）的合约文件是否存在
        if not os.path.exists(item_save_path):
            file_exists_flag = 0
            main_code = symbol
            date = '2005-01-01 00:00:00'
        else:
            file_exists_flag = 1
            sql = "select datetime from dominant_real_data_1min_cross where code='{}' order by datetime desc limit 1".format(main_contract_name)
            result = dbaccessor.do_query(sql)
            if len(result)==0:
                date = '2005-01-01 00:00:00'
                main_code = symbol
            else:
                date = result[0][0]
                main_code = main_contract_name
        compare_results.append([symbol,main_code,date,file_exists_flag])
    # print(compare_results)
    compare_results = pd.DataFrame(np.array(compare_results),columns=['code','main_code','date','file_exists_flag'])
    compare_results['file_exists_flag'] = compare_results['file_exists_flag'].astype(int)
    compare_results['date'] = compare_results['date'].astype('datetime64[ns]')
    match_date_date = datetime.strptime(str(match_date), "%Y%m%d")
    lack_data = compare_results[compare_results['date']<match_date_date]
    print(compare_results)
    print("lack data:{}".format(lack_data))
    # print("min date:{}".format(lack_data['date'].min()))
    return lack_data
           
def swig_object_to_dict(obj):
    """Swig Object to Dict"""
    
    if not hasattr(obj, '__swig_getmethods__'):
        return None

    result = {}
    for name, method in obj.__swig_getmethods__.items():
        try:
            if callable(method):
                result[name] = method(obj)
            else:
                result[name] = method
        except Exception:
            pass
    return result
        
if __name__ == "__main__":
    file_path = "custom/data/aug/test_100.npy"
    pd_file_path = "custom/data/aug/test_100.pkl"
    train_path = "custom/data/aug/test100_all_train.npy"
    test_path = "custom/data/aug/test100_all_test.npy"
    # aug_data_process(file_path,train_path=train_path,test_path=test_path)
    # aug_data_view(file_path)
    # aug_data_to_pd(file_path,pd_file_path,['datetime','instrument','dayofweek','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label'])\
    # aug_pd_data_view(pd_file_path)
    # compare_dataset_consistence()
    # compare_clean_data_and_continus_data(match_date=20250930)
    compare_clean_data_and_1min_cross_data(match_date=20250930)
    
       
    
