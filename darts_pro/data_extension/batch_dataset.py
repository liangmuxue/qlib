import pickle
from typing import Dict
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from cus_utils.encoder_cus import StockNormalizer   
from sktime.transformations.panel.pca import PCATransformer
from darts_pro.data_extension.series_data_utils import StatDataAssis
import cus_utils.global_var as global_var
from cus_utils.log_util import AppLogger
logger = AppLogger()

class BatchDataset(Dataset):
    def __init__(self,filepath=None,fit_names=None,mode="process",range_num=None):
        
        self.mode = mode
        self.filepath = filepath
        self.fit_names = fit_names      
        self.range_num = range_num        
        
        batch_data = []
        with open(filepath, "rb") as fin:
            while True:
                try:
                    batch_data.append(pickle.load(fin))
                except EOFError:
                    break            
            
        first_sample = batch_data[0]
        aggregated = []
        for i in range(len(first_sample)):
            elem = first_sample[i][0]
            if isinstance(elem, np.ndarray):
                sample_list = np.concatenate([sample[i] for sample in batch_data],axis=0)
                aggregated.append(
                    sample_list
                )
            elif isinstance(elem, MinMaxScaler):
                s_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        s_list.append(item)
                aggregated.append(s_list)
            elif isinstance(elem, StockNormalizer):
                aggregated.append([sample[i] for sample in batch_data])                
            elif isinstance(elem, Dict):
                d_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
                
        if self.mode=="process":
            self.batch_data = aggregated     
        
        if self.mode=="analysis":
            self.target_class = aggregated[0][range_num[0]:range_num[1],0,0]
            self.target_data = self.get_df_data(fit_names,target_info=aggregated[-1][range_num[0]:range_num[1]])
            print("self.target_data shape:{}".format(self.target_data.shape))
    
    def build_pca_data(self):
        """create pca data,using target data and relation data"""
        
        size = 1000
        future_target = self.batch_data[-2][:size]
        target_class = self.batch_data[-3][:size,0,0]
        transformer = PCATransformer(n_components=2)
        train_data = self.get_df_data(self.fit_names).values
        train_data = train_data[:size]
        train_data = train_data.transpose(0,2,1)
        rtn = transformer.fit_transform(train_data, target_class)
        print(rtn)
    
    def get_df_data(self,fit_names,target_info=None):
        
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        target_data = None
        if target_info is None:
            target_info = self.batch_data[-1]
        for ts in target_info:
            df_item = df_all[(df_all["instrument"]==ts["instrument"])&
                             (df_all["time_idx"]>=ts["future_start"])&(df_all["time_idx"]<ts["future_end"])] 
            item_values = np.expand_dims(df_item[fit_names].values,axis=0)
            if target_data is None:
                target_data = item_values
            else:
                target_data = np.concatenate((target_data,item_values),axis=0)
        # target_data = pd.DataFrame(target_data,columns=fit_names)
        return target_data

    def analysis_df_pca(self,fit_names,range_num=1000,ret_file=None):
        data_assis = StatDataAssis()
        dataset = global_var.get_value("dataset")
        target_class = self.batch_data[-3][:range_num,0,0]
        total_imp = np.sum(target_class==3)
        train_data = self.get_df_data(fit_names)
        results = []
        result_cols = ["name","acc_cnt","acc_rate","recall"]
        for i,name in enumerate(fit_names):
            X = train_data[:,:,i:i+1]
            X = X[:range_num]
            predicted_labels,_ ,y_test = data_assis.fit_target_data(X,target_class,save_weight=False)
            predicted_labels_imp_index = np.where(predicted_labels==3)[0]
            acc_cnt = np.sum(y_test[predicted_labels_imp_index]==3)
            acc_rate = acc_cnt/predicted_labels_imp_index.shape[0]
            recall = acc_cnt/total_imp
            logger.info("name:{}| acc_cnt:{},acc_rate:{},recall:{}".format(name,acc_cnt,acc_rate,recall))
            results.append([name,acc_cnt,acc_rate,recall])
        results = np.array(results)
        np.save(ret_file,results)
        # results_df = pd.DataFrame(results,columns=result_cols)
        
        
    def __getitem__(self, index):
        
        if self.mode=="process":
            batch_data = [item[index] for item in self.batch_data]
            (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = batch_data  
            price_array = target_info["price_array"]
            raise_range = (price_array[-1] - price_array[-5])/price_array[-5]*10
            target_info["raise_range"] = raise_range
            return past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info
        if self.mode=="analysis":
            return self.target_data[index],self.target_class[index]
            
    def __len__(self):
        if self.mode=="process":
            return self.batch_data[0].shape[0]  
        if self.mode=="analysis":
            return self.range_num[1] - self.range_num[0]
            
    
    