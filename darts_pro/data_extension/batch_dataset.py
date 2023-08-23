import pickle
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from cus_utils.encoder_cus import StockNormalizer   
from sktime.transformations.panel.pca import PCATransformer

import cus_utils.global_var as global_var

class BatchDataset(Dataset):
    def __init__(self,filepath=None,fit_names=None):
        
        train_file = "{}/train_batch.npy".format(filepath)
        with open(filepath, "rb") as fin:
            aggregated = pickle.load(fin)   
            
        # aggregated = []   
        # first_sample = batch[0]
        # for i in range(len(first_sample)):
        #     elem = first_sample[i][0]
        #     if isinstance(elem, np.ndarray):
        #         sample_list = np.concatenate([sample[i] for sample in batch],axis=0)
        #         aggregated.append(
        #             sample_list
        #         )
        #     elif isinstance(elem, MinMaxScaler):
        #         s_list = []
        #         for sample in batch:
        #             for item in sample[i]:
        #                 s_list.append(item)
        #         aggregated.append(s_list)
        #     elif isinstance(elem, StockNormalizer):
        #         aggregated.append([sample[i] for sample in batch])                
        #     elif isinstance(elem, Dict):
        #         d_list = []
        #         for sample in batch:
        #             for item in sample[i]:
        #                 d_list.append(item)    
        #         aggregated.append(d_list)       
        #     elif elem is None:
        #         aggregated.append(None)  
        
        self.batch_data = aggregated     
        self.fit_names = fit_names      
    
    def build_pca_data(self):
        """create pca data,using target data and relation data"""
        
        size = 1000
        future_target = self.batch_data[-2][:size]
        target_class = self.batch_data[-3][:size,0,0]
        transformer = PCATransformer(n_components=2)
        train_data = self.get_df_data()
        train_data = train_data[:size]
        train_data = train_data.transpose(0,2,1)
        rtn = transformer.fit_transform(train_data, target_class)
        print(rtn)
    
    def get_df_data(self):
        
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        target_data = None
        target_info = self.batch_data[-1]
        for ts in target_info:
            df_item = df_all[(df_all["instrument"]==ts["instrument"])&
                             (df_all["time_idx"]>=ts["future_start"])&(df_all["time_idx"]<ts["future_end"])] 
            item_values = np.expand_dims(df_item[self.fit_names].values,axis=0)
            if target_data is None:
                target_data = item_values
            else:
                target_data = np.concatenate((target_data,item_values),axis=0)
        return target_data
       
    def __getitem__(self, index):
        batch_data = [item[index] for item in self.batch_data]
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = batch_data  
        return past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info
    
    def __len__(self):
        return self.batch_data[0].shape[0]  
    
    