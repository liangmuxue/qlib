import pickle
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from cus_utils.encoder_cus import StockNormalizer   

class BatchDataset(Dataset):
    def __init__(self,filepath=None):
        train_file = "{}/train_batch.npy".format(filepath)
        with open(filepath, "rb") as fin:
            batch = pickle.load(fin)   
            
        aggregated = []   
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i][0]
            if isinstance(elem, np.ndarray):
                sample_list = np.concatenate([sample[i] for sample in batch],axis=0)
                aggregated.append(
                    sample_list
                )
            elif isinstance(elem, MinMaxScaler):
                s_list = []
                for sample in batch:
                    for item in sample[i]:
                        s_list.append(item)
                aggregated.append(s_list)
            elif isinstance(elem, StockNormalizer):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                d_list = []
                for sample in batch:
                    for item in sample[i]:
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
        
        self.batch_data = aggregated                
        
    def __getitem__(self, index):
        batch_data = [item[index] for item in self.batch_data]
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = batch_data  
        return past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info
    
    def __len__(self):
        return self.batch_data[0].shape[0]  
    
    