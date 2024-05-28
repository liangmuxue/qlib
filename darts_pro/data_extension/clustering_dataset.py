import pickle
from typing import Dict
import torch
import pandas as pd
import numpy as np
from datetime import datetime 
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from cus_utils.encoder_cus import StockNormalizer   
from sktime.transformations.panel.pca import PCATransformer
from sklearn.decomposition import PCA
import time
from darts_pro.data_extension.series_data_utils import StatDataAssis
from cus_utils.common_compute import batch_cov,normalization_axis,eps_rebuild,same_value_eps,slope_compute
import cus_utils.global_var as global_var

from .batch_dataset import BatchDataset

class ClustringBatchOutputDataset(BatchDataset):    
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process",range_num=[0,10000]):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
            

        output_batch_data = []
        target_batch_data = []
        loss_batch_data = []
        
        # 文件中已经包含了输出数据和目标数据
        with open(filepath, "rb") as fin:
            while True:
                try:
                    data = pickle.load(fin)
                    output_batch_data.append(data[0])
                    target_batch_data.append(data[1])
                    # loss_batch_data.append(data[2])
                except EOFError:
                    break  
                        
        # 训练数据 
        aggregated = self.create_aggregated_data(target_batch_data)    
        price_data = aggregated[-2]
        # 输出数据    
        x_bar = []
        z_pca = [] 
        lattend = []
        for item in output_batch_data[0]:
            x_bar.append(item[0].cpu().numpy())
            z_pca.append(item[1].cpu().numpy())
            lattend.append(item[2].cpu().numpy())
        x_bar = np.stack(x_bar).transpose(1,2,0)
        z_pca = np.stack(z_pca).transpose(1,2,0)
        lattend = np.stack(lattend).transpose(1,2,0)
        output_combine = (x_bar,z_pca,lattend)
        # loss_batch_data = np.stack(loss_batch_data)
        # loss_batch_data = loss_batch_data.reshape(loss_batch_data.shape[0]*loss_batch_data.shape[1],loss_batch_data.shape[2])
        # 目标数据
        self.target_data = aggregated 
        self.output_inverse_data = output_combine
        self.output_data = output_combine
        self.price_data = price_data
        # self.loss_batch_data = loss_batch_data

    def create_aggregated_data(self,batch_data):
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
            elif isinstance(elem, tuple):
                t_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        t_list.append(item)
                aggregated.append(t_list)    
            elif isinstance(elem, list):
                t_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        t_list.append(item)
                aggregated.append(t_list)                                    
            elif isinstance(elem, Dict):
                d_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
        return aggregated
    
    def build_price_data(self,target_info):
        price_data = []
        for i in range(len(target_info)):
            p_data = [ts["price_array"][25:] if ts is not None else [0 for _ in range(5)] for ts in target_info[i]]
            price_data.append(p_data)
        price_data = np.array(price_data)
        return price_data   
        
    def __getitem__(self, index):
        
        batch_data = [item[index] for item in self.target_data]
        (past_target,scaler,target_class,future_target,pca_price_target,adj_target,target_info) = batch_data
        # 反归一化取得实际目标数据
        whole_target = np.concatenate((past_target,future_target),axis=0)
        target_inverse = scaler.inverse_transform(whole_target)  
        output = (self.output[0][index],self.output[1][index])
        return target_inverse,target_class[0],pca_price_target,output,target_info 
    
    
class VareBatchOutputDataset(BatchDataset):    
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process"):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
            

        output_batch_data = []
        target_batch_data = []
        loss_batch_data = []
        
        # 文件中已经包含了输出数据和目标数据
        with open(filepath, "rb") as fin:
            while True:
                try:
                    data = pickle.load(fin)
                    output_batch_data.append(data[0])
                    target_batch_data.append(data[1])
                except EOFError:
                    break  
                        
        # 训练数据 
        aggregated = self.create_aggregated_data(target_batch_data)    
        # price_data = self.build_price_data(aggregated[-1])
        # 输出数据    
        yita = []
        z = [] 
        latent = [] 
        cell_output = []
        for item in output_batch_data[0]:
            yita.append(item[0])
            z.append(item[1])
            latent.append(item[2])
            cell_output.append(item[2])
        yita = np.stack(yita).transpose(1,2,0)
        z = np.stack(z).transpose(1,2,0)
        latent = np.stack(latent).transpose(1,2,0)
        cell_output = np.stack(cell_output).transpose(1,2,0)
        output_combine = (yita,z,latent,cell_output)
        # 目标数据
        self.target_data = aggregated 
        self.output_data = output_combine

    def create_aggregated_data(self,batch_data):
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
            elif isinstance(elem, tuple):
                t_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        t_list.append(item)
                aggregated.append(t_list)    
            elif isinstance(elem, list):
                t_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        t_list.append(item)
                aggregated.append(t_list)                                    
            elif isinstance(elem, Dict):
                d_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
        return aggregated
    
    def build_price_data(self,target_info):
        price_data = []
        for i in range(target_info.shape[0]):
            p_data = [ts["price_array"][25:] if ts is not None else [0 for _ in range(5)] for ts in target_info[i]]
            price_data.append(p_data)
        price_data = np.array(price_data)
        return price_data   
        
    def __getitem__(self, index):
        
        batch_data = [item[index] for item in self.target_data]
        (past_target,target_class,future_target,price_target,target_info) = batch_data
        return past_target,target_class,future_target,price_target,target_info  
    