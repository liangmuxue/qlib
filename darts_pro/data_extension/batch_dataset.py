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
    """二次训练数据集"""
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process",range_num=None):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
        self.range_num = range_num        
        
        batch_data = []
        with open(filepath, "rb") as fin:
            while True:
                try:
                    batch_data.append(pickle.load(fin))
                except EOFError:
                    break            
                
        aggregated = self.create_aggregated_data(batch_data)    
        
        if self.mode=="process":
            self.batch_data = aggregated 
        if self.mode.startswith("analysis"):
            self.batch_data = [aggregated_item[range_num[0]:range_num[1]] for aggregated_item in aggregated]
            self.target_data = self.get_df_data(fit_names,target_info=aggregated[-1][range_num[0]:range_num[1]])
            self.target_class = aggregated[0][range_num[0]:range_num[1],0,0]
        if self.mode=="analysis_reg":
            self.batch_data = [aggregated_item[range_num[0]:range_num[1]] for aggregated_item in aggregated]
            self.target_class = aggregated[0][range_num[0]:range_num[1],0,0]
            self.target_data = self.get_df_data(fit_names,target_info=aggregated[-1][range_num[0]:range_num[1]])
            self.analysis_data = self.get_df_data(target_col,target_info=aggregated[-1][range_num[0]:range_num[1]])
            print("self.target_data shape:{}".format(self.target_data.shape))
        if self.mode=="analysis_reg_ota":               
            self.target_data = aggregated[-2][range_num[0]:range_num[1]]
            target_inverse_data = []
            for i in range(range_num[1]-range_num[0]):
                scaler = aggregated[5][i]
                target_data = aggregated[-2][i]
                inverse_data = scaler.inverse_transform(target_data)
                target_inverse_data.append(inverse_data)
            target_inverse_data = np.stack(target_inverse_data)   
            # x_conv = target_inverse_data[:,:,self.fit_names[0]:self.fit_names[1]]
            y = np.expand_dims(np.sum(target_inverse_data[:,:,0],axis=1),axis=-1)
            x_conv_transform = target_inverse_data[:,:,self.fit_names[0]:self.fit_names[1]]
            shape_ori = x_conv_transform.shape
            x_conv_transform = x_conv_transform.reshape((shape_ori[0]*shape_ori[1], shape_ori[2]))
            self.x_conv_transform = MinMaxScaler().fit_transform(x_conv_transform).reshape(shape_ori)
            self.y_transform = MinMaxScaler().fit_transform(y)
            print("self.target_data shape:{}".format(self.target_data.shape))           
    
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
            elif isinstance(elem, Dict):
                d_list = []
                for sample in batch_data:
                    for item in sample[i]:
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
        return aggregated
    
    def build_origin_target_data(self):
        """生成原数据"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = self.batch_data 
        target_class = target_class[:,:,0]
        last_target_vr_class = target_class[:,1]
        whole_target = np.concatenate((past_target,target),axis=1)
        target_inverse = [scaler[i].inverse_transform(whole_target[i]) for i in range(len(scaler))]     
        target_inverse = np.stack(target_inverse)
        price_range = [ts["price_array"] for ts in target_info]
        price_range = np.stack(price_range)
        return target_inverse,price_range,target_class[:,0]
                    
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
            # 生成价位幅度目标 
            price_array = target_info["price_array"]
            raise_range = (price_array[-1] - price_array[-5])/price_array[-5]*10
            target_info["raise_range"] = raise_range
            past_target_ori = scaler.inverse_transform(past_target)
            # avoid infinite
            mask_idx = np.where(past_target_ori<0.01)[0]
            past_target_ori[mask_idx] = 0.01
            past_target_slope = (past_target_ori[1:,:] - past_target_ori[:-1,:])/past_target_ori[:-1,:]*10
            # 生成目标缩放器，用于后续归一化
            target_range_scaler = MinMaxScaler()
            target_range_scaler.fit(past_target_slope)
            target_info["target_range_scaler"] = target_range_scaler
            return past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info
        if self.mode=="analysis":
            return self.target_data[index],self.target_class[index]
        if self.mode=="analysis_reg":
            y = np.sum(self.analysis_data[index],axis=0)
            return self.target_data[index],y
        if self.mode=="analysis_reg_ota":
            return self.x_conv_transform[index],self.y_transform[index]           
                
    def __len__(self):
        if self.mode=="process":
            return self.batch_data[0].shape[0]  
        if self.mode.startswith("analysis"):
            return self.range_num[1] - self.range_num[0]
            
    
class BatchOutputDataset(BatchDataset):    
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process",range_num=[0,10000]):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
        self.range_num = range_num        

        output_batch_data = []
        target_batch_data = []
        
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
          
        # 输出数据    
        output_combine = []    
        for item in output_batch_data:
            out_item = np.stack([conv for conv in item],axis=2)[:,:,:,0,0]
            output_combine.append(out_item)
        output = np.concatenate(output_combine,axis=0)
        output_inverse_data = []
        target_inverse_data = []
        for index in range(output.shape[0]):
            output_item = output[index]
            scaler = aggregated[5][index]            
            output_inverse = scaler.inverse_transform(output_item)      
            output_inverse_data.append(output_inverse)
            # target_data = aggregated[-2][index]
            # inverse_data = scaler.inverse_transform(target_data)
            # target_inverse_data.append(inverse_data)      
        output_inverse_data = np.stack(output_inverse_data)[range_num[0]:range_num[1]]
        
        # 目标数据
        self.target_data = aggregated 
        self.output_data = output_inverse_data
        
    def __getitem__(self, index):
        
        batch_data = [item[index] for item in self.target_data]
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = batch_data
        # 反归一化取得实际目标数据
        # whole_target = np.concatenate((past_target,target),axis=0)
        target_inverse = scaler.inverse_transform(target)  
        return target_inverse
    
    