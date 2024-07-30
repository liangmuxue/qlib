import os
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
from cus_utils.common_compute import batch_cov,normalization_axis,eps_rebuild,same_value_eps,get_trunck_index
import cus_utils.global_var as global_var
from cus_utils.log_util import AppLogger
logger = AppLogger()

class BatchDataset(Dataset):
    """二次训练数据集"""
    
    def __init__(self,filepath=None,is_training=False,trunk_mode=False,target_col=None,batch_size=0,fit_names=None,mode="process",range_num=None):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
        self.range_num = range_num    
        self.is_training = is_training 
        self.trunk_mode = trunk_mode
        self.batch_size = batch_size
        
        batch_data = []
        with open(filepath, "rb") as fin:
            while True:
                try:
                    batch_data.append(pickle.load(fin))
                except EOFError:
                    break            
                
        aggregated = self.create_aggregated_data(batch_data)    
        # 清除不合规数据
        aggregated = self.clear_inf_data(aggregated)
            
        if self.mode=="process":
            # 训练集数据量大，需要存储到硬盘上，以节省内存
            if trunk_mode:
                file_dir = os.path.dirname(filepath)
                file_dir = os.path.join(file_dir,"trunk")
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                total_size = aggregated[0].shape[0]
                batch_range = total_size//batch_size
                self.total_len = total_size
                for i in range(batch_range+1):
                    print("process {}".format(i))
                    save_path = os.path.join(file_dir,"trunk_{}.pkl".format(i))
                    if i==batch_range:
                        index_range = [j for j in range(i*batch_size,total_size)]
                    else:
                        index_range = [j for j in range(i*batch_size,(i+1)*batch_size)]
                    data = self._part_data(aggregated, index_range)
                    with open(save_path, "wb") as trunk_out:
                        pickle.dump(data,trunk_out)            
            else:
                self.batch_data = aggregated 
                self.total_len = aggregated[0].shape[0]
                aggregated = None
            # self.agg_data_check()
        if self.mode.startswith("analysis"):
            self.batch_data = [aggregated_item[range_num[0]:range_num[1]] for aggregated_item in aggregated]
            self.target_data = self.get_df_data(fit_names,target_info=aggregated[-2][range_num[0]:range_num[1]])
            self.target_class = aggregated[0][range_num[0]:range_num[1],0,0]
        if self.mode=="analysis_reg":
            self.batch_data = [aggregated_item[range_num[0]:range_num[1]] for aggregated_item in aggregated]
            self.target_class = aggregated[0][range_num[0]:range_num[1],0,0]
            self.target_data = self.get_df_data(fit_names,target_info=aggregated[-2][range_num[0]:range_num[1]])
            self.analysis_data = self.get_df_data(target_col,target_info=aggregated[2][range_num[0]:range_num[1]])
            print("self.target_data shape:{}".format(self.target_data.shape))
        if self.mode=="analysis_reg_ota":               
            self.target_data = aggregated[-3][range_num[0]:range_num[1]]
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
                # 忽略scaler
                if isinstance(elem[0],MinMaxScaler):
                    s_list = []
                    for sample in batch_data:
                        for item in sample[i]:
                            s_list.append(item[1])
                    aggregated.append(s_list)       
                    continue             
                sample_list = np.concatenate([sample[i] for sample in batch_data],axis=0)
                aggregated.append(
                    sample_list
                )
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
                        # 删除多余数据节省内存
                        del item["label_array"] 
                        del item["future_target"] 
                        d_list.append(item)    
                aggregated.append(d_list)       
            elif elem is None:
                aggregated.append(None)  
        return aggregated
    
    
    def clear_inf_data(self,agg_data):
        target = agg_data[-3]
        keep_idx = []
        for i in range(target.shape[0]):
            t_item = target[i]
            # 如果目标序列值都是一个，则排除
            flag = True
            for j in range(t_item.shape[1]):
                if np.unique(t_item[:,j].shape[0]==1):
                    flag = False
                    break
            if not flag:
                continue
            keep_idx.append(i)
        keep_idx = np.array(keep_idx)
        part_data = self._part_data(agg_data, keep_idx)
        if self.is_training:
            (past_target,past_covariates, historic_future_covariates,future_covariates,
             static_covariates,future_past_covariate,target_class,target,target_info,price_target) = part_data              
            return past_target,past_covariates, historic_future_covariates,future_covariates, \
                    static_covariates,target_class,target 
        else:
            return part_data
    
    def _part_data(self,agg_data,index_range):
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,future_past_covariate,target_class,target,target_info,price_target) = agg_data
        past_target = past_target[index_range]  
        past_covariates = past_covariates[index_range] 
        historic_future_covariates = historic_future_covariates[index_range]
        future_covariates = future_covariates[index_range]  
        static_covariates = static_covariates[index_range] 
        target_class = target_class[index_range]
        target = target[index_range]
        target_info = np.array(target_info)[index_range]
        price_target = price_target[index_range]
        return (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,future_past_covariate,target_class,target,target_info,price_target)
             
    def agg_data_check(self):
        aggregated = self.batch_data
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,price_target) = aggregated  
        c_tar = []       
        for i in range(len(target_info)):
            if target_info[i]['future_start_datetime']==20221012:
                c_tar.append(target_info[i])
        print("ctar len:",len(c_tar))
        
    def cluster_compare_data(self,aggregated_data,sampler_cnt=10):
        """对目标进行聚类，选取具有代表性的数据"""
        
        target_class = np.array([item[0] for item in aggregated_data[-3]])[:,0]
        future_target = aggregated_data[-2]
        results = []
        for i in range(4):
            imp_index = np.where((target_class==i))[0]
            s_indexes = np.random.choice(imp_index, sampler_cnt)
            t = np.stack(future_target[s_indexes].tolist())
            results.append(t)
        return np.stack(results)
        # kclusters = 10
        # center_list = []
        # for i in range(3):
        #     kmeans = KMeans(n_clusters=kclusters, random_state=12).fit(future_target[i])        
        #     center_list.append(kmeans.cluster_centers_)
        # return np.stack(center_list)
    
    def build_origin_target_data(self):
        """生成原数据"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = self.batch_data 
        target_class_real = np.array([tc[0] for tc in target_class])
        whole_target = [np.concatenate((past_target[i],target[i]),axis=0) for i in range(past_target.shape[0])]
        whole_target = np.stack(whole_target)
        target_inverse = [scaler[i][0].inverse_transform(whole_target[i]) for i in range(len(scaler))]     
        target_inverse = np.stack(target_inverse)
        price_range = [ts["price_array"] for ts in target_info]
        price_range = np.stack(price_range)
        return target_inverse,price_range,target_class_real
                    
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
    
    def get_df_data(self,fit_names,target_info=None,get_all=False):
        
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        if get_all:
            return df_all
        target_data = None
        for ts in target_info:
            df_item = df_all[(df_all["instrument"]==ts["instrument"])&
                             (df_all["time_idx"]>=ts["past_start"])&(df_all["time_idx"]<ts["future_end"])] 
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
            # 如果是分片存储模式，则需要读取文件
            if self.trunk_mode:
                batch_index = index//self.batch_size
                file_dir = os.path.dirname(self.filepath)
                filepath = os.path.join(file_dir,"trunk","trunk_{}.pkl".format(batch_index))
                # 计算偏移量，以取得单条数据
                index_range = index % self.batch_size
                with open(filepath, "rb") as fin:
                    batch_data = pickle.load(fin) 
                    batch_data = self._part_data(batch_data, index_range)   
            else:             
                batch_data = [item[index] for item in self.batch_data]
            return batch_data
        if self.mode=="analysis":
            return self.target_data[index],self.target_class[index]
        if self.mode=="analysis_reg":
            y = np.sum(self.analysis_data[index],axis=0)
            return self.target_data[index],y
        if self.mode=="analysis_reg_ota":
            return self.x_conv_transform[index],self.y_transform[index]           
                
    def __len__(self):
        if self.mode=="process":
            return self.total_len 
        if self.mode.startswith("analysis"):
            return self.range_num[1] - self.range_num[0]
            
class BatchCluDataset(BatchDataset):
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process",range_num=None,pre_static_datas=None):
        
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
        # 生成新的维度形状
        if pre_static_datas is not None:
            # 如果使用预先设定的静态数据，需要过滤当前数据不存在对应股票的记录
            aggregated = self.refileter_missing_data(aggregated,pre_static_datas)
            
        # static_datas = None
        aggregated,static_datas = self.combine_batch_data(aggregated,pre_static_datas=pre_static_datas)
        # aggregated = self.combine_smb_data(aggregated)
        self.batch_data = aggregated 
        # 保存静态属性数据
        self.static_datas = static_datas
                
    def combine_smb_data(self,data):
        """合并为多只股票并列的格式"""
        
        start_date_rec = [] 
        instrument_data = [] 
        for item in data[-2]:
            start_date_rec.append(item["start"])
            instrument_data.append(item["item_rank_code"])
        
        total_len = len(data[-1])
        # 根据批次内维度进行聚合
        batch_item_size = 256
        batch_len = total_len//batch_item_size + 1
        
        def concat_shape(item_data):
            return [batch_len,batch_item_size] + list(item_data.shape)
        
        # 按照新结构生成空数据，然后填充
        past_target_combine = np.zeros(concat_shape(data[0][0]))
        past_covariates_combine = np.zeros(concat_shape(data[1][0]))
        historic_future_covariates_combine = np.zeros(concat_shape(data[2][0]))
        future_covariates_combine = np.zeros(concat_shape(data[3][0]))
        static_covariates_combine = np.zeros([batch_len,batch_item_size,1,data[4].shape[-1]])
        scaler_tuple_combine = np.array([[None for _ in range(batch_item_size)] for _ in range(batch_len)])
        target_class_combine = np.zeros(concat_shape(data[6][0]))
        target_combine = np.zeros(concat_shape(data[7][0]))
        target_info_combine = np.array([[None for _ in range(batch_item_size)] for _ in range(batch_len)])
        rank_index_combine = np.zeros([batch_len,batch_item_size])
        price_target_combine = np.zeros(concat_shape(data[-1][0]))
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,price_target) = data       
        # 在这里对静态协变量进行全局归一化
        static_covariates = MinMaxScaler().fit_transform(static_covariates[:,0,:])
             
        def fill_data(combine_data,item_data,i,j):
            if len(combine_data.shape)==3:
                combine_data[i,j,:] = item_data
            else:
                combine_data[i,j,:,:] = item_data
                
             
        # 遍历数据，进行位置匹配
        for i in range(total_len):
            b_idx = i // batch_item_size
            item_idx = i % batch_item_size
            # 直接按照索引坐标填充
            fill_data(past_target_combine,past_target[i], b_idx,item_idx)
            fill_data(past_covariates_combine,past_covariates[i], b_idx,item_idx)
            fill_data(historic_future_covariates_combine,historic_future_covariates[i], b_idx,item_idx)
            fill_data(future_covariates_combine,future_covariates[i], b_idx,item_idx)
            fill_data(static_covariates_combine,static_covariates[i], b_idx,item_idx)
            # 提前准备股票索引编号
            rank_index_combine[b_idx,item_idx] = target_info[i]["item_rank_code"]
            scaler_tuple_combine[b_idx,item_idx] = (scaler_tuple[i][0],scaler_tuple[i][1])
            fill_data(target_class_combine,target_class[i], b_idx,item_idx)
            fill_data(target_combine,target[i], b_idx,item_idx)
            target_info_combine[b_idx,item_idx] = target_info[i]
            fill_data(price_target_combine,price_target[i], b_idx,item_idx)
         
        past_price_target = price_target_combine[:,:,:past_target_combine.shape[-2],0]
        future_price_target = price_target_combine[:,:,past_target_combine.shape[-2]:,0]
        # 提前计算距离矩阵，包括过去和未来两部分
        adj_target_combine = batch_cov(torch.Tensor(past_price_target).to("cuda:0")).cpu().numpy()
        adj_future_combine = batch_cov(torch.Tensor(future_price_target).to("cuda:0")).cpu().numpy()
        # 合并传值
        adj_target_combine = np.concatenate([adj_target_combine,adj_future_combine],axis=1)
        
        # 生成一维pca主成分，用于后续横向比较
        transformer = PCA(n_components=1)
        pca_target = []
        for i in range(future_price_target.shape[0]):
            p = transformer.fit_transform(future_price_target[i])
            pca_target.append(p)
        pca_target = np.stack(pca_target)
        # 合并传输数据
        price_target_combine = np.concatenate((price_target_combine,np.expand_dims(pca_target,-1)),axis=2)
        return (past_target_combine,past_covariates_combine, historic_future_covariates_combine,future_covariates_combine,
                static_covariates_combine,scaler_tuple_combine,target_class_combine,target_combine,target_info_combine,
                rank_index_combine,adj_target_combine,price_target_combine)   
    
    
    def refileter_missing_data(self,data,pre_static_datas):
        """过滤当前数据不存在对应股票的记录"""
        
        total_len = len(data[-1])
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,price_target) = data     
        
        indexes = []            
        for i in range(total_len):
            # 取得数据中对应的股票代码，如果不在预留数据集合中，则标记
            rank_code = target_info[i]["item_rank_code"]
            if rank_code in pre_static_datas:
                indexes.append(i)
        indexes = np.array(indexes)
        return (past_target[indexes],past_covariates[indexes], historic_future_covariates[indexes],future_covariates[indexes],
         static_covariates[indexes],scaler_tuple[indexes],target_class[indexes],target[indexes],np.array(target_info)[indexes].tolist(),price_target)                    
        
        
    def combine_batch_data(self,data,split_flag=False,pre_static_datas=None):
        """合并为以时间段为单位的多只股票并列的格式"""
        
        start_date_rec = [] 
        instrument_data = [] 
        for item in data[-2]:
            start_date_rec.append(item["future_start_datetime"])
            instrument_data.append(item["item_rank_code"])
        
        total_len = len(data[-2])
        # 以开始日期为单位进行聚合
        start_date_uni = list(set(start_date_rec))
        # 股票代码集合，如果已经传参使用传单的集合，否则根据实际数据生成集合
        if pre_static_datas is not None:
            instrument_data_uni = pre_static_datas[:,0].tolist()
        else:
            instrument_data_uni = list(set(instrument_data))
        instrument_data_uni = np.array(instrument_data_uni).astype(np.int32).tolist()
        
        def concat_shape(item_data):
            return [len(start_date_uni),len(instrument_data_uni)] + list(item_data.shape)
        
        # 按照新结构生成空数据，然后填充
        past_target_combine = np.zeros(concat_shape(data[0][0]))
        past_covariates_combine = np.zeros(concat_shape(data[1][0]))
        historic_future_covariates_combine = np.zeros(concat_shape(data[2][0]))
        future_covariates_combine = np.zeros(concat_shape(data[3][0]))
        static_covariates_combine = np.zeros(concat_shape(data[4][0]))
        scaler_tuple_combine = np.array([[None for _ in range(len(instrument_data_uni))] for _ in range(len(start_date_uni))])
        target_class_combine = np.zeros(concat_shape(data[6][0]))
        target_combine = np.zeros(concat_shape(data[7][0]))
        target_info_combine = np.array([[None for _ in range(len(instrument_data_uni))] for _ in range(len(start_date_uni))])
        price_target_combine = np.zeros(concat_shape(data[-1][0]))
        # 增加涨跌幅度数据
        raise_target_combine = np.zeros([price_target_combine.shape[0],price_target_combine.shape[1],future_covariates_combine.shape[2]])
        
        def fill_data(combine_data,item_data,i,j):
            combine_data[i,j,:,:] = item_data
        
        instru_idx = [[] for _ in range(len(start_date_uni))]    
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,price_target) = data   
        
        for i in range(total_len):
            start = target_info[i]["start"]
            # if start==2:
            #     print("2 rank:",target_info[i]["item_rank_code"])
        # 维护索引列表，后续作为对照
        rank_index_combine = np.zeros([len(start_date_uni),len(instrument_data_uni)])
        # 维护股票静态属性，用于图卷积的关系学习层
        static_datas = np.zeros([len(instrument_data_uni),data[4].shape[-1]])
        # 遍历数据，进行位置匹配
        t = datetime.now()
        for i in range(total_len):
            # 根据开始日期，以及股票编号，反向查询对应的位置索引
            start = target_info[i]["future_start_datetime"]
            instrument = target_info[i]["item_rank_code"]
            start_date_idx = start_date_uni.index(start)
            # 根据日期内索引号，依次填充实际数据(实现末尾补0)
            # value_index = len(instru_idx[start_date_idx]) - 1
            # 根据日期内索引号，填充实际数据(实现中间缺失值补0)
            value_index = instrument_data_uni.index(instrument)
            # 维护对应日期下的股票列表索引,两种方式
            instru_idx[start_date_idx].append(instrument)            
            rank_index_combine[start_date_idx,value_index] = instrument
            # 股票静态属性,覆盖式填充
            static_datas[value_index,:] = static_covariates[i]
            # 直接按照索引坐标填充
            fill_data(past_target_combine,past_target[i], start_date_idx,value_index)
            fill_data(past_covariates_combine,past_covariates[i], start_date_idx,value_index)
            fill_data(historic_future_covariates_combine,historic_future_covariates[i], start_date_idx,value_index)
            fill_data(future_covariates_combine,future_covariates[i], start_date_idx,value_index)
            fill_data(static_covariates_combine,static_covariates[i], start_date_idx,value_index)
            scaler_tuple_combine[start_date_idx,value_index] = scaler_tuple[i][0],scaler_tuple[i][1]
            fill_data(target_class_combine,target_class[i], start_date_idx,value_index)
            fill_data(target_combine,target[i], start_date_idx,value_index)
            target_info_combine[start_date_idx,value_index] = target_info[i] 
            fill_data(price_target_combine,price_target[i], start_date_idx,value_index)
            # 维护涨跌幅度数据部分
            price_array = target_info[i]["price_array"][-future_covariates_combine.shape[2]-1:]
            raise_range = (price_array[1:] - price_array[:-1])/price_array[:-1]   
            raise_target_combine[start_date_idx,value_index,:] = raise_range
        print("loop time:{}".format((datetime.now()-t).seconds))
        
        # Norm target 
        past_target_combine = normalization_axis(past_target_combine,axis=1)   
          
        past_price_target = price_target_combine[:,:,:past_target_combine.shape[-2],0]
        future_price_target = price_target_combine[:,:,past_target_combine.shape[-2]:,0]
      
        # 提前计算距离矩阵，包括过去和未来两部分
        # adj_target_combine = batch_cov(torch.Tensor(past_price_target).to("cuda:0")).cpu().numpy()
        # adj_future_combine = batch_cov(torch.Tensor(future_price_target).to("cuda:0")).cpu().numpy()
        # # 合并传值
        # adj_target_combine = np.concatenate([adj_target_combine,adj_future_combine],axis=1)
        adj_target_combine = raise_target_combine # np.zeros([len(start_date_uni),len(instrument_data_uni),1])
        transformer = PCA(n_components=1)
        
        # 价格pca主成分，用于后续横向比较  
        pca_price_target = []
        for i in range(future_price_target.shape[0]):
            # 处理价格重复值的问题
            future_price_target[i] = same_value_eps(future_price_target[i])       
            pca_price_target.append(transformer.fit_transform(future_price_target[i]))
        price_target_combine = np.concatenate((past_price_target,future_price_target),axis=2)  
        price_target_combine = np.expand_dims(price_target_combine,axis=-1)
        
        pca_price_target = np.expand_dims(np.array(pca_price_target),-1)
        price_target_combine = np.concatenate((price_target_combine,pca_price_target),axis=2) 
        
        # 目标pca主成分，用于后续横向比较  
        pca_target = []
        transformer = PCA(n_components=2)
        for i in range(future_price_target.shape[0]):
            pca_item_target = []
            for j in range(target_combine.shape[-1]):
                p = transformer.fit_transform(target_combine[i,:,:,j])
                pca_item_target.append(p)
            pca_target.append(pca_item_target)
           
        pca_target = np.array(pca_target).transpose(0,2,3,1) 
        # 合并传输数据   
        target_combine = np.concatenate((target_combine,pca_target),axis=2) 
        # Norm static data here
        static_covariates_combine = normalization_axis(static_covariates_combine,axis=1)     
        # static_datas = normalization_axis(static_datas,axis=1)      
        
        # 使用掩码矩阵用于处理缺失值问题
        rank_index_combine = (rank_index_combine>0).astype(np.int32) 
        if not split_flag:
            # 返回重组后的数据，以及对应的索引
            return (past_target_combine,past_covariates_combine, historic_future_covariates_combine,future_covariates_combine,
                    static_covariates_combine,scaler_tuple_combine,target_class_combine,
                    target_combine,target_info_combine,rank_index_combine,adj_target_combine,price_target_combine),static_datas
            
        # 拆分为小的数据批次
        split_rate = 3
        # 根据股票数量计算拆分范围，最后一段一般是不全的
        size = len(instrument_data_uni)//split_rate           
        def sp_data(data,date_index,begin_index,end_index=0,data_rebuild=None):
            if end_index>0:
                data_split = data[date_index,begin_index:end_index]
            else:
                data_split = data[date_index,begin_index:]
                # 最后的一段数据需要补0对齐
                split_size = size - data_split.shape[1]
                if isinstance(data_split[0,0],np.ndarray):
                    padding_ele = np.zeros((split_size,data.shape[2],data.shape[3]))
                else:
                    padding_ele = np.array([None for _ in range(split_size)])
                data_split = np.concatenate((data_split,padding_ele))
                data_rebuild.append(data_rebuild)
            return data_split        

        past_target_rebuild = []
        past_covariates_rebuild = []
        historic_future_covariates_rebuild = []
        future_covariates_rebuild = []
        static_covariates_rebuild = []
        scaler_tuple_rebuild = []
        target_class_rebuild = []
        target_rebuild = []
        target_info_rebuild = []
          
        dt_ins_index = []   
        for i in range(len(start_date_uni)):
            dt_ins_index = instru_idx[i]
            for j in range(split_rate):
                begin_idx = j * size
                end_idx = begin_idx + size if j<split_rate-1 else 0
                # 生成索引数据
                instru_index = instru_idx[i,begin_idx:end_idx]
                dt_ins_index.append([i,instru_index])
                # 生成实际数据
                sp_data(past_target_combine,i,begin_idx,end_idx,data_rebuild=past_target_rebuild)         
                sp_data(past_covariates_combine,i,begin_idx,end_idx,data_rebuild=past_covariates_rebuild)   
                sp_data(historic_future_covariates_combine,i,begin_idx,end_idx,data_rebuild=historic_future_covariates_rebuild)     
                sp_data(future_covariates_combine,i,begin_idx,end_idx,data_rebuild=future_covariates_rebuild)     
                sp_data(static_covariates_combine,i,begin_idx,end_idx,data_rebuild=static_covariates_rebuild)    
                sp_data(scaler_tuple_combine,i,begin_idx,end_idx,data_rebuild=scaler_tuple_rebuild)   
                sp_data(target_class_combine,i,begin_idx,end_idx,data_rebuild=target_class_rebuild)     
                sp_data(target_combine,i,begin_idx,end_idx,data_rebuild=target_rebuild)             
                sp_data(target_info,i,begin_idx,end_idx,data_rebuild=target_info_rebuild)  
           
        return (past_target_rebuild,past_covariates_rebuild, historic_future_covariates_rebuild,future_covariates_rebuild,
                static_covariates_rebuild,scaler_tuple_rebuild,target_class_rebuild,target_rebuild,target_info_rebuild)   
                
    def __getitem__(self, index):
        
        batch_data = [item[index] for item in self.batch_data]
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,rank_index,adj_target,price_target) = batch_data
        # target_info["raise_range"] = raise_range
        # (scaler,future_past_covariate) = scaler_tuple
        # 生成图矩阵相关数据,使用价格数据作为关联数据
        # pa = MinMaxScaler().fit_transform(price_array[:past_target.shape[-2]])
        return past_target,past_covariates, historic_future_covariates,future_covariates, \
            static_covariates, (scaler_tuple, None),target_class,target,target_info,rank_index,adj_target,price_target
    
class BatchInferDataset(BatchDataset):
    """用于预测推理的数据集"""
    
    def __init__(self,filepath=None,cur_date=None,target_col=None,fit_names=None):
        
        self.cur_date = cur_date
        super().__init__(filepath,target_col=target_col,fit_names=fit_names)
                
    def create_aggregated_data(self,batch_data):
        """过滤为只包含指定日期的结果集"""
        
        aggregated_data = super().create_aggregated_data(batch_data)
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler_tuple,target_class,target,target_info,price_target) = aggregated_data
        keep_idx = []
        for i in range(len(target_info)):
            future_start_datetime = target_info[i]["future_start_datetime"]
            # 数据中的未来预测开始日期需要和当前日期一致
            if future_start_datetime!=int(self.cur_date):
                continue
            keep_idx.append(i)
        keep_idx = np.array(keep_idx)
        
        past_target = past_target[keep_idx]  
        past_covariates = past_covariates[keep_idx] 
        historic_future_covariates = historic_future_covariates[keep_idx]
        future_covariates = future_covariates[keep_idx]  
        static_covariates = static_covariates[keep_idx] 
        target_class = target_class[keep_idx]
        target = target[keep_idx]
        scaler_tuple = [scaler_tuple[i] for i in keep_idx]
        target_info = [target_info[i] for i in keep_idx]
        price_target = price_target[keep_idx]
        
        return (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler_tuple,target_class,target,target_info,price_target)
                                     
class BatchOutputDataset(BatchDataset):    
    
    def __init__(self,filepath=None,target_col=None,fit_names=None,mode="process",range_num=[0,10000]):
        
        self.mode = mode
        self.filepath = filepath
        self.target_col = target_col
        self.fit_names = fit_names      
            

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
        target_info = aggregated[-1]
        save_data_path = "custom/data/asis/ana_data.npy"
        # np_data = self.get_df_data(fit_names,target_info=target_info)
        # np.save(save_data_path,np_data) 
        np_data = np.load(save_data_path)
        output = self.build_output(output_batch_data, range_num)
        # 目标数据
        self.target_data = aggregated 
        self.output_data = output
        self.np_data = np_data
    
    def build_output(self,output_batch_data,range_num=None):
        
        
        # 输出数据    
        output_combine = []    
        return output_combine
        for item in output_batch_data:
            out_item = np.stack([conv for conv in item],axis=2)[:,:,:,0,0]
            output_combine.append(out_item)
        output = np.concatenate(output_combine,axis=0)
        output_inverse_data = []
        target_inverse_data = []
        for index in range(output.shape[0]):
            output_item = output[index]
            # scaler = aggregated[5][index][0]           
            # output_inverse = scaler.inverse_transform(output_item)      
            # output_inverse_data.append(output_inverse)
            # target_data = aggregated[-2][index]
            # inverse_data = scaler.inverse_transform(target_data)
            # target_inverse_data.append(inverse_data)      
            
        output_inverse_data = np.stack(output_inverse_data)
        if range_num is not None:
            output_inverse_data = output_inverse_data[range_num[0]:range_num[1]]
            self.range_num = range_num    
        else:
            self.range_num = [0,output_inverse_data.shape[0]] 
        
        return output_combine
               
    def __getitem__(self, index):
        
        batch_data = [item[index] for item in self.target_data]
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler_tuple,target_class,target,target_info) = batch_data
        # 反归一化取得实际目标数据
        whole_target = np.concatenate((past_target,target),axis=0)
        target_inverse = scaler_tuple[0].inverse_transform(whole_target)  
        output_inverse = self.output_inverse_data[index]
        return target_inverse,target_class[0],output_inverse,target_info

    