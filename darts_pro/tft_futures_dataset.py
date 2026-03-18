import warnings

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES
from trader.utils.date_util import tradedays,get_tradedays_dur,get_tradedays
import pandas as pd
import numpy as np
import pickle
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.tft_series_dataset import TFTSeriesDataset
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.data_filter import DataFilter
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
logger = AppLogger()

class TFTFuturesDataset(TFTSeriesDataset):
            
    def _pre_process_df(self,df,val_range=None):
        """数据预处理"""
 
        # 补充行业数据
        indus_sql = "select code,industry_id,IF(length(night_time_range)>1,1,0) as night_flag,exchange_id,price_range,limit_rate," \
            "magin_radio from trading_variety where magin_radio is not null union " \
            "(select upper(concat('zs_',code)), id,0,0,0,0,0 from futures_industry where delete_flag=0)"   
        indus_data = self.dbaccessor.do_query(indus_sql)
        indus_info_arr = []
        for item in indus_data:
            indus_info_arr.append([item[i] for i in range(len(item))])     
        indus_info = pd.DataFrame(np.array(indus_info_arr),columns=["instrument","industry","night_flag","exchange_id","price_range","limit_rate","magin_radio"]) \
            .astype({"instrument":str,"industry":str,"night_flag":int,"exchange_id":int,"price_range":int,"limit_rate":int,"magin_radio":int})                     
        # 补充扩展数据
        ext_sql = "select CAST(date_format(e.date,'%Y%m%d') AS SIGNED),t.code,e.dom_basis_rate,e.near_basis_rate from " \
            "extension_trade_info e left join trading_variety t on e.var_id=t.id where e.var_id is not null"
        ext_data = self.dbaccessor.do_query(ext_sql)
        ext_info_arr = []
        for item in ext_data:
            ext_info_arr.append([item[i] for i in range(len(item))])
        ext_info = pd.DataFrame(np.array(ext_info_arr),columns=["datetime_number","instrument","dom_basis_rate","near_basis_rate"]).astype(
            {"datetime_number":int,"instrument":str,"dom_basis_rate":float,"near_basis_rate":float})   
        # 补充外部数据
        outer_sql = "select CAST(date_format(ot.date,'%Y%m%d') AS SIGNED),t.code,ot.close as ot_close from outer_trading_data ot " \
            "left join trading_variety_outer o on ot.code=o.code left join trading_variety t on o.var_id=t.id"
        outer_data = self.dbaccessor.do_query(outer_sql)
        outer_info_arr = []
        for item in outer_data:
            outer_info_arr.append([item[i] for i in range(len(item))])
        outer_info = pd.DataFrame(np.array(outer_info_arr),columns=["datetime_number","instrument","ot_close"]).astype(
            {"datetime_number":int,"instrument":str,"ot_close":float})          
        # 清除序列长度不够的品种
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"]     
        # Ignore Data Clean--lmx
        # df = data_filter.data_clean(df, self.step_len,valid_range=val_range,group_column=group_column,time_column=time_column)  
        # 重置异常值      
        df = self.reset_outlier(df)              
        # 生成时间字段
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
        logger.debug("begin group process")
        df["min_time"] = df.groupby(group_column)[time_column].transform("min")
        df[time_column] = df[time_column] - df["min_time"]
        df = df.drop(['min_time'], axis=1)
        # 取得品种创建年份
        df["create_year"] = df.groupby("instrument")['year'].transform("min")  
        df.reset_index(drop = True, inplace = True)        
        # 合并扩展数据
        df = pd.merge(indus_info,df,on=["instrument"],how="left",validate="one_to_many")   
        df = pd.merge(df,ext_info,on=["instrument","datetime_number"],how="left",validate="one_to_one")    
        df = pd.merge(df,outer_info,on=["instrument","datetime_number"],how="left",validate="one_to_one")
        # 消除nan数据
        df = df[df['industry']!='None']    
        df = df.fillna(0) 
        df = df[df['datetime_number']!=0]  
        # 生成价格差分数据
        df = df.sort_values(by=["instrument","datetime_number"],ascending=True)
        df['OPEN_COM'] = df['OPEN']
        def rl_apply(df_target,div,open_mode=False):
            values = df_target.values
            if open_mode:
                begin = values[-2]
            else:
                begin = values[0]            
            if div:
                if begin==0:
                    begin += 1e-3
                diff_range = (values[-1] - begin)/begin*100
            else:
                diff_range = (values[-1] - begin)
            # df_target['diff_range'] = diff_range
            return diff_range     
        def compute_diff(source_col,target_col,div=True,open_mode=False):
            if open_mode:
                rolling_size = 2
            else:
                rolling_size = self.cut_len
            diff_range = df.groupby(group_column)[source_col].rolling(window=rolling_size).apply(rl_apply,args=(div,open_mode,)).values
            df[target_col] = diff_range  
        compute_diff("VOLUME_CLOSE","VOLUME_RANGE")
        compute_diff("RSV5","rsv_diff",div=False)
        compute_diff("QTLUMA5","qtluma_diff",div=False)
        compute_diff("QTLU5","qtlu_diff")
        compute_diff("CCI5","cci_diff",div=False)
        compute_diff("SUMPMA5","sumpma_diff",div=False)
        compute_diff("OPEN","open_diff")
        compute_diff("OPEN","diff_range",open_mode=True)
        compute_diff("OPEN","open_range",open_mode=True)
        compute_diff("CLOSE","close_range",open_mode=True)
        # 剔除diff_range超出范围的异常值
        df['diff_range_norm'] = df['diff_range']
        df.loc[df['diff_range_norm']>5,'diff_range_norm'] = 5
        df.loc[df['diff_range_norm']<-5,'diff_range_norm'] = -5      
        df_train = df[df["datetime"]<pd.to_datetime(str(val_range[0].strftime("%Y-%m-%d")))]
        # 针对diff_range数据，统一使用训练集的标准化参数.进行训练集和验证集数据的标准化
        scaler_train = StandardScaler()
        scaler_train.fit(df_train[['diff_range_norm']])
        df[['diff_range_norm']] = scaler_train.transform(df[['diff_range_norm']])
        # 针对其他训练指标数据，统一使用训练集的标准化参数.进行训练集和验证集数据的标准化,需要按照品种分组进行
        norm_cols = self.get_past_columns()[:15]
        # 计算训练集每组的均值和标准差（针对所有特征列）
        group_stats = df_train.groupby('instrument')[norm_cols].agg(['mean', 'std']).reset_index()   
        # 处理标准差为 0 的情况（可选）
        for feat in norm_cols:
            group_stats[(feat, 'std')] = group_stats[(feat, 'std')].replace(0, 1)    
        # 展平列名：将 ('feat1', 'mean') 变为 'feat1_mean'
        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]  
        group_stats['instrument'] = group_stats['instrument_']
        # 合并到整个数据集
        df = df.merge(group_stats, on='instrument', how='left')
        for feat in norm_cols:
            mean_col = f'{feat}_mean'
            std_col = f'{feat}_std'
            df[f'{feat}'] = (df[feat] - df[mean_col]) / df[std_col]
        # 删除临时统计列
        df.drop(columns=[f'{feat}_{stat}' for feat in norm_cols for stat in ['mean', 'std']], inplace=True)            
        # df_val = df[(df["datetime"]>=pd.to_datetime(str(val_range[0]))) & (df["datetime"]<pd.to_datetime(str(val_range[1])))]

        # 生成行业均值数据
        df = self.build_industry_mean(df,indus_info=indus_info)     
        df['industry'] = df['industry'].astype(int)          
        df[time_column] = df[time_column].astype(int)          
        # 消除异常数据-Again
        df = df[df['industry']!='None']    
        df = df.fillna(0) 
        df = df[df['datetime_number']!=0]   
        cate_dict = {}               
        # 静态协变量:离散型生成嵌入数值，连续型生成标准化数据
        for conv_col in self.get_static_cate_columns():
            num_class = df[conv_col].unique().shape[0]
            cate_dict[conv_col] = num_class
            conv_col_scale = conv_col + "_scale"
            df[conv_col_scale] = LabelEncoder().fit_transform(df[conv_col].values)
        # Reset industry's create_year
        df.loc[df['instrument'].str.startswith('ZS_'),'create_year_scale'] = 0
        # 保存离散数量，用于后续模型参数
        self.cate_static_dict = cate_dict
        for conv_col in self.get_static_cont_columns():
            conv_col_scale = conv_col + "_scale"
            df[conv_col_scale] = (df[conv_col] - df[conv_col].mean())/df[conv_col].std()
        # 未来协变量提前进行归一化
        future_covariate_col = self.get_future_columns()     
        for conv_col in future_covariate_col:
            conv_col_scale = conv_col + "_scale"
            df[conv_col_scale] = (df[conv_col].astype(int) - df[conv_col].astype(int).min()) / (df[conv_col].astype(int).max() - df[conv_col].astype(int).min())                
        # 按照代码，新增排序字段，用于后续embedding
        rank_group_column = self.get_group_rank_column()
        df[rank_group_column] = df[group_column].rank(method='dense',ascending=True).astype("int")  
        self.build_group_rank_map(df)
        # 对index进行归一化，后续替换原index生成的静态协变量
        group_col_scale = rank_group_column + "_scale"   
        df[group_col_scale] = (df[rank_group_column].astype(int) - df[rank_group_column].astype(int).min()) / (df[rank_group_column].astype(int).max() - df[rank_group_column].astype(int).min())
        # Sort
        df = df.sort_values(by=["instrument","datetime_number"],ascending=True)
        return df    
    
    def get_cate_dict(self):
        return self.cate_static_dict
    
    def build_industry_mean(self,df,indus_info=None):
        """针对特定指标，生成行业平均值"""
        
        group_cols = ['datetime','industry']
        # 添加行并根据行业取平均值
        df_mean = df.groupby(group_cols).mean(numeric_only=True).reset_index()
        df_mean['industry'] = df_mean['industry'].astype(str)
        # 针对整体数值取平均值
        df_total_mean = df.groupby('datetime').mean(numeric_only=True).reset_index()
        indus_sql = "select id from futures_industry where code='all'"   
        indus_id = self.dbaccessor.do_query(indus_sql)[0][0]
        df_total_mean['industry'] = str(indus_id)
        df_mean = pd.concat([df_mean,df_total_mean],axis=0)
        indus_info_merge = indus_info[(indus_info['instrument'].str.startswith('ZS_'))]
        df_mean = pd.merge(indus_info_merge,df_mean,on=["industry"],how="left",validate="one_to_many")  
        df_mean['time_idx'] = df_mean['datetime'].groupby(df_mean['industry']).rank(method='dense',ascending=True)
        df_mean = df_mean[~df_mean['time_idx'].isna()]
        df_mean['time_idx'] = df_mean['time_idx'].astype(int) - 1
        df = pd.concat([df,df_mean])
        return df
    