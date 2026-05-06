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
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.tft_series_dataset import TFTSeriesDataset
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.common_compute import process_outliers_multi_cols
import cus_utils.global_var as global_var

from cus_utils.log_util import AppLogger
logger = AppLogger()


def concat_scale_arr(scale_arr):
    """把2级分片定义合并为1级，用于模型参数设置"""
    
    df_group = scale_arr.groupby('p0', as_index=False)['instruments'].agg(lambda x: np.concatenate(list(x)))
    df_group['p'] = df_group['p0']
            
    return df_group.to_dict('records')

def emb_scale_arr(scale_arr):
    """把2级分片定义合并为涵盖上下级关系的定义"""
    
    # nested_dict = scale_arr.groupby(['p0', 'p1'])['instruments'].first().unstack().to_dict('index')
    # nested_dict = nested_dict.dropna()
    
    df_valid = scale_arr.dropna(subset=['p0', 'p1'])
    
    nested_dict = {}
    for l1, g1 in df_valid.groupby('p0'):
        inner = {row['p1']: row.to_dict() for _, row in g1.iterrows()}
        nested_dict[l1] = inner  
                
    return nested_dict
  
  

def get_scale_conf():
    indus_threhold_bin = [['cdifi','hsjs'],['abpi','yzyl','nffi']]
    cy_threhold_bin = [[0,2013],[2013,2030]]
    nt_threhold_bin = [[0],[1]]
    scale_conf_all = {'indus_scale':indus_threhold_bin,'cy_scale':cy_threhold_bin,'nt_scale':nt_threhold_bin}
    scale_conf = {'indus_scale':indus_threhold_bin}
    scale_conf = {'nt_scale':nt_threhold_bin}
    return scale_conf,scale_conf_all

class TFTFuturesDataset(TFTSeriesDataset):
            
    def _pre_process_df(self,df,val_range=None):
        """数据预处理"""
 
        # 补充行业数据
        indus_sql = "select v.code,industry_id,i.code as industry_code,IF(length(night_time_range)>1,1,0) as night_flag,v.exchange_id,price_range,limit_rate," \
            "magin_radio,create_year from trading_variety v left join futures_industry i on v.industry_id=i.id  where magin_radio is not null union " \
            "(select upper(concat('zs_',code)), id,upper(concat('zs_',code)),0,0,0,0,0,0 from futures_industry where delete_flag=0)"   
        base_info_data = self.dbaccessor.do_query(indus_sql)
        base_info_arr = []
        for item in base_info_data:
            base_info_arr.append([item[i] for i in range(len(item))])     
        base_info = pd.DataFrame(np.array(base_info_arr),columns=["instrument","industry","indus_code","night_flag",
                                    "exchange_id","price_range","limit_rate","magin_radio","create_year"]) \
            .astype({"instrument":str,"industry":str,"indus_code":str,"night_flag":int,
                     "exchange_id":int,"price_range":int,"limit_rate":int,"magin_radio":int,"create_year":int})  
        # 加入业务分片关联数据
        scale_conf,_ = get_scale_conf()
        for key in scale_conf.keys():
            threhold_bin = scale_conf[key]
            if key.startswith("indus"):
                for i,item in enumerate(threhold_bin):
                    base_info.loc[base_info['indus_code'].isin(item),key] = i
                base_info[key] = base_info[key].fillna(-1).astype(int)
            if key.startswith("cy"):
                for i,item in enumerate(threhold_bin):
                    base_info.loc[(base_info['create_year']>=item[0])&(base_info['create_year']<item[1]),key] = i
                base_info[key] = base_info[key].fillna(-1).astype(int)
            if key.startswith("nt"):
                for i,item in enumerate(threhold_bin):
                    base_info.loc[base_info['night_flag']==item[0],key] = i
                base_info[key] = base_info[key].fillna(-1).astype(int)                
        self.base_info = base_info     
         
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
        # 合并扩展数据
        df = pd.merge(base_info,df,on=["instrument"],how="left",validate="one_to_many")   
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
        
        # 做二次差分，为每个节点生成差分数组
        # self.build_past_roll_diff_data(df, group_column, 'open_range', 'open_diff_sec_norm')
        
        df['datetime_number'] = df['datetime_number'].astype(int)
        
        # 剔除open_diff超出范围的异常值
        df.loc[df['open_diff']>5,'open_diff'] = 5
        df.loc[df['open_diff']<-5,'open_diff'] = -5            
        # 生成业务分片均值数据
        scale_columns = list(scale_conf.keys())
        self.scale_dict,df_mean_norm = self.build_scale_mean(df,scale_columns,tar_col='open_diff',val_range=val_range)  
        # 根据分类信息得到分类权重
        self.scale_class_weights = {}
        for key in scale_columns:
            class_col = key +"_class"
            self.scale_class_weights[class_col] = []
            for i in range(2):
                df_mean_norm_item = df_mean_norm[key]
                df_mean_norm_item = df_mean_norm_item[df_mean_norm_item[key]==i].dropna()
                class_count = np.bincount(df_mean_norm_item[class_col].values) 
                class_weights = 1.0 / class_count
                class_weights = class_weights / class_weights.sum()
                self.scale_class_weights[class_col].append(class_weights)
            
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
        # # 针对个别异常数据，标准化后再次处理
        # df = process_outliers_multi_cols(
        #     df=df,
        #     cols=['dom_basis_rate'],
        #     range=2.0,
        #     method='median_fill',
        #     detect_method='iqr'
        # )           
        # 生成行业均值数据
        df = self.build_industry_mean(df,indus_info=base_info)     
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

    def build_past_roll_diff_data(self,df,group_column,source_col,tar_col):
        """针对指定数据，做多区段二次差分"""
        
        # 做二次差分，为每个节点生成差分数组
        window = self.step_len
        def group_rolling_diff(group_series, window_size):
            diff_list = []
            # 每组独立计算
            for i in range(len(group_series)):
                if i < window_size - 1:
                    diff_list.append(np.nan)  # 窗口不足 → NaN
                else:
                    # 取窗口：前 n 个值 + 当前值
                    window_vals = group_series.iloc[i - window_size + 1 : i + 1].values
                    current = window_vals[-1]
                    history = window_vals[:-1]
                    # 差分：当前 - 前面每一个,取绝对值，后续最小化绝对距离
                    diffs = np.abs(current - history).tolist()
                    diff_list.append(diffs)
            return diff_list
        # 按 group 分组，每组独立计算差分数组
        df['diff_range_sec'] = df.groupby(group_column)[source_col].transform(
            lambda x: group_rolling_diff(x, window)
        )
        # 所有差分统一全局标准化
        all_diffs = []
        for arr in df['diff_range_sec'].dropna():
            if any(np.isnan(arr)):
                continue
            all_diffs.extend(arr)
        all_diffs = np.array(all_diffs)
        # 全局均值、标准差
        mean = all_diffs.mean()
        std = all_diffs.std()
        # 标准化函数
        def normalize(arr):
            if isinstance(arr, list):
                return [(x - mean) / std for x in arr]
            return np.nan
        df[tar_col] = df['diff_range_sec'].apply(normalize)    
        del df['diff_range_sec']   
    
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
 
    def build_scale_mean(self,df,scale_columns,tar_col='open_diff',val_range=None):
        """针对特定指标，生成平均值"""
        
        trend_threhold = global_var.get_value("trend_threhold")
        bins = [trend_threhold['min'],trend_threhold['short'],trend_threhold['long'], trend_threhold['max']]
        labels = [0,1,2]
                             
        step_len = self.step_len - 1
        dict_list = {}
        merged = defaultdict(lambda: np.array([]))
        df_mean_norm_total = {}
        for scale_column in scale_columns:
            group_cols = ['datetime_number',scale_column]
            # 添加行并根据业务分区取平均值
            df_mean = df.groupby(group_cols)[tar_col].mean().reset_index()
            df_mean = df_mean[df_mean[scale_column]>=0].dropna()
            # do normalization
            norm_col = scale_column + "_norm"
            norm_col_class = scale_column + '_class'
            df_mean_norm = []
            for i in range(2):
                df_mean_item = df_mean[df_mean[scale_column]==i]
                df_train = df_mean_item[df_mean_item["datetime_number"]<int(val_range[0].strftime("%Y%m%d"))]
                # 针对diff_range数据，统一使用训练集的标准化参数.进行训练集和验证集数据的标准化
                scaler_train = StandardScaler()
                # scaler_train = MinMaxScaler()
                scaler_train.fit(df_train[[tar_col]])
                df_mean_item[[norm_col]] = scaler_train.transform(df_mean_item[[tar_col]])   
                # 分类信息
                df_mean_item[norm_col_class] = pd.cut(df_mean_item[norm_col], bins=bins, labels=labels, right=False) 
                # self.build_past_roll_diff_data(df_mean_item, scale_column, tar_col,norm_col_sec)
                df_mean_norm.append(df_mean_item)         

            df_mean_norm = pd.concat(df_mean_norm)
            df_mean_norm_total[scale_column] = df_mean_norm
            result_dict = df_mean_norm.groupby('datetime_number')[norm_col].apply(lambda x:{scale_column:np.array(x)}).to_dict()
            # result_dict_class = df_mean_norm.groupby('datetime_number')[norm_col_class].apply(lambda x:{scale_column+'_class':np.array(x)}).to_dict()
            nested_dict = {}
            for (k1, k2), value in result_dict.items():
                if k1 not in nested_dict:
                    nested_dict[k1] = {}  
                nested_dict[k1][k2] = value
            merged = dict_list.copy()
            for k, v in nested_dict.items():
                merged[k] = {**merged.get(k, {}), **v}      
            dict_list = merged         
        
        return dict_list,df_mean_norm_total
    
        