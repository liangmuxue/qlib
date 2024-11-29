import warnings

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES
from trader.utils.date_util import tradedays,get_tradedays_dur,get_tradedays

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
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
from numba.core.types import none
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
logger = AppLogger()

class TFTFuturesDataset(TFTSeriesDataset):
            
    def _pre_process_df(self,df,val_range=None):
        """数据预处理"""
        
        # 补充扩展数据
        ext_sql = "select CAST(date_format(e.date,'%Y%m%d') AS SIGNED),t.code,e.dom_basis_rate,e.near_basis_rate from " \
            "extension_trade_info e left join trading_variety t on e.var_id=t.id"
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
        data_filter = DataFilter()
        # 清除序列长度不够的股票
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"]       
        df = data_filter.data_clean(df, self.step_len,valid_range=val_range,group_column=group_column,time_column=time_column)        
        # 生成时间字段
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
        logger.debug("begin group process")
        df["min_time"] = df.groupby(group_column)[time_column].transform("min")
        df[time_column] = df[time_column] - df["min_time"]
        df = df.drop(['min_time'], axis=1)
        # 合并扩展数据
        df = pd.merge(df,ext_info,on=["instrument","datetime_number"],how="left",validate="one_to_one")    
        df = pd.merge(df,outer_info,on=["instrument","datetime_number"],how="left",validate="one_to_one")        
        df = df.fillna(0) 
        # 静态协变量和未来协变量提前进行归一化
        for conv_col in self.get_static_columns():
            conv_col_scale = conv_col + "_scale"
            df[conv_col_scale] = (df[conv_col] - df[conv_col].min()) / (df[conv_col].max() - df[conv_col].min())    
        future_covariate_col = self.get_future_columns()          
        for conv_col in future_covariate_col:
            conv_col_scale = conv_col + "_scale"
            df[conv_col_scale] = (df[conv_col].astype(int) - df[conv_col].astype(int).min()) / (df[conv_col].astype(int).max() - df[conv_col].astype(int).min())
                    
        # 按照股票代码，新增排序字段，用于后续embedding
        rank_group_column = self.get_group_rank_column()
        df[rank_group_column] = df[group_column].rank(method='dense',ascending=True).astype("int")  
        self.build_group_rank_map(df)
        # 对index进行归一化，后续替换原index生成的静态协变量
        group_col_scale = rank_group_column + "_scale"   
        df[group_col_scale] = (df[rank_group_column].astype(int) - df[rank_group_column].astype(int).min()) / (df[rank_group_column].astype(int).max() - df[rank_group_column].astype(int).min())
        # Sort
        df = df.sort_values(by=["instrument","datetime_number"],ascending=True)
        return df    
    
    