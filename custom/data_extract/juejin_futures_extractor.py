from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from trader.utils.date_util import get_previous_month,get_next_month
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType

import os
from pathlib import Path
import csv
from sqlalchemy import create_engine

from akshare.stock_feature.stock_hist_em import (
    code_id_map_em
)
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import datetime
import time
import sqlalchemy

from cus_utils.log_util import AppLogger
from data_extract.his_data_extractor import HisDataExtractor,get_period_name

class JuejinFuturesExtractor(HisDataExtractor):
    """掘金期货数据源"""

    def __init__(self, backend_channel="juejin",savepath=None,sim_path=None,**kwargs):
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
        self.busi_columns = ["datetime","open","high","low","close","volume","hold","settle"]
        self.col_data_types = {"symbol":str,"open":float,"high":float,"low":float,"close":float,
                               "volume":float,"hold":float}        
        self.sim_path = sim_path
    
    def load_sim_data(self,simdata_date,dataset=None):
        """从存储中加载对应的主力合约数据"""
        
        data_path = self.sim_path
        sim_data = None
        df_data = dataset.df_all
        begin = int(simdata_date[0].strftime('%y%m'))
        end = int(simdata_date[1].strftime('%y%m'))
        # 筛选主力合约数据，只获取数据集中具备的品种
        instrument_arr = df_data['instrument'].unique()
        # 循环取得所有数据文件并获取对应数据
        for p in Path(data_path).iterdir():
            for file in p.rglob('*.csv'):  
                base_name = file.name.split('.')[0]
                # 去掉4位后缀，就是合约编码
                s_name = base_name[:-4]
                # 只使用指定日期内的数据
                date_name = base_name[-4:]
                if int(date_name)<begin or int(date_name)>end:
                    continue
                # 筛选出名字匹配的品种
                if not np.any(instrument_arr==s_name):
                    continue
                filepath = file
                item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
                item_df["symbol"] = s_name
                if sim_data is None:
                    sim_data = item_df
                else:
                    sim_data = pd.concat([sim_data,item_df])
                    
        sim_data = sim_data.rename(columns={"date":"datetime"})
        # 生成时间戳方便后续时间比较,注意需要提前转换时区
        sim_data['timestamp'] = sim_data['datetime'].dt.tz_localize(tz='Asia/Shanghai').astype(np.int64)//10 ** 9
        self.sim_data = sim_data

    def load_sim_data_continues(self,simdata_date,dataset=None):
        """从存储中加载对应的主力连续合约数据"""
        
        data_path = self.sim_path
        sim_data = None
        df_data = dataset.df_all
        # 筛选主力合约数据，只获取数据集中具备的品种
        instrument_arr = df_data['instrument'].unique()
        # 循环取得所有数据文件并获取对应数据
        list_dir = os.listdir(data_path)
        for file in list_dir:
            base_name = file.split('.')[0]
            # 去掉后缀9999，就是品种名称
            s_name = base_name[:-4]
            # 筛选出名字匹配的品种
            if not np.any(instrument_arr==s_name):
                continue
            filepath = os.path.join(data_path,file)
            item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
            item_df = item_df.rename(columns={"date":"datetime"})
            # 只使用指定日期内的数据
            item_df = item_df[(item_df['datetime']>=pd.to_datetime(simdata_date[0]))&(item_df['datetime']<=pd.to_datetime(simdata_date[1]))]
            if sim_data is None:
                sim_data = item_df
            else:
                sim_data = pd.concat([sim_data,item_df])
        # 生成时间戳方便后续时间比较,注意需要提前转换时区
        sim_data['timestamp'] = sim_data['datetime'].dt.tz_localize(tz='Asia/Shanghai').astype(np.int64)//10 ** 9
        self.sim_data = sim_data
            
    def get_likely_main_contract_names(self,instrument,date):
        """根据品种编码和指定日期，取得可能的主力合约名称"""
        
        #取得 当前月，下个月，下下个月3个月份合约名称
        cur_month = date.strftime('%y%m')
        next_month = get_next_month(date,next=1)
        next_month = next_month.strftime("%y%m")
        next_two_month = get_next_month(date,next=2)
        next_two_month = next_two_month.strftime("%y%m")
        
        return [instrument+cur_month,instrument+next_month,instrument+next_two_month]

    def get_time_data_by_day(self,symbol,day,period=None):
        """取得指定品种和对应日期的分时交易记录"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            date = datetime.datetime.strptime(str(day), '%Y%m%d')
            # 根据日期和品种名称，取得可能的合约名称,使用合约名称和日期进行筛选
            contract_names = self.get_likely_main_contract_names(symbol,date)
            item_df = sim_data[(sim_data['symbol'].isin(contract_names))&(sim_data['datetime'].dt.strftime('%Y%m%d')==str(day))]
            return item_df
        
        return None

    def get_time_data(self,contract_symbol,date,period=PeriodType.MIN1.value):
        """取得指定品种和对应时间的分时交易记录"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            # date = datetime.datetime.strptime(str(date), '%Y%m%d %')
            item_df = sim_data[(sim_data['symbol']==contract_symbol)&(sim_data['timestamp']==int(date.timestamp()))]
            return item_df
        
        return None
            
    def get_whole_item_datapath(self,period,institution=False):
        period_name = get_period_name(period)
        return "{}/item/{}/csv_data".format(self.savepath,period_name)

    def load_item_df(self,instrument,period=PeriodType.MIN5.value,institution=False):
        """加载单个品种"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            item_df = sim_data[(sim_data['symbol']==instrument.order_book_id)]
        else:
            item_df = None
        return item_df
    
    ####################### 数据导入部分 #########################################

    def import_continues_his_data_local(self,date_range=[2201,2412]):
        """从本地导入主力连续历史行情数据"""
        
        data_path = self.sim_path
        list_dir = os.listdir(data_path)
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%y%m')
        end_date = datetime.datetime.strptime(str(date_range[1]), '%y%m')
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'symbol': sqlalchemy.String,
            "datetime": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
        }      
                
        # 从分时数据中累加得到日K数据
        for file in list_dir:
            filepath = os.path.join(data_path,file)
            base_name = file.split('.')[0]
            # 去掉后缀9999，就是品种名称
            s_name = base_name[:-4]
            item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
            # 只使用指定日期内的数据
            item_df = item_df[(item_df['datetime']>=pd.to_datetime(begin_date))&(item_df['datetime']<=pd.to_datetime(end_date))]
            # 按照日期累加，并入库
            df_sum = item_df.groupby(["symbol",item_df.date.dt.date],as_index=True)["volume", "money", "open_interest"].apply(lambda x : x.sum())
            df_open = item_df.groupby(["symbol",item_df.date.dt.date],as_index=True)["open"].first()
            df_close = item_df.groupby(["symbol",item_df.date.dt.date],as_index=True)["close"].last()
            df_high = item_df.groupby(["symbol",item_df.date.dt.date],as_index=True)["high"].max()
            df_low = item_df.groupby(["symbol",item_df.date.dt.date],as_index=True)["low"].min()
            df_concat = pd.concat([df_sum,df_open,df_close,df_high,df_low],axis=1)
            df_concat = df_concat.drop('money', axis=1)
            df_concat = df_concat.reset_index()
            df_concat = df_concat.rename(columns={"open_interest": "hold", "symbol": "code"})
            df_concat.to_sql('dominant_real_data', engine, index=False, if_exists='append',dtype=dtype)
            print("{} ok,shape:{}".format(filepath,df_concat.shape[0]))
        # 关联字段挂接
        upt_sql = "update dominant_real_data d set d.var_id=(select id from trading_variety t where t.code=LEFT(d.code, LENGTH(d.code)-4))"
        self.dbaccessor.do_updateto(upt_sql)                

    def import_main_his_data_local(self,date_range=[2201,2512]):
        """从本地导入主力合约历史行情数据"""
        
        data_path = self.sim_path
        list_dir = os.listdir(data_path)
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%y%m')
        end_date = datetime.datetime.strptime(str(date_range[1]), '%y%m')
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'symbol': sqlalchemy.String,
            "datetime": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
        }      
                
        # 从分时数据中累加得到日K数据
        for p in Path(data_path).iterdir():
            for file in p.rglob('*.csv'):  
                filepath = file
                base_name = file.name.split('.')[0]
                # 去掉后缀\，就是品种名称
                s_name = base_name[:-4]
                item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
                # 只使用指定日期内的数据
                item_df = item_df[(item_df['date']>=pd.to_datetime(begin_date))&(item_df['date']<=pd.to_datetime(end_date))]
                if item_df.shape[0]==0:
                    continue
                # 按照日期累加，并入库
                df_sum = item_df.groupby([item_df.date.dt.date],as_index=True)["volume", "money", "open_interest"].apply(lambda x : x.sum())
                df_open = item_df.groupby([item_df.date.dt.date],as_index=True)["open"].first()
                df_close = item_df.groupby([item_df.date.dt.date],as_index=True)["close"].last()
                df_high = item_df.groupby([item_df.date.dt.date],as_index=True)["high"].max()
                df_low = item_df.groupby([item_df.date.dt.date],as_index=True)["low"].min()
                df_concat = pd.concat([df_sum,df_open,df_close,df_high,df_low],axis=1)
                df_concat = df_concat.drop('money', axis=1)
                df_concat = df_concat.reset_index()
                df_concat = df_concat.rename(columns={"open_interest": "hold"})
                df_concat['code'] = base_name
                df_concat.to_sql('dominant_real_data', engine, index=False, if_exists='append',dtype=dtype)
                print("{} ok,shape:{}".format(filepath,df_concat.shape[0]))
        # 关联字段挂接
        upt_sql = "update dominant_real_data d set d.var_id=(select id from trading_variety t where t.code=LEFT(d.code, LENGTH(d.code)-4))"
        self.dbaccessor.do_updateto(upt_sql)    
        
                    
if __name__ == "__main__":    
    
    extractor = JuejinFuturesExtractor(savepath="/home/qdata/futures_data",sim_path="/home/qdata/futures_data/juejin/main_1min")   
    save_path = "custom/data/results/futures"
    extractor.import_main_his_data_local()
            