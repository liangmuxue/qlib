from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from data_extract.his_data_extractor import FutureExtractor,PeriodType,MarketType,get_period_name

import os
from pathlib import Path
import csv
from sqlalchemy import create_engine
import warnings
from pandas.errors import SettingWithCopyWarning

import numpy as np
import pandas as pd
import datetime
import time
import sqlalchemy
from cus_utils.string_util import find_first_digit_position
from trader.utils.date_util import get_tradedays_dur,date_string_transfer,get_tradedays,get_next_month

from cus_utils.log_util import AppLogger
from backtrader.utils import date
from pydantic.types import conint
logger = AppLogger()

_FUTURE_REAL1MIN_FIELD_NAMES = [
    'datetime','code', 'open', 'close', 'high', 'low', 'volume', 'hold','settle'
]

class JuejinFuturesExtractor(FutureExtractor):
    """掘金期货数据源"""

    def __init__(self, backend_channel="juejin",savepath=None,sim_path=None,**kwargs):
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
        self.busi_columns = ["datetime","open","high","low","close","volume","hold","settle"]
        self.col_data_types = {"symbol":str,"open":float,"high":float,"low":float,"close":float,
                               "volume":float,"hold":float}        
        self.sim_path = sim_path
        self.folder_mapping = self.build_main1m_folder_mapping()

    def get_instrument_code_from_contract_code(self,contract_code):
        """根据合约编码，取得品种代码"""
        
        # 取得第一个数字的位置，并截取前面的字符串作为品种代码
        dig_pos = find_first_digit_position(contract_code)
        instrument = contract_code[:dig_pos]
        return instrument
        
    def load_sim_data(self,simdata_date,dataset=None,folder_name=None,contract_name=None):
        """从存储中加载对应的主力合约数据"""
        
        data_path = self.sim_path
        sim_data = None
        begin = int(simdata_date[0].strftime('%y%m'))
        end = int(simdata_date[1].strftime('%y%m'))
        # 筛选主力合约数据，只获取数据集中具备的品种
        if dataset is not None and hasattr(dataset,"df_all"):
            instrument_arr = dataset.df_all['instrument'].unique()
        else:
            instrument_arr = None
            
        if contract_name is not None:
            s_name = self.get_instrument_code_from_contract_code(contract_name)
            filepath = os.path.join(data_path,folder_name,s_name,"{}.{}.csv".format(contract_name,folder_name))
            item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
            # 填入品种编码
            item_df["v_symbol"] = s_name
            item_df["symbol"] = contract_name        
            sim_data = item_df  
        else:
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
                    if instrument_arr is not None and not np.any(instrument_arr==s_name):
                        continue
                    filepath = file
                    item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
                    # 填入品种编码
                    item_df["v_symbol"] = s_name
                    item_df["symbol"] = base_name
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
            
    def get_time_data_by_symbol_and_day(self,symbol,day,contract_name=None,period=PeriodType.MIN1.value):
        """取得指定品种和对应日期的分时交易记录"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            date = datetime.datetime.strptime(str(day), '%Y%m%d')
            # 根据日期和品种名称，取得可能的合约名称,使用合约名称和日期进行筛选
            if contract_name is not None:
                contract_names = [contract_name]
            else:
                contract_names = self.get_likely_main_contract_names(symbol,date)
            item_df = sim_data[(sim_data['symbol'].isin(contract_names))&(sim_data['datetime'].dt.strftime('%Y%m%d')==str(day))]
            return item_df
        
        return None
    
    def get_time_data_by_day(self,contract_names,day,contract_name=None,period=PeriodType.MIN1.value):
        """取得指定品种和对应日期的分时交易记录"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            item_df = sim_data[(sim_data['symbol']==contract_names)&(sim_data['datetime'].dt.strftime('%Y%m%d')==str(day))]
            return item_df
        
        return None

    def get_time_data_by_day_sql(self,day,symbol):
        """取得指定品种和对应日期的分时交易记录,注意是从交错记录表中获取"""

        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        
        column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
        item_sql = "select {} from dominant_real_data_1min_cross where code='{}' " \
            "and Date(datetime)='{}'".format(column_str,symbol,date_string_transfer(day))     
        SQL_Query = pd.read_sql_query(item_sql, engine.connect())
        item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_REAL1MIN_FIELD_NAMES)
        return item_data
    
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

    def load_item_by_time(self,order_book_id,datetime,period=PeriodType.MIN5.value):
        """加载单个品种指定时间的数据"""
        
        if period==PeriodType.MIN1.value:
            sim_data = self.sim_data
            item_df = sim_data[(sim_data['symbol']==order_book_id)&(sim_data['datetime']==datetime)]
        else:
            item_df = None
        return item_df
  
    ####################### 数据导入部分 #########################################

    def import_main_his_data_local(self,date_range=[2201,2512]):
        """从本地导入主力合约历史行情数据"""
        
        data_path = self.sim_path
        list_dir = os.listdir(data_path)
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%y%m')
        end_date = datetime.datetime.strptime(str(date_range[1]), '%y%m')
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
        dtype = {
            'symbol': sqlalchemy.String,
            "datetime": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
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
        # 使用收盘价作为结算价
        upt_sql = "update dominant_real_data set settle=close"        
        self.dbaccessor.do_updateto(upt_sql)    

    def import_main_his_data_part(self,date_range=[200501,202512]):
        """导入所有合约历史日行情(分日盘阶段)数据"""
        
        data_path = self.sim_path
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%Y%m')
        end_date = datetime.datetime.strptime(str(date_range[1]), '%Y%m')
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
        dtype = {
            'code': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'type': sqlalchemy.INT, 
        }      
        columns = list(dtype.keys())
        def _build_type_flag(dt):
            if dt.hour>=9 and dt.hour<12:
                return 1
            if dt.hour>=12 and dt.hour<18:
                return 2            
            return 3
            
        def _apply_combine(group_data,columns=None):
            if group_data.shape[0]==1:
                return group_data
            code = group_data.iloc[0]['code']
            date = group_data.iloc[0]['date'].date()
            open_price = group_data.iloc[0]['open']
            close = group_data.iloc[-1]['close']
            high = group_data['high'].max()
            low = group_data['low'].min()
            volume = group_data['volume'].astype(float).sum()
            k_type = group_data.iloc[0]['type']
            rtn_data = pd.DataFrame(np.array([[code,date,open_price,close,high,low,volume,k_type]]),columns=columns)
            return rtn_data
    
        # 循环取得所有数据文件并获取对应数据
        for p in Path(data_path).iterdir():
            for file in p.rglob('*.csv'):  
                base_name = file.name.split('.')[0]
                # if base_name!="AL2210":
                #     continue
                # 去掉4位后缀，就是合约编码
                s_name = base_name[:-4]
                # 只使用指定日期内的数据
                date_name = base_name[-4:]
                date_name_compare = str(datetime.datetime.strptime(str(date_name),"%y%m").year)[:2] + date_name
                if int(date_name_compare)<date_range[0] or int(date_name_compare)>date_range[1]:
                    continue
                filepath = file
                item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
                if item_df.shape[0]==0:
                    continue
                item_df = item_df.drop('money', axis=1)
                item_df = item_df.rename(columns={"open_interest": "hold"})
                item_df['code'] = base_name
                # 打日盘类型标记
                item_df['type'] = item_df['date'].apply(_build_type_flag)      
                # 通过手动增加夜盘时间的方式，保证夜盘和早盘在同一日期
                item_df.loc[item_df['type']==3,'date'] = item_df[item_df['type']==3]['date'] + pd.Timedelta(hours=3)
                # 分组汇总取得每日的早、午、晚盘数据
                gp = item_df.groupby([item_df['date'].dt.date,'type'])
                agg_data = gp.apply(_apply_combine,columns=columns).reset_index(drop=True, inplace=False)  
                # 有可能第二天是非工作日，还需要进一步校准
                tradedays_begin = item_df['date'].min().date()
                tradedays_end = item_df['date'].max().date()                
                tradedays = get_tradedays(tradedays_begin,tradedays_end)
                agg_data = self.adj_agg_part_data(agg_data, tradedays)     
                # 清洗无效数据
                agg_data = agg_data[agg_data['date'].astype(str).str.len()==10]   
                agg_data = agg_data[agg_data['volume'].astype(float)>0]                                              
                agg_data.to_sql('dominant_real_data_part', engine, index=False, if_exists='append',dtype=dtype)
                print("import_main_his_data_part {} ok,shape:{}".format(filepath,item_df.shape[0]))
                
        # 关联字段挂接
        upt_sql = "update dominant_real_data_part d set d.var_id=(select id from trading_variety t where t.code=LEFT(d.code, LENGTH(d.code)-4))"
        self.dbaccessor.do_updateto(upt_sql)   
        # 使用收盘价作为结算价
        upt_sql = "update dominant_real_data_part set settle=close"        
        self.dbaccessor.do_updateto(upt_sql)    

    def import_main_his_data_part_from_1min_data(self,date_range=[20250912,20251012]):
        """从1分钟数据，导入所有合约历史日行情(分日盘阶段)数据"""
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database),
            pool_size=30,
            max_overflow=50)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
        dtype = {
            'code': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'type': sqlalchemy.INT, 
        }     
        dtype_1min = {
            'code': sqlalchemy.String,
            "datetime": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
        }          
        columns = list(dtype.keys())
        columns_1min = list(dtype_1min.keys())
        columns_str = ",".join(columns_1min)
        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)
        instrument_arr = clean_data.values[:,0]
            
        begin_date = date_range[0]
        end_date = date_range[1]
        begin_date_str = date_string_transfer(str(begin_date))
        end_date_str = date_string_transfer(str(end_date))
        # 前置一天，用于匹配前一天的午盘数据
        before_begin_date_str = get_tradedays_dur(str(begin_date),-1).strftime("%Y%m%d")
        tradedays = get_tradedays(str(begin_date),str(end_date))
        
        def _build_type_flag(dt):
            if dt.hour>=9 and dt.hour<12:
                return 1
            if dt.hour>=12 and dt.hour<18:
                return 2            
            return 3
            
        def _apply_combine(group_data,columns=None):
            if group_data.shape[0]==1:
                return group_data
            code = group_data.iloc[0]['code']
            date = group_data.iloc[0]['date'].date()
            open_price = group_data.iloc[0]['open']
            close = group_data.iloc[-1]['close']
            high = group_data['high'].max()
            low = group_data['low'].min()
            volume = group_data['volume'].astype(float).sum()
            k_type = group_data.iloc[0]['type']
            rtn_data = pd.DataFrame(np.array([[code,date,open_price,close,high,low,volume,k_type]]),columns=columns)
            return rtn_data
        
        # 清空之前多余的数据,由于需要错位数据，因此分几个部分进行删除
        del_sql = "delete from dominant_real_data_part where date>='{}' and date<'{}'".format(begin_date_str,end_date_str)
        self.dbaccessor.do_updateto(del_sql) 
        # 删除前一天下午盘的数据
        del_sql = "delete from dominant_real_data_part where date='{}' and type=2".format(before_begin_date_str)
        self.dbaccessor.do_updateto(del_sql) 
        # 删除当天晚盘和早盘的数据
        del_sql = "delete from dominant_real_data_part where date='{}' and type in(1,3)".format(end_date_str)
        self.dbaccessor.do_updateto(del_sql)                
        # 循环取得所有数据文件并获取对应数据
        for instrument in instrument_arr:
            # if instrument!="AL":
            #     continue
            agg_data_total = []
            # 按照日期轮询
            for index,day in enumerate(tradedays):
                # 需要进行跨天的查询
                if index>0:
                    prev_day = tradedays[index-1]
                else:
                    prev_day = get_tradedays_dur(day,-1).strftime("%Y%m%d")
                # 由于交错模式需要从中午开始，因此取数从前一天下午开盘到当天上午收盘之间，作为一个交易日
                begin_time = date_string_transfer(prev_day) + " 13:00:00"
                end_time = date_string_transfer(day) + " 12:00:00"                
                g_sql = "select distinct code from dominant_real_data_1min_cross where code like '{}____' and datetime>='{}' and datetime<='{}'".format(instrument,begin_time,end_time)
                code_list = self.dbaccessor.do_query(g_sql)
                if len(code_list)==0:
                    logger.warning("g_sql no result:{}".format(g_sql))
                    continue
                code_list = [item[0] for item in code_list]
                for code in code_list:
                    sql = "select {} from dominant_real_data_1min_cross where code='{}' and datetime>='{}' and datetime<='{}'".format(columns_str,code,begin_time,end_time)
                    item_df = pd.read_sql(sql,engine.connect(),parse_dates=['datetime'])  
                    item_df = item_df.rename(columns={'datetime': 'date'})
                    if item_df.shape[0]==0:
                        continue
                    # 打日盘类型标记
                    item_df['type'] = item_df['date'].apply(_build_type_flag)    
                    # 通过手动增加夜盘时间的方式，保证夜盘和早盘在同一日期
                    item_df.loc[item_df['type']==3,'date'] = item_df[item_df['type']==3]['date'] + pd.Timedelta(hours=3)    
                    # 分组汇总取得每日的早、午、晚盘数据
                    gp = item_df.groupby([item_df['date'].dt.date,'type'])
                    agg_data = gp.apply(_apply_combine,columns=columns).reset_index(drop=True, inplace=False)
                    # 清洗无效数据
                    agg_data = agg_data[agg_data['date'].astype(str).str.len()==10]   
                    agg_data = agg_data.groupby("date").apply(
                        lambda x: x.sort_values(by='volume', ascending=False).drop_duplicates(subset=['type'],keep='first'))  
                    agg_data.reset_index(drop=True, inplace=True)  
                    agg_data_total.append(agg_data)    
            agg_data_total = pd.concat(agg_data_total)
            # 有可能第二天是非工作日，还需要进一步校准
            agg_data_total = self.adj_agg_part_data(agg_data_total, tradedays)
            agg_data_total.to_sql('dominant_real_data_part', engine, index=False, if_exists='append',dtype=dtype) 
            logger.info("import_main_his_data_part_from_1min_data,{} to sql ok".format(instrument))               
    
    def rebuild_part_data(self):
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database),
            pool_size=30,
            max_overflow=50)    
        dtype = {
            'code': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'type': sqlalchemy.INT, 
        }  
        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)
        instrument_arr = clean_data.values[:,0]        
        for instrument in instrument_arr:
            # if instrument!="AL":
            #     continue            
            sql = "select code,date,open,close,high,low,volume,type from dominant_real_data_part_copy where code like '{}____'".format(instrument)
            agg_df = pd.read_sql(sql,engine.connect(),parse_dates=['date'])  
            tradedays_begin = agg_df['date'].min().date()
            tradedays_end = agg_df['date'].max().date()
            tradedays = get_tradedays(tradedays_begin,tradedays_end)
            for code,agg_item_df in agg_df.groupby("code"):
                # if not code=="AL2510":
                #     continue
                agg_adj_df = self.adj_agg_part_data(agg_item_df,tradedays)
                # 清洗无效数据
                agg_adj_df = agg_adj_df[agg_adj_df['date'].astype(str).str.len()==10]   
                agg_adj_df = agg_adj_df[agg_adj_df['volume'].astype(float)>0]     
                agg_adj_df = agg_adj_df.groupby("date").apply(
                    lambda x: x.sort_values(by='volume', ascending=False).drop_duplicates(subset=['type'],keep='first'))  
                agg_adj_df.reset_index(drop=True, inplace=True)     
                agg_adj_df.to_sql('dominant_real_data_part', engine, index=False, if_exists='append',dtype=dtype) 
            logger.info("rebuild_part_data,{} to sql ok".format(instrument))    
            
    def adj_agg_part_data(self,agg_data,tradedays):
        
        # 对非工作日数据部分进行校准
        agg_data_night_notin_workingday = agg_data[(agg_data['type']==3)
                &(~pd.to_datetime(agg_data['date']).dt.strftime("%Y%m%d").isin(tradedays))]  
        agg_data_night_in_workingday = agg_data[(agg_data['type']==3)
                &(pd.to_datetime(agg_data['date']).dt.strftime("%Y%m%d").isin(tradedays))]  
        agg_data_day = agg_data[(agg_data['type']!=3)]
        
        def change_date(date):    
            new_date = get_tradedays_dur(date,1).strftime("%Y-%m-%d")
            return new_date
             
        agg_data_night_notin_workingday.loc[agg_data_night_notin_workingday['type']>0,'date'] = agg_data_night_notin_workingday['date'].apply(change_date)
        return pd.concat([agg_data_day,agg_data_night_in_workingday,agg_data_night_notin_workingday])
                   
    def import_continues_data_local_by_main(self,date_range=[201301,202512]):
        """参考主力合约，从本地导入主力连续历史行情数据"""
        
        dtype = {
            'code': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }    
        d_list = list(dtype.keys())
        column_str = ','.join([str(i) for i in d_list])

        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
                
        ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
        # 从品种列表文件中选取需要导入的品种
        ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
        sql = "select distinct(date) from dominant_real_data"
        results = self.dbaccessor.do_query(sql)
        date_list = [item[0] for item in results]
        # 循环取得对应品种，并根据逻辑导入对应的主力合约数据
        with engine.begin() as conn:
            for row in ins_data.values:
                symbol = row[0]
                combine_data = []
                for date in date_list:
                    contract_names = self.get_likely_main_contract_names(symbol,date)
                    contract_names_str = "('" + "','".join([i for i in contract_names if i is not None]) + "')"
                    c_sql = "select {} from dominant_real_data where date='{}' and code in {}".format(column_str,date,contract_names_str)
                    SQL_Query = pd.read_sql_query(c_sql, conn)
                    item_data = pd.DataFrame(SQL_Query, columns=d_list)   
                    if item_data.shape[0]==0:
                        continue 
                    max_idx = np.argmax(item_data['volume'].values)        
                    # 选择成交量最多的作为主力合约数据源
                    main_contract_data = item_data.iloc[max_idx]
                    # 编码转换为连续品种编码
                    main_contract_data = main_contract_data.replace(to_replace='code',value=main_contract_data['code'][:-4])
                    combine_data.append(main_contract_data.values)
                    
                if len(combine_data)==0:
                    continue
                combine_data = pd.DataFrame(np.stack(combine_data),columns=item_data.columns)        
                combine_data.to_sql('dominant_continues_data', engine, index=False, if_exists='append',dtype=dtype)
                print("{} ok".format(symbol))
            
        # # 关联字段挂接
        # upt_sql = "update dominant_continues_data d set d.var_id=(select id from trading_variety t where t.code=LEFT(d.code, LENGTH(d.code)-4))"
        # self.dbaccessor.do_updateto(upt_sql)   
        # # 使用收盘价作为结算价
        # upt_sql = "update dominant_continues_data set settle=close"        
        # self.dbaccessor.do_updateto(upt_sql)   

    def _compute_section_trading_data(self,section_trading,symbol=None,date_str=None,columns=None):
        if section_trading.shape[0]==0:
            print("ggg")
        max_index = section_trading['datetime'].idxmax()
        min_index = section_trading['datetime'].idxmin()
        max_row = section_trading.loc[max_index]
        min_row = section_trading.loc[min_index]
        open_price = min_row['open']
        close = max_row['close']
        high = section_trading['high'].max()
        low = section_trading['low'].min()
        volume = section_trading['volume'].sum()
        settle = close
        rtn_data = pd.DataFrame(np.array([[symbol,date_str,open_price,close,high,low,volume,settle]]),columns=columns)
        return rtn_data
                
    def _apply_compute(self,group_data,columns=None):
        date_str = group_data.iloc[0]['datetime'].date().strftime("%Y-%m-%d")
        symbol = group_data.iloc[0]['code']
        specific_time_str = date_str + " 12:00:00"
        specific_time = pd.to_datetime(specific_time_str)
        early_trading = group_data[group_data['datetime']<specific_time]
        if early_trading.shape[0]>0:
            early_data = self._compute_section_trading_data(early_trading,symbol=symbol,date_str=date_str,columns=columns)
            early_data['type'] = 'early'
        else:
            early_data = None
        late_trading = group_data[group_data['datetime']>specific_time]
        if late_trading.shape[0]>0:
            late_data = self._compute_section_trading_data(late_trading,symbol=symbol,date_str=date_str,columns=columns)
            late_data['type'] = 'late'
        else:
            late_data = None            
        if late_data is None and early_data is None:
            rtn_data = pd.DataFrame(np.array([[]]),columns=columns)
        elif late_data is None:
            rtn_data = early_data
        elif early_data is None:
            rtn_data = late_data
        else:
            rtn_data = pd.concat([early_data,late_data])               
        return rtn_data

    def _apply_combine(self,group_data,columns=None):
        if group_data.shape[0]==1:
            return group_data
        symbol = group_data.iloc[0]['symbol']
        date_str = group_data.iloc[0]['date']
        open_price = group_data[group_data['type']=='late']['open'].values[0]
        close = group_data[group_data['type']=='early']['close'].values[0]
        high = group_data['high'].max()
        low = group_data['low'].min()
        volume = group_data['volume'].astype(float).sum()
        settle = close
        rtn_data = pd.DataFrame(np.array([[symbol,date_str,open_price,close,high,low,volume,settle]]),columns=columns)
        return rtn_data
    
    def import_continues_cross_from_his_data_part(self,date_range=[20050701,20251010]):
        """从分片合约表中取得数据，导入到历史数据日K表（交错表）"""

        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)
        instrument_arr = clean_data.values[:,0]    
          
        begin_date = date_range[0]
        end_date = date_range[1]
        begin_date_str = date_string_transfer(str(begin_date))
        end_date_str = date_string_transfer(str(end_date))
        # 后置一天，用于匹配下一天的午盘数据作为当天后一段数据
        next_end_date = get_tradedays_dur(str(end_date),1).strftime("%Y%m%d")
        tradedays = get_tradedays(str(begin_date),str(next_end_date))
                
        for instrument in instrument_arr:
            # if instrument<="EG":
            #     continue            
            for day in tradedays:
                day_str = date_string_transfer(day)
                # 获取主力合约
                main_sql = "select code,sum(volume),max(type),max(high),min(low) from dominant_real_data_part where " \
                    "date='{}' and code like '{}____' group by code".format(day_str,instrument)
                result = self.dbaccessor.do_query(main_sql)
                if len(result)==0:
                    # logger.warning("{} has no data".format(instrument))
                    continue
                result = np.array(result)
                sort_seq = np.argsort(-result[:,1].astype(float))
                main_item = result[sort_seq[0]]
                main_code = main_item[0]
                volume = float(main_item[1])
                type = int(main_item[2])
                high = float(main_item[3])
                low = float(main_item[4])
                # 由于是交错模式，因此还需要取得第二天的数据
                next_date = get_tradedays_dur(str(day),1).strftime("%Y-%m-%d")
                # 开盘价同一从第一天的下午开盘取
                open_sql = "select open from dominant_real_data_part where date='{}' and code='{}' and type=2".format(day_str,main_code)
                open_result = self.dbaccessor.do_query(open_sql)
                if len(open_result)==0:
                    continue
                open_price = open_result[0][0]
                # 收盘价从第二天早盘收盘价中取
                close_sql = "select close from dominant_real_data_part where date='{}' and code='{}' and type=1".format(next_date,main_code) 
                close_result = self.dbaccessor.do_query(close_sql)
                if len(close_result)==0:
                    if len(result)<=1:
                        logger.warning("close_result no second data:{}_{}".format(day,instrument))
                        continue
                    # 主力合约切换时,需要使用第二主力合约
                    sec_item = result[sort_seq[1]]                    
                    close_sql = "select close from dominant_real_data_part where date='{}' and code='{}' and type=1".format(next_date,sec_item[0]) 
                    close_result = self.dbaccessor.do_query(close_sql)
                    next_main_code = sec_item[0]
                    if len(close_result)==0:
                        logger.warning("close_result no data:{}".format(close_sql))
                        continue
                else:
                    next_main_code = main_code
                close_price = close_result[0][0]   
                # 合并2天的数据，并计算最高最低和成交量
                combine_sql = "select high,low,volume from dominant_real_data_part where date='{}' and code='{}' and type=2" \
                    " union (select high,low,volume from dominant_real_data_part where date='{}' and code='{}' and type in (1,3))" \
                    .format(day_str,main_code,next_date,next_main_code)
                combine_result = np.array(self.dbaccessor.do_query(combine_sql))
                high = combine_result[:,0].astype(float).max()
                low = combine_result[:,1].astype(float).min()  
                volume = combine_result[:,2].astype(float).sum()      
                insert_sql = "insert into dominant_continues_data_cross(code,date,open,close,high,low,volume,main_code) values('{}','{}',{},{},{},{},{},'{}') " \
                    .format(instrument,day_str,open_price,close_price,high,low,volume,main_code)
                self.dbaccessor.do_inserto(insert_sql)
            logger.info("import_continues_from_his_data_part {} ok".format(instrument))
        upt_sql = "update dominant_continues_data_cross set hold=0,settle=close"
        self.dbaccessor.do_updateto(upt_sql)
        
    def import_continues_from_1min_data_cross(self,date_range=[200501,202512]):
        """通过1分钟连续合约数据，导入到主力连续合约日历史行情数据,注意是从午盘后错位导入"""

        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)
        instrument_arr = clean_data.values[:,0]
                
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'symbol': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }      
        columns = list(dtype.keys())
        del columns[-2]
        dtype_sql = dtype.copy()
        dtype_sql['code'] = dtype.pop('symbol')
        
        continues_1min_file_path = "/home/qdata/futures_data/juejin/continues_1min/"
        continues_file_path = "/home/qdata/futures_data/juejin/continues_day/"

        # 循环取得所有数据文件并获取对应数据
        for instrument in instrument_arr:
            # if instrument!="EB":
            #     continue
            sql = "select count(*) from dominant_continues_data_cross where code='{}'".format(instrument)
            cnt = self.dbaccessor.do_query(sql)[0][0]
            if cnt>0:
                continue
            exchange_code = self.get_exchange_from_instrument(instrument)
            file_name = "{}9999.{}.csv".format(instrument, exchange_code)
            data_path = os.path.join(continues_1min_file_path,file_name)
            if not os.path.exists(data_path):
                logger.warning("file path not exists:{}".format(data_path))
                continue
            item_df = pd.read_csv(data_path,parse_dates=['date'])  
            # 分组汇总取得早盘的开盘和收盘，以及午盘的开盘和收盘，并进行错位计算
            gp = item_df.groupby(item_df['date'].dt.date)
            agg_data = gp.apply(self._apply_compute,columns=columns).reset_index(drop=True, inplace=False)
            early_data = agg_data[agg_data['type']=='early']
            late_data = agg_data[agg_data['type']=='late']
            late_data['date'] = late_data['date'].astype(str)
            # 把早盘日期前推一天，并和午盘（及晚盘）合并
            early_data_cross = []
            for index,row in early_data.iterrows():
                date = row['date']
                prev_date = get_tradedays_dur(date,-1)
                row['date'] = prev_date
                early_data_cross.append(row.values)
            early_data_cross = pd.DataFrame(np.array(early_data_cross),columns=early_data.columns)
            early_data_cross['date'] = early_data_cross['date'].astype(str)
            data_cross = pd.concat([late_data,early_data_cross])
            data_cross = data_cross.groupby("date").apply(self._apply_combine,columns=columns).reset_index(drop=True, inplace=False).drop(columns=['type'])
            data_cross['hold'] = 0
            data_cross['code'] = instrument
            data_cross['date'] = data_cross['date'].astype(str)
            data_cross['main_code'] = data_cross['symbol']
            data_cross = data_cross.drop(columns=['symbol'])
            data_cross.to_sql('dominant_continues_data_cross', engine, index=False, if_exists='append',dtype=dtype_sql)
            save_path = os.path.join(continues_file_path,"{}.csv".format(instrument))
            # data_cross.to_csv(save_path)
            logger.info("{} ok,shape:{}".format(instrument,data_cross.shape[0]))
    
    def import_continues_cross_from_main_1min_data(self,date_range=[20250701,20250927]):
        """通过1分钟主力合约数据，导入到连续合约日历史行情数据,注意是从午盘后错位导入"""
        
        warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
                
        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)
        instrument_arr = clean_data.values[:,0]
                
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'symbol': sqlalchemy.String,
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }      
        columns = list(dtype.keys())
        del columns[-2]
        dtype_sql = dtype.copy()
        dtype_sql['code'] = dtype.pop('symbol')
        
        begin_date = date_range[0]
        end_date = date_range[1]
        begin_date_str = date_string_transfer(str(begin_date))
        end_date_str = date_string_transfer(str(end_date))
        # 后置一天，用于匹配下一天的午盘数据作为当天后一段数据
        next_end_date = get_tradedays_dur(str(end_date),1).strftime("%Y%m%d")
        tradedays = get_tradedays(str(begin_date),str(next_end_date))
        from cus_utils.data_aug import compare_clean_data_and_continus_data
        lack_data = compare_clean_data_and_continus_data(match_date=20250925) 
        # 循环取得所有数据文件并获取对应数据
        for instrument in instrument_arr:
            # if instrument!="WR":
            #     continue
            # if np.all(lack_data['code']!=instrument):
            #     continue       
            # 清空之前多余的数据
            sql = "delete from dominant_continues_data_cross where code='{}' and date>='{}' and date<='{}'".format(instrument,begin_date_str,end_date_str)
            self.dbaccessor.do_updateto(sql)
            data_cross_total = []
            # 按照日期轮询
            for day in tradedays:
                # 从主力一分钟数据中选取
                symbol = self.get_main_contract_name(instrument,day,use_1min_data=True)
                next_date = get_tradedays_dur(day,1).strftime("%Y%m%d")
                item_df = self.get_time_data_by_day_sql(day, symbol)
                # 有可能由于日期切换，发生主力合约切换，因此使用cross数据再次进行主力合约匹配
                symbol_next = self.get_main_contract_name(instrument,next_date,use_1min_data=True)
                item_df_next = self.get_time_data_by_day_sql(next_date, symbol_next)
                if item_df.shape[0]==0 or item_df_next.shape[0]==0:
                    continue
                # 分别从前后2天取得早盘的开盘和收盘，以及午盘的开盘和收盘，并进行错位计算
                gp = item_df.groupby(item_df['datetime'].dt.date)
                agg_data = gp.apply(self._apply_compute,columns=columns).reset_index(drop=True, inplace=False)
                late_data = agg_data[agg_data['type']=='late']
                late_data['date'] = late_data['date'].astype(str)                
                gp = item_df_next.groupby(item_df['datetime'].dt.date)
                agg_data = gp.apply(self._apply_compute,columns=columns).reset_index(drop=True, inplace=False)
                early_data = agg_data[agg_data['type']=='early']
                early_data['date'] = date_string_transfer(day)  
                data_cross = pd.concat([late_data,early_data])
                # 直接合并2段数据
                data_cross = self._apply_combine(data_cross,columns=columns).reset_index(drop=True, inplace=False)
                data_cross['hold'] = 0
                data_cross['code'] = instrument
                data_cross['date'] = data_cross['date'].astype(str)
                data_cross['main_code'] = data_cross['symbol']
                data_cross = data_cross.drop(columns=['symbol'])
                # logger.info("{} single ok,date:{}".format(instrument,day)) 
                data_cross.to_sql('dominant_continues_data_cross', engine, index=False, if_exists='append',dtype=dtype_sql)
            logger.info("{} to sql ok".format(instrument))        

    def import_main_1min_data_local(self,date_range=[202201,202512]):
        """从本地导入主力合约历史1分钟行情数据"""
        
        data_path = self.sim_path
        list_dir = os.listdir(data_path)
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%Y%m')
        end_date = datetime.datetime.strptime(str(date_range[1]), '%Y%m')
        
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
            'settle': sqlalchemy.FLOAT, 
        }      

        data_path = self.sim_path
        sim_data = None
        # 循环取得所有数据文件并获取对应数据
        for p in Path(data_path).iterdir():
            for file in p.rglob('*.csv'):  
                base_name = file.name.split('.')[0]
                # 去掉4位后缀，就是合约编码
                s_name = base_name[:-4]
                # 只使用指定日期内的数据
                date_name = base_name[-4:]
                date_name_compare = "20" + date_name
                if int(date_name_compare)<date_range[0] or int(date_name_compare)>date_range[1]:
                    continue
                filepath = file
                item_df = pd.read_csv(filepath,dtype=self.col_data_types,parse_dates=['date'])  
                item_df = item_df.drop('money', axis=1)
                item_df = item_df.rename(columns={"open_interest": "hold"})
                item_df = item_df.rename(columns={"date": "datetime"})
                item_df['code'] = base_name
                item_df.to_sql('dominant_real_data_1min', engine, index=False, if_exists='append',dtype=dtype)
                print("{} ok,shape:{}".format(filepath,item_df.shape[0]))
        # 关联字段挂接
        # upt_sql = "update dominant_real_data_1min d set d.var_id=(select id from trading_variety t where t.code=LEFT(d.code, LENGTH(d.code)-4))"
        # self.dbaccessor.do_updateto(upt_sql)   
        # 使用收盘价作为结算价
        upt_sql = "update dominant_real_data_1min set settle=close"        
        self.dbaccessor.do_updateto(upt_sql)   
                
    def build_main1m_folder_mapping(self,):
        """生成主力1分钟数据目录与品种的映射关系"""
        
        mappings = []
        list_dir = os.listdir(self.sim_path)
        for file in list_dir:
            folder_name = file
            folder = os.path.join(self.sim_path,folder_name)
            sub_list_dir = os.listdir(folder)
            for sub_file in sub_list_dir:  
                mappings.append([folder_name,sub_file])
        
        mappings = pd.DataFrame(np.array(mappings),columns=["cate","instrument"])
        return mappings

if __name__ == "__main__":    
    
    extractor = JuejinFuturesExtractor(savepath="/home/qdata/futures_data",sim_path="/home/qdata/futures_data/juejin/main_1min")   
    save_path = "custom/data/results/futures"
    
    # 导入主力合约历史日行情数据
    # extractor.import_main_his_data_local()
    # extractor.import_main_his_data_part()
    # extractor.import_main_his_data_part_from_1min_data()
    # extractor.rebuild_part_data()
    # 导入主力合约1分钟历史行情数据
    # extractor.import_main_1min_data_local(date_range=[202401,202512])
    # 导入主力连续合约日K数据,错位导入
    extractor.import_continues_cross_from_his_data_part()
    # extractor.import_continues_cross_from_main_1min_data()
    # 根据主力合约数据，导入主力连续日行情数据
    # extractor.import_continues_data_local_by_main()
    # begin = datetime.datetime.strptime("20220501", "%Y%m%d").date()
    # end = datetime.datetime.strptime("20220701", "%Y%m%d").date()
    # simdata_date = [begin,end]
    # ins_name = "CJ"
    # contract_name = 'CJ2207'
    # folder_name = extractor.folder_mapping[extractor.folder_mapping['instrument']==ins_name]['cate'].values[0]
    # symbol_name = contract_name[:-4]
    # extractor.load_sim_data(simdata_date,folder_name=folder_name,contract_name=contract_name)
    # date = datetime.datetime.strptime("20220505 09:01:00", "%Y%m%d %H:%M:%S")
    # # data = extractor.get_time_data(contract_name,date)
    # data = extractor.get_time_data_by_symbol_and_day(ins_name,20220505,contract_name=contract_name)
    # print("data len:{}".format(data.shape[0]))
            