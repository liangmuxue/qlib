from data_extract.rqalpha.futures_ds import FuturesDataSource

from datetime import date,timedelta,datetime
import numpy as np
import pandas as pd
import pickle
import os

from rqalpha.environment import Environment

from trader.utils.date_util import get_prev_working_day,date_string_transfer
from cus_utils.db_accessor import DbAccessor
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from data_extract.juejin_futures_extractor import JuejinFuturesExtractor
from data_extract.akshare_extractor import AkExtractor
from trader.rqalpha.dict_mapping import judge_market,transfer_instrument
import cus_utils.global_var as global_var
from data_extract.akshare_futures_extractor import AkFuturesExtractor
from data_extract import akshare_futures_extractor
from gunicorn import instrument

_STOCK_FIELD_NAMES = [
    'datetime', 'open', 'high', 'low', 'close', 'vol', 'amount'
]

_FUTURE_REAL1MIN_FIELD_NAMES = [
    'datetime','code', 'open', 'close', 'high', 'low', 'volume', 'hold','settle'
]
_FUTURE_CONTINUES_FIELD_NAMES = [
    'date','code', 'open', 'close', 'high', 'low', 'volume', 'hold','settle'
]
   
class FuturesDataSourceSql(FuturesDataSource):
    """期货自定义数据源,数据库模式"""
    
    def __init__(self,path,stock_data_path=None,sim_path=None,frequency_sim=True):
        super(FuturesDataSourceSql, self).__init__(path,stock_data_path=stock_data_path,sim_path=sim_path,frequency_sim=frequency_sim)
        self.dbaccessor.get_connection(reuse=True)

    def load_recent_data(self,day):
        """预加载近期数据"""
        
        # 从昨晚9点到今天16点，作为交易时间段加载
        prev_day = get_prev_working_day(day)
        datetime_begin = prev_day.strftime("%Y-%m-%d ") + "21:00:00"
        datetime_end = day.strftime("%Y-%m-%d ") + "16:00:00"
        # 通过创建临时表进行查询
        self.dbaccessor.do_ddl("drop table if exists temp_1min_data")
        sql = "CREATE TEMPORARY TABLE temp_1min_data as select * from dominant_real_data_1min where datetime>='{}' and datetime<='{}'".format(datetime_begin,datetime_end)
        self.dbaccessor.do_ddl(sql)
    
    def has_current_data(self,day,symbol):
        """当日是否开盘交易"""

        # 直接使用sql查询,检查当日是否有分时数据
        item_sql = "select count(1) from dominant_continues_data_1min where code='{}' " \
            "and Date(datetime)='{}'".format(symbol,day.strftime('%Y-%m-%d'))
        cnt = self.dbaccessor.do_query(item_sql)[0][0]      
        if cnt==0:
            return False
        return True

    def get_continue_data_by_day(self,symbol,day):
        """取得指定品种和对应日期的主力连续交易记录"""
        
        column_str = ','.join([str(i) for i in _FUTURE_CONTINUES_FIELD_NAMES])
        item_sql = "select {} from dominant_continues_data where code='{}' " \
            "and Date(date)='{}'".format(column_str,symbol,date_string_transfer(day))     
        SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection())
        item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_CONTINUES_FIELD_NAMES)
        return item_data

    def get_recent_date_by_date(self,symbol,date):
        """取得指定时间最近的交易时间"""
        
        day = date.strftime("%Y%m%d")
        datetime_str = date.strftime("%Y-%m-%d %H:%M:%S")
        column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
        item_sql = "select datetime from dominant_real_data_1min where code='{}' " \
            "and Date(datetime)='{}' and datetime<='{}' order by datetime desc".format(symbol,date_string_transfer(day),datetime_str)     
        SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection())
        item_data = pd.DataFrame(SQL_Query, columns=["datetime"])
        if item_data.shape[0]==0:
            return None
        return item_data['datetime'].dt.to_pydatetime()[0]
       
    def get_time_data_by_day(self,day,symbol):
        """取得指定品种和对应日期的分时交易记录,注意需要加入前一天晚盘"""

        column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
        # 取得前一天晚盘开始时间作为开始时间，当天下午盘收盘作为结束时间
        day_date = datetime.strptime(day, "%Y%m%d")
        prev_day_date = get_prev_working_day(day_date)
        begin_time = prev_day_date.strftime("%Y-%m-%d ") + "21:00:00"
        end_time = date_string_transfer(day) + " 16:00:00"
        item_sql = "select {} from dominant_real_data_1min where code='{}' " \
            "and datetime>='{}' and datetime<='{}'".format(column_str,symbol,begin_time,end_time)     
        SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection())
        item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_REAL1MIN_FIELD_NAMES)
        return item_data

    # def get_prev_price(self,order_book_id,now_dt):
    #     """取得上一分钟交易数据"""
    #
    #     column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
    #     # 从预加载的临时表中获取数据
    #     item_sql = "select {} from temp_1min_data where code='{}' and datetime<'{}' order by datetime desc limit 1".format(column_str,order_book_id,now_dt.strftime("%Y-%m-%d %H:%M:%S"))     
    #     SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection(reuse=True))
    #     item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_REAL1MIN_FIELD_NAMES)
    #     if item_data.shape[0]==0:
    #         return None
    #     price = item_data["close"].values[0]
    #     return price

    def get_current_price(self,order_book_id,now_dt):
        """取得当前分钟交易数据"""
        
        column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
        # 从预加载的临时表中获取数据
        item_sql = "select {} from temp_1min_data where code='{}' and datetime='{}'".format(column_str,order_book_id,now_dt.strftime("%Y-%m-%d %H:%M:%S"))     
        SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection(reuse=True))
        item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_REAL1MIN_FIELD_NAMES)
        if item_data.shape[0]==0:
            return None
        price = item_data["close"].values[0]
        return price
                    
    def get_k_data(self, order_book_id, start_dt, end_dt,frequency=None,need_prev=True):
        """从已下载的文件中，加载K线数据"""
        
        self.time_inject(begin=True)
        # 可以加载不同的频次类型数据
        if frequency=="1m":
            column_str = ','.join([str(i) for i in _FUTURE_REAL1MIN_FIELD_NAMES])
            item_sql = "select {} from dominant_real_data_1min where code='{}' " \
                "and datetime='{}'".format(column_str,order_book_id,start_dt.strftime('%Y-%m-%d %H:%M:%S'))     
            SQL_Query = pd.read_sql_query(item_sql, self.dbaccessor.get_connection())
            item_data = pd.DataFrame(SQL_Query, columns=_FUTURE_REAL1MIN_FIELD_NAMES)
            if item_data is None or item_data.shape[0]==0:
                return None                
            item_data["last"] = item_data["close"]    
            self.time_inject(code_name="load_item_df")   
                       
        # 日线数据使用akshare的数据源
        if frequency=="1d":
            start_dt = datetime(start_dt.year, start_dt.month, start_dt.day).date() 
            end_dt = datetime(end_dt.year, end_dt.month, end_dt.day).date()
            item_data = self.load_item_day_data(order_book_id,start_dt)
            if item_data is None or item_data.shape[0]==0:
                return None               
            # 使用结算价作为当日最终价格
            item_data["last"] = item_data["settle"]    
            # 字段和rq统一
            item_data['symbol'] = item_data['code']        
            item_data = item_data.rename(columns={"date":"datetime"})
        self.time_inject(code_name="dt query,{}".format(frequency))     

        # 取得前一个交易时段收盘
        item_data["prev_close"]= np.NaN
        if need_prev:
            item_data["prev_close"] = self._get_prev_close(order_book_id, start_dt,frequency=frequency)
        # item_data = item_data.iloc[0].to_dict()
        return item_data    
            