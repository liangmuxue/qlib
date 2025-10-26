from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType

import copy
import re
import time
import datetime
import math
import pickle
import os
import csv
from sqlalchemy import create_engine
import sqlalchemy

import akshare as ak

import numpy as np
import pandas as pd
import requests

from trader.utils.date_util import get_prev_working_day,get_next_working_day,get_next_month,is_working_day,date_string_transfer
from cus_utils.log_util import AppLogger
from data_extract.his_data_extractor import FutureExtractor,get_period_name
from data_extract.akshare.futures_daily_bar import futures_hist_em,futures_zh_minute_sina,get_exchange_symbol_map
from data_extract.akshare.futures_daily import get_futures_daily

from cus_utils.log_util import AppLogger
logger = AppLogger()

class AkFuturesExtractor(FutureExtractor):
    """akshare期货数据源"""

    def __init__(self, backend_channel="ak",savepath=None,**kwargs):
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
        self.busi_columns = ["code","date","open","high","low","close","volume","hold","settle"]

    def get_whole_item_datapath(self,period,institution=False):
        period_name = get_period_name(period)
        if institution:
            return "{}/item/{}/institution".format(self.savepath,period_name)
        return "{}/item/{}/csv_data".format(self.savepath,period_name)
      
    #######################   数据加载 ######################################     

          
    #######################   数据导入 ######################################      
    def import_trading_variety(self):
        """导入交易品种数据"""

        exchange_sql = "select id,short_name from futures_exchange"
        result_rows = self.dbaccessor.do_query(exchange_sql)        
        exchange_map = {}
        for result in result_rows:
            exchange_map[result[1]] = result[0]
        
        insert_sql = "insert into trading_variety(exchange_id,code,name,magin_radio,limit_rate,multiplier,price_range) values(%s,%s,%s,%s,%s,%s,%s)"
        # update_sql = "update trading_variety set multiplier={},limit_rate={}, price_range={} where code='{}'"
        futures_rule_df = ak.futures_rule()        
        for idx,row in futures_rule_df.iterrows():
            exchange_id = exchange_map[row['交易所']]
            magin_radio = row['交易保证金比例'] if not math.isnan(row['交易保证金比例']) else 0
            limit_rate = row['涨跌停板幅度'] if not math.isnan(row['涨跌停板幅度']) else 0
            multiplier = row['合约乘数'] if not math.isnan(row['合约乘数']) else 0
            price_range = row['最小变动价位'] if not math.isnan(row['最小变动价位']) else 0
            self.dbaccessor.do_inserto_withparams(insert_sql, (exchange_id,row['代码'],row['品种'],magin_radio,limit_rate,multiplier,price_range))     
            # update_sql_real = update_sql.format(multiplier,limit_rate,price_range,row['代码']) 
            # self.dbaccessor.do_inserto(update_sql_real)  

    def extract_item_data(self,code,start_date=None,end_date=None,period=PeriodType.DAY.value):  
        """取得单个品种的主力连续历史行情数据"""
        
        # AKSHARE目前只支持按照日导入
        if period!=PeriodType.DAY.value:
            raise NotImplementedError
        
        item_data = ak.futures_zh_daily_sina(symbol=code)    
        if item_data is None or item_data.shape[0]==0:
            return None
        
        # 插入编号
        item_data.insert(loc=0, column='code', value=code)
        # 清洗无效数据
        item_data = self.data_clean(item_data)
                 
        return item_data

    def data_clean(self,item_data):
        """数据清洗"""
        
        # 收盘排查
        item_data_clean = item_data[item_data["close"]>0]
        item_data_clean = item_data_clean[item_data_clean["close"]/item_data_clean["open"]-1<0.2]
        # 高低价格排查
        item_data_clean = item_data_clean[(item_data_clean["high"]-item_data_clean["low"])>=0]
        
        return item_data_clean
    
    def extract_main_item_data(self,code,start_date=None,end_date=None,period=PeriodType.DAY.value):  
        """取得单个品种的主力合约历史行情数据"""
        
        # AKSHARE目前只支持按照日导入
        if period!=PeriodType.DAY.value:
            raise NotImplementedError
        try:
            item_data = ak.futures_zh_daily_sina(symbol=code)    
        except Exception as e:
            print("err for:{}".format(code))
            return None
        if item_data is None or item_data.shape[0]==0:
            return None
        
        # 插入编号
        item_data.insert(loc=0, column='code', value=code)
        # 清洗无效数据
        item_data = self.data_clean(item_data)
                 
        return item_data
    
    def import_variety_trade_schedule(self): 
        """从固定表格中更新导入期货品种的交易时间段"""      
        
        datapath = "/home/qdata/futures_data/ak/ins_trading_sche.csv"
        data = pd.read_csv(datapath)
        for index,row in data.iterrows():
            upt_sql = "update trading_variety set ac_time_range='{}',night_time_range='{}',day_time_range='{}' where code='{}'" \
                .format(row['集合竞价'],row['夜盘时间'],row['日盘时间'],row['品种代码'])
            self.dbaccessor.do_inserto(upt_sql)
        
    def import_his_data(self):
        """导入历史行情数据"""
        
        period = PeriodType.DAY.value
        # 不下载期权数据
        variety_sql = "select code from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'code': sqlalchemy.String,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
            'settle': sqlalchemy.FLOAT,          
            "date": sqlalchemy.DateTime
        }        
        # 遍历所有品种，并分别取得历史数据
        for result in result_rows:
            code = result[0]
            # 加0表示主力连续合约
            main_code = code + "0"
            item_data = self.extract_item_data(main_code)
            item_data['code'] = item_data['code'].str[:-1]
            # 保存csv数据文件,全量
            # savepath = "{}/{}".format(self.item_savepath,get_period_name(period))
            # self.export_item_data(code,item_data,is_complete=True,savepath=savepath,period=period,institution=False)     
            # 保存到数据库表
            item_data.to_sql('dominant_continues_data', engine, index=False, if_exists='append',dtype=dtype)  
            print("code:{} ok".format(code))
        
        # 数据表关联字段挂接
        upt_sql = "update dominant_continues_data d set d.var_id=(select id from trading_variety t where t.code=d.code)"
        self.dbaccessor.do_updateto(upt_sql)
        # 结算价为0并且收盘价不为0的，把收盘价赋给结算价
        upt_sql = "update dominant_continues_data set settle=close where settle=0 and close>0 "
        self.dbaccessor.do_updateto(upt_sql)

    def import_continius_his_data_local(self,date_range=[2201,2412]):
        """从本地导入主力连续历史行情数据"""
        
        # 不下载期权数据
        variety_sql = "select id,code from trading_variety where isnull(magin_radio)=0 and id>27 order by id asc"
        result_rows = self.dbaccessor.do_query(variety_sql)        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            'var_id': sqlalchemy.INT,
            'code': sqlalchemy.String,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
            'settle': sqlalchemy.FLOAT,          
            "date": sqlalchemy.DateTime
        }        
        begin_date = datetime.datetime.strptime(str(date_range[0]), '%y%m')
        # 遍历所有品种，并分别取得历史数据
        for result in result_rows:
            var_id = result[0]
            code = result[1]
            cur_date = begin_date
            cur_month = date_range[0]
            # 在指定日期内循环取得每个月的数据
            while cur_month<=date_range[1]:
                # 主力合约名称使用品种编码+YYMM的格式
                main_code = code + str(cur_month)
                item_data = self.extract_main_item_data(main_code)
                if item_data is None:
                    continue
                # 填充固定字段
                item_data['code'] = main_code
                item_data['var_id'] = var_id
                # 保存到数据库表
                item_data.to_sql('dominant_real_data_sina', engine, index=False, if_exists='append',dtype=dtype)  
                # 切换到下个月
                cur_date = get_next_month(cur_date,1)
                cur_month = int(cur_date.strftime('%y%m'))
                time.sleep(5)
            print("code:{} ok".format(code))
        
        # 结算价为0并且收盘价不为0的，把收盘价赋给结算价
        upt_sql = "update dominant_real_data_sina set settle=close where settle=0 and close>0 "
        self.dbaccessor.do_updateto(upt_sql)

    def build_industry_data(self):
        """生成行业板块历史行情数据"""
        
        # 汇总每天每个行业板块的成交信息（忽略已经生成的行业数据）
        combine_sql = "select d.date,concat('ZS_',i.code) as v_code,avg(d.open),avg(d.close),avg(d.high), "\
            "avg(d.low),avg(d.volume),avg(hold),avg(settle) " \
            "from dominant_continues_data d,trading_variety v,futures_industry i where d.var_id=v.id and v.industry_id=i.id " \
            " and v.available=1 and v.magin_radio is not null" \
            " and d.date not in(select date from dominant_continues_data where code like 'ZS_%') group by d.date,v.industry_id"
        combine_sql = "insert into dominant_continues_data(date,code,open,close,high,low,volume,hold,settle) ({})".format(combine_sql)
        self.dbaccessor.do_inserto(combine_sql)    
        # 生成总体指数数据
        # combine_sql = "select d.date,'ZS_all' as v_code,avg(d.open),avg(d.close),avg(d.high), "\
        #     "avg(d.low),avg(d.volume),avg(hold),avg(settle) from dominant_continues_data d where d.code like 'ZS_%' and d.code<>'ZS_all'" \
        #     " and d.date not in(select date from dominant_continues_data where code='ZS_all') group by d.date"
        # combine_sql = "insert into dominant_continues_data(date,code,open,close,high,low,volume,hold,settle) ({})".format(combine_sql)
        # self.dbaccessor.do_inserto(combine_sql)   
                  
    def import_extension_data(self,begin_date=None):
        """导入历史行情辅助数据"""
        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        if begin_date is None:
            begin_date = '2015-03-01'
    
        # 获取基差信息
        self.extract_basis_rate(begin_date,engine=engine)

    def extract_outer_data(self):
        """导入外盘数据"""
        
        # 外盘品种入库
        # futures_hq_subscribe_exchange_symbol_df = ak.futures_hq_subscribe_exchange_symbol()
        # for row in futures_hq_subscribe_exchange_symbol_df.iterrows():
        #     insert_sql = "insert into trading_variety_outer(name,code) values('{}','{}')".format(row[1]["symbol"],row[1]["code"])
        #     self.dbaccessor.do_inserto(insert_sql)

        # 入库外盘历史行情
        variety_sql = "select code from trading_variety_outer where isnull(var_id)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)        
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))

        dtype = {
            "date": sqlalchemy.DateTime,
            'code': sqlalchemy.String,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'position': sqlalchemy.INT,
            's': sqlalchemy.INT 
        }        
        # 遍历所有品种，并分别取得历史数据
        for result in result_rows:
            code = result[0]
            item_data = ak.futures_foreign_hist(symbol=code)
            if code=='SC':
                print("ggg")
            item_data["code"] = code
            # 保存到数据库表
            item_data.to_sql('outer_trading_data', engine, index=False, if_exists='append',dtype=dtype)  
            print("code:{} ok".format(code))        
                           
    def extract_basis_rate(self,begin_date,engine=None):
        """获取基差信息"""

        # 取得对应历史记录中的日期，并以此日期为基准遍历
        date_sql = "select distinct(date) from dominant_continues_data where date>='{}'".format(begin_date)
        result_rows = self.dbaccessor.do_query(date_sql)    
        for index,result in enumerate(result_rows):
            date = result[0]
            item_data = ak.futures_spot_price(date)
            # 首先导入到临时表，后续一并拼接
            if index==0:
                item_data.to_sql('temp_spot_price', engine, index=False, if_exists='replace') 
            else:
                item_data.to_sql('temp_spot_price', engine, index=False, if_exists='append') 
            print("futures_spot_price {} ok".format(date))
                
        # 先清除重复的
        filter_sql = "delete from temp_spot_price where exists (select 1 from extension_trade_info e where " \
            "temp_spot_price.date=e.date and temp_spot_price.symbol=e.code)"
        self.dbaccessor.do_inserto(filter_sql)
        # 然后批量插入
        combine_sql = "insert into extension_trade_info(date,code,dom_basis_rate,near_basis_rate) " \
                        " select date,symbol,dom_basis_rate,near_basis_rate from temp_spot_price"   
        self.dbaccessor.do_inserto(combine_sql)      
        # 关联字段挂接
        upt_sql = "update extension_trade_info d set d.var_id=(select id from trading_variety t where t.code=d.code)"
        self.dbaccessor.do_updateto(upt_sql)  
        # 清空临时表
        self.dbaccessor.do_updateto("drop table temp_spot_price")  
    
    def export_to_qlib(self,cross_mode=False):
        """导出到qlib"""
        
        # 首先从数据库导出到csv
        save_path = "{}/day/csv_data".format(self.item_savepath)
        # 使用交错表
        if cross_mode:
            table = "dominant_continues_data_cross"
        else:
            table = "dominant_continues_data"
        date_sql = "select distinct(code) from {}".format(table)
        result_rows = self.dbaccessor.do_query(date_sql)    
        for index,result in enumerate(result_rows):
            code = result[0]
            item_sql = "select date,open,close,high,low,volume,hold,settle from {} where code='{}'".format(table,code)
            item_rows = self.dbaccessor.do_query(item_sql)    
            filename = "{}/{}.csv".format(save_path,code)
            with open(filename,mode='w',encoding='utf-8') as f:
                writer = csv.writer(f,dialect='excel')
                # 改变字段名，把结算价设置为收盘价，收盘价命名为参考收盘价
                header = ["datetime","open","refclose","high","low","volume","hold","close"]
                writer.writerow(header)
                for row in item_rows:
                    writer.writerow(row)            
                print("{} ok".format(code))
                
        # 然后导出到qlib
        qlib_dir = "/home/qdata/qlib_data/futures_data"
        super().export_to_qlib(qlib_dir,PeriodType.DAY.value,file_name="all.txt",institution=False)        
    
    def import_day_range_contract_data(self,data_range=None):
        """导入指定日期范围的合约数据"""
        
        begin_date,end_date = data_range
        begin_date = datetime.datetime.strptime(str(begin_date), '%Y%m%d').date()
        end_date = datetime.datetime.strptime(str(end_date), '%Y%m%d').date()

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
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }  
        tar_cols = list(dtype.keys())                
        date_sql = "select max(date) from dominant_real_data"
        result_rows = self.dbaccessor.do_query(date_sql) 
        max_date = result_rows[0][0]          
        # 如果日期范围内包含了记录中的日期，则修改起始日期
        if max_date>=end_date:
            print("exceed end date,max date:{}".format(max_date))
            return        
        if max_date>begin_date:
            begin_date = max_date
        
        market_sql = "select code from futures_exchange"
        result_rows = self.dbaccessor.do_query(market_sql) 
        # 依次轮询各个市场，取得对应合约并插入数据库  
        for row in result_rows:
            market_code = row[0]
            # 先不使用金融期货数据
            if market_code in ['CFFEX','INE']:
                continue
            get_futures_daily_df = get_futures_daily(start_date=begin_date, end_date=end_date, market=market_code)
            if get_futures_daily_df.shape[0]==0:
                continue
            get_futures_daily_df = get_futures_daily_df.rename(columns={"turnover": "hold","symbol": "code"})
            if get_futures_daily_df['open'].dtypes.hasobject:
                get_futures_daily_df['open'] = get_futures_daily_df['open'].astype(str)
                get_futures_daily_df = get_futures_daily_df[get_futures_daily_df['open'].str.len()>0]
                get_futures_daily_df['open'] = get_futures_daily_df['open'].astype(float)
            # 郑商所代码规范和其他不一致，需要补全
            if market_code=='CZCE':
                get_futures_daily_df['code'] = get_futures_daily_df['code'].map(lambda code: code[:2]+'2'+code[2:])                
            get_futures_daily_df[tar_cols].to_sql('dominant_real_data', engine, index=False, if_exists='append',dtype=dtype)
            print("market {} ok".format(market_code))

    def import_day_range_contract_data_em(self,data_range=None):
        """导入指定日期范围的合约数据,东方财富渠道"""
        
        begin_date,end_date = data_range
        begin_date = datetime.datetime.strptime(str(begin_date), '%Y%m%d').date()
        end_date = datetime.datetime.strptime(str(end_date), '%Y%m%d').date()

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
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }  
        tar_cols = list(dtype.keys())                
        date_sql = "select max(date) from dominant_real_data"
        result_rows = self.dbaccessor.do_query(date_sql) 
        max_date = result_rows[0][0]          
        # 如果日期范围内包含了记录中的日期，则修改起始日期
        if max_date>=end_date:
            print("exceed end date,max date:{}".format(max_date))
            return        
        if max_date>begin_date:
            begin_date = max_date
        

        symbole_list =  ak.futures_hist_table_em()['合约代码']

        for symbol in symbole_list:
            if not re.match(".*\d$",symbol):
                continue
            futures_hist_em_df = futures_hist_em(symbol=symbol,start_date=str(begin_date),end_date=str(end_date))
            if futures_hist_em_df is None:
                continue
            futures_hist_em_df = futures_hist_em_df.rename(
                columns={"时间": "date","开盘": "open","最高": "high","最低": "low","收盘": "close",
                         "成交量": "volume","持仓量": "hold"})
            futures_hist_em_df['code'] = symbol
            futures_hist_em_df['settle'] = futures_hist_em_df['close']
            futures_hist_em_df.drop(columns=['涨跌','涨跌幅','成交额'])
            futures_hist_em_df[tar_cols].to_sql('dominant_real_data', engine, index=False, if_exists='append',dtype=dtype)
            print("{} import ok".format(symbol))            
            
    def import_day_range_continues_data(self,data_range=None):
        """导入指定日期范围的主连数据"""
        
        begin_date,end_date = data_range
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        dtype = {
            'code': sqlalchemy.String(10),
            "date": sqlalchemy.DateTime,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,  
            'settle': sqlalchemy.FLOAT, 
        }  
        tar_cols = list(dtype.keys())                
        date_sql = "select max(date) from dominant_continues_data"
        result_rows = self.dbaccessor.do_query(date_sql) 
        max_date = int(result_rows[0][0].strftime("%Y%m%d"))      
        # 如果日期范围内包含了记录中的日期，则修改起始日期
        if max_date>=end_date:
            print("exceed end date,max date:{}".format(max_date))
            return        
        if max_date>begin_date:
            begin_date = get_next_working_day(datetime.datetime.strptime(str(max_date), "%Y%m%d"))
        
        if begin_date==end_date and not is_working_day(str(end_date)):
            print("not working day:{}".format(end_date))
            return
        
        variety_sql = "select code from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)   
        _, _, e_symbol_mkt, _ = (
            get_exchange_symbol_map()
        )
        # 依次轮询各个品种，取得对应数据并插入数据库  
        for row in result_rows:
            var_code = row[0]
            # if var_code!="AP":
            #     continue
            futures_hist_em_df = futures_hist_em(symbol=var_code,start_date=str(begin_date),end_date=str(end_date),e_symbol_mkt=e_symbol_mkt)
            # futures_hist_em_df = ak.futures_hist_em(symbol=var_code,start_date=str(begin_date),end_date=str(end_date))
            if futures_hist_em_df is None:
                continue
            futures_hist_em_df = futures_hist_em_df.rename(
                columns={"时间": "date","开盘": "open","最高": "high","最低": "low","收盘": "close",
                         "成交量": "volume","持仓量": "hold"})
            futures_hist_em_df['code'] = var_code
            futures_hist_em_df['settle'] = futures_hist_em_df['close']
            futures_hist_em_df.drop(columns=['涨跌','涨跌幅','成交额'])
            futures_hist_em_df[tar_cols].to_sql('dominant_continues_data', engine, index=False, if_exists='append',dtype=dtype)
            print("import_day_range_continues_data {}  ok".format(var_code))
            time.sleep(15)

    def import_day_range_1min_data(self,data_range=None):
        """导入分钟历史数据"""
        
        begin_date,end_date = data_range
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        
        dtype = {
            'code': sqlalchemy.String,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
            'settle': sqlalchemy.FLOAT,          
            "date": sqlalchemy.DateTime
        }        
        cur_date = datetime.date.today()
        date_sql = "select max(datetime) from dominant_real_data_1min"
        result_rows = self.dbaccessor.do_query(date_sql) 
        max_date = int(result_rows[0][0].strftime("%Y%m%d")) 
        if max_date>=end_date:
            print("exceed date:{}".format(max_date))
            return
        if max_date>begin_date:
            begin_date = max_date      
        # 不下载期权数据
        variety_sql = "select code from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)            
        # 遍历所有品种，并分别取得历史数据
        for result in result_rows:
            code = result[0]
            # if code!="CJ":
            #     continue
            # 取得比较接近的合约
            contract_names = self.get_likely_main_contract_names(code,cur_date)
            for contract_name in contract_names:
                # 直接从文件中读取
                item_save_path = os.path.join(self.get_1min_save_path(),"{}.csv".format(contract_name))
                if not os.path.exists(item_save_path):
                    logger.warning("main data not found in:{}".format(item_save_path))
                    continue
                futures_zh_minute_sina_df = pd.read_csv(item_save_path)                
                # 只取超出原有数据日期的数据
                futures_zh_minute_sina_df = futures_zh_minute_sina_df[
                    (pd.to_numeric(pd.to_datetime(futures_zh_minute_sina_df['datetime']).dt.strftime('%Y%m%d'))>=begin_date)&
                    (pd.to_numeric(pd.to_datetime(futures_zh_minute_sina_df['datetime']).dt.strftime('%Y%m%d'))<=end_date)]
                futures_zh_minute_sina_df['code'] = contract_name
                futures_zh_minute_sina_df.to_sql('dominant_real_data_1min', engine, index=False, if_exists='append',dtype=dtype)  
                time.sleep(5)
            print("code:{} ok".format(code))

    def import_day_range_1min_data_cross(self,data_range=None):
        """导入分钟历史数据,交错模式"""
        
        begin_date,end_date = data_range
        engine = self.create_engine()
        dtype = {
            'code': sqlalchemy.String,
            'open': sqlalchemy.FLOAT,
            'close': sqlalchemy.FLOAT,
            'high': sqlalchemy.FLOAT,
            'low': sqlalchemy.FLOAT,
            'volume': sqlalchemy.FLOAT,
            'hold': sqlalchemy.FLOAT,   
            'settle': sqlalchemy.FLOAT,          
            "date": sqlalchemy.DateTime
        }        
        date_sql = "select max(datetime) from dominant_real_data_1min_cross"
        result_rows = self.dbaccessor.do_query(date_sql) 
        max_datetime = result_rows[0][0]
        if max_datetime is None:
            max_date = 0
        else:
            max_date = int(max_datetime.strftime("%Y%m%d")) 
        
        # 当天作为截止日期 
        next_date = end_date
        next_date_date = datetime.datetime.strptime(str(end_date), "%Y%m%d")
        next_date_begin_str = next_date_date.strftime("%Y-%m-%d") +  " 00:00:00"
        next_date_begin = datetime.datetime.strptime(next_date_begin_str,"%Y-%m-%d %H:%M:%S")
        next_date_end_str = next_date_date.strftime("%Y-%m-%d") +  " 23:59:59"
        next_date_mid_str = next_date_date.strftime("%Y-%m-%d") +  " 12:00:00"
        next_date_mid = datetime.datetime.strptime(next_date_mid_str,"%Y-%m-%d %H:%M:%S")
        # 如果超出了当天上午收盘时间，则认为已经导入过了
        if max_date>end_date or (max_date==next_date and max_datetime.hour>12):
            print("exceed date in import_day_range_1min_data_cross:{}".format(max_datetime))
            return
        if max_date>begin_date:
            begin_date = max_date   
        
        # 在此模式下，首先拷贝前一天已经导入的分钟数据，然后导入当天上午的数据
        prev_day = get_prev_working_day(datetime.datetime.strptime(str(begin_date), "%Y%m%d"))
        begin_time_str = prev_day.strftime("%Y-%m-%d") + " 00:00:00"
        end_time_str = date_string_transfer(str(end_date)) + " 23:59:59"
        end_time_before_str = date_string_transfer(str(end_date)) + " 00:00:00"
        del_sql = "delete from dominant_real_data_1min_cross where datetime>='{}' and datetime<='{}'". \
            format(begin_time_str,next_date_end_str)   
        # 先删后增
        self.dbaccessor.do_updateto(del_sql)
        sql = ("insert into dominant_real_data_1min_cross select * from dominant_real_data_1min " + 
            "where datetime>='{}' and datetime<'{}'").format(begin_time_str,end_time_before_str)   
        self.dbaccessor.do_inserto(sql)               
        # 不下载期权数据
        variety_sql = "select code from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)    
        # 遍历所有品种，并分别取得历史数据,只需要取得当天上午的数据
        for result in result_rows:
            code = result[0]
            # 取得比较接近的合约
            contract_names = self.get_likely_main_contract_names(code,next_date_date.date())
            for contract_name in contract_names:
                # 直接从文件中读取
                item_save_path = os.path.join(self.get_1min_save_path(),"{}.csv".format(contract_name))
                if not os.path.exists(item_save_path):
                    logger.warning("main data not found in:{}".format(item_save_path))
                    continue
                futures_zh_minute_sina_df = pd.read_csv(item_save_path)
                # 只取当天上午的数据
                futures_zh_minute_sina_df = futures_zh_minute_sina_df[
                    (pd.to_datetime(futures_zh_minute_sina_df['datetime'])>=next_date_begin)&
                    (pd.to_datetime(futures_zh_minute_sina_df['datetime'])<=next_date_mid)]                
                if futures_zh_minute_sina_df.shape[0]>0:
                    futures_zh_minute_sina_df.to_sql('dominant_real_data_1min_cross', engine, index=False, if_exists='append',dtype=dtype)  
                else:
                    logger.warning("no match data in:{}".format(contract_name))
    
    def get_1min_save_path(self):
        save_path = os.path.join(self.savepath,"item/min")
        return save_path
    
    def store_1min_data(self,data_range=None):    
        """保存1分钟数据到本地文件""" 
        
        cur_date = datetime.datetime.now().date()
        # 筛选合适的数据
        variety_sql = "select code from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(variety_sql)       
        # 遍历所有品种，并分别取得历史数据,只需要取得下一天上午的数据
        for result in result_rows:
            code = result[0]
            # 取得比较接近的合约
            contract_names = self.get_likely_main_contract_names(code,cur_date)
            for contract_name in contract_names:
                status_code,futures_zh_minute_sina_df = futures_zh_minute_sina(symbol=contract_name, period="1")
                # 如果频繁调用限制，则等一会儿再试
                if status_code==-1:
                    print("try again later")
                    time.sleep(60)
                    status_code,futures_zh_minute_sina_df = futures_zh_minute_sina(symbol=contract_name, period="1")
                if status_code==0:
                    print("contract {} has no data".format(contract_name))
                    continue                    
                if futures_zh_minute_sina_df.shape[0]>0:
                    futures_zh_minute_sina_df['code'] = contract_name
                    futures_zh_minute_sina_df = futures_zh_minute_sina_df.reset_index(drop=True)
                    # 存储到文件
                    item_save_path = os.path.join(self.get_1min_save_path(),"{}.csv".format(contract_name))
                    futures_zh_minute_sina_df.to_csv(item_save_path,index=None)
                time.sleep(5)
            logger.info("store_1min_data,code:{} ok".format(code))        
        
    def get_last_minutes_data(self,symbol):
        """取得指定品种最近一分钟数据"""
        
        status_code,futures_zh_minute_sina_df = futures_zh_minute_sina(symbol=symbol, period="1")
        # 如果频繁调用限制，则等一会儿再试
        if status_code==-1:
            print("try again later")
            time.sleep(10)
        status_code,futures_zh_minute_sina_df = futures_zh_minute_sina(symbol=symbol, period="1")
        if status_code==0 or status_code==-1:
            print("contract {} has no data".format(symbol))
            return None
        return futures_zh_minute_sina_df.iloc[-1]         
    
    def build_qlib_instrument(self):
        """qlib品种名单列表生成"""
        
        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        total_file = "all.txt"
        # 去除金融期货、重金属，以及上市日期太短的
        total_path = os.path.join(qlib_dir,total_file)
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        columns = ['code','begin','end']
        all = pd.read_table(total_path,sep='\t',header=None)   
        all.columns = columns
        keep_instruments = [] 
        for index,row in all.iterrows():
            code = row['code']
            begin = int(date_string_transfer(row['begin'],2))
            end = int(date_string_transfer(row['end'],2))
            if begin>20200101 or end<20250901:
                continue
            exchange_code = self.get_exchange_from_instrument(code)
            if exchange_code in ['CFFEX','INE']:
                continue
            if code in ['AU','AG','ZS_JRQH','ZS_NMFI']:
                continue
            keep_instruments.append(code)
        clean_data = all[all['code'].isin(keep_instruments)]
        clean_data.to_csv(clean_data_path, sep='\t', index=False,header=None)

    def build_cleandata_table(self):
        ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
        ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
        data = pd.DataFrame(ins_data.values,columns=['instrument','begin_date','end_date'])
        engine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(
            self.dbaccessor.user,self.dbaccessor.password,self.dbaccessor.host,self.dbaccessor.port,self.dbaccessor.database))
        dtype = {
            "instrument": sqlalchemy.String,         
            "begin_date": sqlalchemy.DateTime,
            "end_date": sqlalchemy.DateTime
        }        
        data.to_sql('clean_data', engine, index=False, if_exists='append',dtype=dtype)                              
        
    def rebuild_qlib_instrument(self):
        """qlib品种名单列表重新生成，主要是更新日期"""
        
        qlib_dir = "/home/qdata/qlib_data/futures_data/instruments"
        clean_data_file = "clean_data.txt"
        total_file = "all.txt"
        # 上一步骤已经生成了all.txt的日期对照，在这里把日期对照关系同步到具体文件
        total_path = os.path.join(qlib_dir,total_file)
        clean_data_path = os.path.join(qlib_dir,clean_data_file)
        columns = ['code','begin','end']
        all = pd.read_table(total_path,sep='\t',header=None)   
        all.columns = columns
        clean_data = pd.read_table(clean_data_path,sep='\t',header=None)   
        clean_data.columns = columns
        clean_data = all[all['code'].isin(clean_data['code'].values)]
        clean_data.to_csv(clean_data_path, sep='\t', index=False,header=None)
            
    def get_realtime_data(self,symbol,market):
        """新浪实时行情"""
        
        futures_zh_spot_df = ak.futures_zh_spot(symbol=symbol, market='CF', adjust='0')
        futures_zh_spot_df['close'] = futures_zh_spot_df['last_close']
        return futures_zh_spot_df.iloc[0]
    
    def get_day_contract(self):
        """取得当前可交易合约"""

        dce_text = ak.match_main_contract(symbol="dce")
        czce_text = ak.match_main_contract(symbol="czce")
        shfe_text = ak.match_main_contract(symbol="shfe")
        gfex_text = ak.match_main_contract(symbol="gfex")
        
        all_str = ",".join([dce_text, czce_text, shfe_text, gfex_text])  
        all_contract_arr = all_str.split(",")
        return all_contract_arr

    def get_day_contract_info(self):
        """取得当前可交易合约信息"""

        dce_text = ak.match_main_contract(symbol="dce")
        czce_text = ak.match_main_contract(symbol="czce")
        shfe_text = ak.match_main_contract(symbol="shfe")
        gfex_text = ak.match_main_contract(symbol="gfex")
        time.sleep(3)
        futures_zh_spot_df = ak.futures_zh_spot(
            symbol=",".join([dce_text, czce_text, shfe_text, gfex_text]),
            market="CF",
            adjust='0')
        return futures_zh_spot_df         
               
if __name__ == "__main__":    
    
    extractor = AkFuturesExtractor(savepath="/home/qdata/futures_data")   
    save_path = "custom/data/results/futures"
    # 期货规则-交易日历表,交易品种
    # futures_rule_df = ak.futures_rule(date="20231205")
    # print(futures_rule_df)
    # futures_rule_df.to_csv(save_path+ "/rule.csv",index=False)
    # 合约情况
    # contract_df = ak.match_main_contract(symbol="shfe") 
    # print(contract_df)
    # hist_em = ak.futures_hist_table_em() 
    # print(hist_em)
    
    # 实时行情
    # futures_zh_spot_df = ak.futures_zh_spot(symbol='RB2510', market="CF", adjust='0')
    # print(futures_zh_spot_df)
    # extractor.get_day_contract_info()
    # extractor.get_day_contract()
    # futures_zh_realtime_df = ak.futures_zh_realtime(symbol="白糖")
    # print(futures_zh_realtime_df)    
    # 分时数据
    # futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="FU2509", period="1")
    # print(futures_zh_minute_sina_df)
    # 历史行情
    # futures_hist_em_df = futures_hist_em(symbol="BB", period="daily")
    # futures_hist_em_df['date'] = futures_hist_em_df['时间']
    # print(futures_hist_em_df)    
    # futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="RB0", period="1")
    # print(futures_zh_minute_sina_df)
    # get_futures_daily_df = ak.get_futures_daily(start_date="20250501", end_date="20250516", market="SHFE")
    # print(get_futures_daily_df)    
    # futures_zh_daily_em_df = ak.futures_hist_em(symbol="BR2201")
    # print(futures_zh_daily_em_df)
    # 外盘品种
    # futures_hq_subscribe_exchange_symbol_df = ak.futures_hq_subscribe_exchange_symbol()
    # print(futures_hq_subscribe_exchange_symbol_df)
    # 外盘历史行情
    # futures_foreign_hist_df = ak.futures_foreign_hist(symbol="ZSD")
    # print(futures_foreign_hist_df)    
    # 交易费用
    # futures_fees_info_df = ak.futures_fees_info()
    # print(futures_fees_info_df)   
    # futures_comm_info_df = ak.futures_comm_info(symbol="所有")
    # print(futures_comm_info_df)    
    # 连续合约
    # futures_main_sina_hist = ak.futures_main_sina(symbol="RB0", start_date="20130513", end_date="20220101")
    # print(futures_main_sina_hist.iloc[:10])    
    # 合约详情
    # futures_contract_detail_df = ak.futures_contract_detail(symbol='V2001')
    # print(futures_contract_detail_df)    
    # 现货价格和基差 
    # futures_spot_price_df = ak.futures_spot_price("20110105")
    # print(futures_spot_price_df)       
    # 历史价格和基差 
    # futures_spot_price_previous_df = ak.futures_spot_price_previous('20240430')
    # print(futures_spot_price_previous_df)
    # 注册仓单
    # reg_receipt = ak.get_receipt(start_date="20180712", end_date="20180719", vars_list=["RB"])
    # print(reg_receipt)
    # 库存数据
    # futures_inventory_em_df = ak.futures_inventory_em(symbol="豆一")
    # print(futures_inventory_em_df)    
    # 期转现
    # futures_to_spot_czce_df = ak.futures_to_spot_dce(date="202410")
    # print(futures_to_spot_czce_df)
    # 交割统计
    # futures_delivery_dce_df = ak.futures_delivery_dce(date="201501")
    # print(futures_delivery_dce_df)
    # 展期收益率
    # df = ak.get_roll_yield_bar(type_method="symbol", var="RB", date="20191008") 
    # df = ak.get_roll_yield_bar(type_method="date", var="J", start_day="20210809", end_day="20211030")
    # df = ak.get_roll_yield_bar(type_method="var", date="20191008")
    # print(df)

    
    # 导入期货交易品种
    # extractor.import_trading_variety()    
    # 导入交易时间表
    # extractor.import_variety_trade_schedule()    
    # 导入主力连续历史数据
    # extractor.import_his_data()
    # 导入历史拓展数据
    # extractor.import_extension_data()
    # 生成行业板块历史行情数据
    # extractor.build_industry_data()
    # 导入外盘数据
    # extractor.extract_outer_data()
    # 导出到qlib
    extractor.export_to_qlib()
    # extractor.load_item_day_data("CU2205", "2022-03-03")
    # extractor.build_cleandata_table()
    # qlib品种名单列表生成
    # extractor.build_qlib_instrument()
    # extractor.rebuild_qlib_instrument()
    
    ############ 历史合约数据导入 ###################
    # extractor.import_day_range_contract_data(data_range=(20250924,20250925))
    # extractor.import_day_range_contract_data_em(data_range=(20250630,20250630))
    # extractor.import_day_range_continues_data(data_range=(20250926,20250926))
    # extractor.import_day_range_1min_data(data_range=(20250924,20250925))
            