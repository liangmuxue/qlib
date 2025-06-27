from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType

import time
import datetime
import math
import pickle
import os
import csv
from sqlalchemy import create_engine
import sqlalchemy

import akshare as ak
from akshare.stock_feature.stock_hist_em import (
    code_id_map_em
)
from tqsdk import TqApi, TqAuth

import numpy as np
import pandas as pd
import requests

from trader.utils.date_util import get_first_and_last_datetime,tradedays,get_next_month
from cus_utils.log_util import AppLogger

logger = AppLogger()

class TqFuturesExtractor(HisDataExtractor):
    """天勤期货数据源"""

    def __init__(self, backend_channel="tq",savepath=None,**kwargs):
        self.busi_columns = ["code","date","open","high","low","close","volume","hold","settle"]
        self.api = TqApi(auth=TqAuth("liangmuxue", "182828"))
          
    #######################   数据导入 ######################################      

    def extract_item_data(self,symbol,date=None,period=PeriodType.MIN1.value):  
        """取得单个品种的某日分钟历史行情数据"""
        
        api = self.api
        
        tq_symbol = self.transfer_symbol_code(symbol)
        cur_date = datetime.date.today()
        # 根据指定日期距离当前日期的交易天数，决定调用api的间隔数量
        date_duration = tradedays(date,cur_date)
        if date_duration>8:
            # 如果超出太多，则不处理
            logger.warning("date expire:{}".format(date_duration)) 
            return None
        # 一天按照360分钟交易时间粗略评估
        minute_numbers = 360*date_duration
        # 取得一直到目前的所有数据，并按照日期筛选实际结果
        klines = api.get_kline_serial(tq_symbol, 60,minute_numbers)
        first_timestamp,last_timestamp = get_first_and_last_datetime(date)
        klines = klines[(klines['datetime']>=first_timestamp)&(klines['datetime']<=last_timestamp)]
                 
        return klines

    def transfer_symbol_code(self,symbol_code): 
        """标准合约名称转换为天勤对应的合约名称"""      
        
        sql = "select f.code as ex_code from future_exchange f,trading_variety t where t.exchange_id=f.id and f.code='{}'".format(symbol_code)
        results = self.dbaccessor.do_query(sql)
        if len(results)==0:
            return None
        ex_code = results[0][0]
        tq_symbol_code = "{}.{}".format(ex_code,symbol_code.lower())
        return tq_symbol_code
        
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
                  

    def test_api(self):
        api = self.api
        # quote = self.api.get_quote("SHFE.ni2206")        
        # print(quote)
        symbol = "SHFE.cu2506"
        klines = api.get_kline_serial(symbol, 60)
        print("kline data:{}".format(klines))
        while True:
            api.wait_update()
            print(klines.iloc[-1].close)        
    
if __name__ == "__main__":    
    
    extractor = TqFuturesExtractor(savepath="/home/qdata/futures_data")   
    save_path = "custom/data/results/futures"
    extractor.test_api()
    
    # 期货规则-交易日历表,交易品种
    # futures_rule_df = ak.futures_rule(date="20231205")
    # print(futures_rule_df)
    # futures_rule_df.to_csv(save_path+ "/rule.csv",index=False)
    # 合约情况
    # contract_df = ak.match_main_contract(symbol="shfe") 
    # print(contract_df)
    # 实时行情
    # futures_zh_spot_df = ak.futures_zh_spot(symbol='FU2501', market="shfe", adjust='0')
    # print(futures_zh_spot_df)
    # 分时数据
    # futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="FU2501", period="1")
    # print(futures_zh_minute_sina_df)
    # 历史行情
    
            