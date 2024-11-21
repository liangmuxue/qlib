from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType

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
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests

from cus_utils.log_util import AppLogger
from data_extract.akshare_extractor import AkExtractor
from data_extract.his_data_extractor import get_period_name

class AkFuturesExtractor(AkExtractor):
    """akshare期货数据源"""

    def __init__(self, backend_channel="ak",savepath=None,**kwargs):
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
        self.busi_columns = ["date","open","high","low","close","volume","hold","settle"]

    def get_whole_item_datapath(self,period,institution=False):
        period_name = get_period_name(period)
        if institution:
            return "{}/item/{}/institution".format(self.savepath,period_name)
        return "{}/item/{}/csv_data".format(self.savepath,period_name)
            
    def import_trading_variety(self):
        """导入交易品种数据"""

        exchange_sql = "select id,short_name from futures_exchange"
        result_rows = self.dbaccessor.do_query(exchange_sql)        
        exchange_map = {}
        for result in result_rows:
            exchange_map[result[1]] = result[0]
        
        insert_sql = "insert into trading_variety(exchange_id,code,name,magin_radio,limit_rate) values(%s,%s,%s,%s,%s)"
        futures_rule_df = ak.futures_rule()        
        for idx,row in futures_rule_df.iterrows():
            exchange_id = exchange_map[row['交易所']]
            magin_radio = row['交易保证金比例'] if not math.isnan(row['交易保证金比例']) else None
            limit_rate = row['涨跌停板幅度'] if not math.isnan(row['交易保证金比例']) else None
            self.dbaccessor.do_inserto_withparams(insert_sql, (exchange_id,row['代码'],row['品种'],magin_radio,limit_rate))           

    def extract_item_data(self,code,start_date=None,end_date=None,period=PeriodType.DAY.value):  
        """取得单个股票历史行情数据"""
        
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
            # 变回原来的编码
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
    
    def build_industry_data(self):
        """生成行业板块历史行情数据"""
        
        # 汇总每天每个行业板块的成交信息（忽略已经生成的行业数据）
        combine_sql = "select d.date,concat('ZS_',i.code) as v_code,avg(d.open),avg(d.close),avg(d.high), "\
            "avg(d.low),avg(d.volume),avg(hold),avg(settle) " \
            "from dominant_continues_data d,trading_variety v,futures_industry i where d.var_id=v.id and v.industry_id=i.id " \
            " and d.date not in(select date from dominant_continues_data where code like 'ZS_%') group by d.date,v.industry_id"
        
        combine_sql = "insert into dominant_continues_data(date,code,open,close,high,low,volume,hold,settle) ({})".format(combine_sql)
        print(combine_sql)
        self.dbaccessor.do_inserto(combine_sql)    
                      
          
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
    
    def export_to_qlib(self):
        """导出到qlib"""
        
        # 首先从数据库导出到csv
        save_path = "{}/day/csv_data".format(self.item_savepath)
        date_sql = "select distinct(code) from dominant_continues_data"
        result_rows = self.dbaccessor.do_query(date_sql)    
        for index,result in enumerate(result_rows):
            code = result[0]
            item_sql = "select date,open,close,high,low,volume,hold,settle from dominant_continues_data where code='{}'".format(code)
            item_rows = self.dbaccessor.do_query(item_sql)    
            filename = "{}/{}.csv".format(save_path,code)
            with open(filename,mode='w',encoding='utf-8') as f:
                writer = csv.writer(f,dialect='excel')
                header = ["datetime","open","close","high","low","volume","hold","settle"]
                writer.writerow(header)
                for row in item_rows:
                    writer.writerow(row)            
                print("{} ok".format(code))
                
        # 然后导出到qlib
        qlib_dir = "/home/qdata/qlib_data/futures_data"
        super().export_to_qlib(qlib_dir,PeriodType.DAY.value,file_name="all.txt",institution=False)        
        
    
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
    # 实时行情
    # futures_zh_spot_df = ak.futures_zh_spot(symbol='FU2501', market="shfe", adjust='0')
    # print(futures_zh_spot_df)
    # 分时数据
    # futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="FU2501", period="1")
    # print(futures_zh_minute_sina_df)
    # 历史行情
    # futures_zh_daily_sina_df = ak.futures_zh_daily_sina(symbol="FU0")
    # print(futures_zh_daily_sina_df[1000:1010])
    # 外盘品种
    # futures_hq_subscribe_exchange_symbol_df = ak.futures_hq_subscribe_exchange_symbol()
    # print(futures_hq_subscribe_exchange_symbol_df)
    # 外盘历史行情
    # futures_foreign_hist_df = ak.futures_foreign_hist(symbol="ZSD")
    # print(futures_foreign_hist_df)    
    # 交易费用
    # futures_fees_info_df = ak.futures_fees_info()
    # print(futures_fees_info_df)   
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
    # 导入历史数据
    # extractor.import_his_data()
    # 导入历史拓展数据
    # extractor.import_extension_data()
    # 生成行业板块历史行情数据
    # extractor.build_industry_data()
    # 导入外盘数据
    # extractor.extract_outer_data()
    # 导出到qlib
    extractor.export_to_qlib()
            