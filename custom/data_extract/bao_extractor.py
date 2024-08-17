from .his_data_extractor import HisDataExtractor,PeriodType,MarketType
from mootdx.quotes import Quotes


import numpy as np
import pandas as pd
import datetime

from trader.utils.date_util import tradedays,get_trade_min_dur
from trader.utils.date_util import get_tradedays_dur,date_string_transfer
import baostock as bs

from cus_utils.log_util import AppLogger
logger = AppLogger()

class BaoExtractor(HisDataExtractor):
    """BaoStock数据源"""

    def __init__(self, backend_channel="bao",savepath=None,**kwargs):
        
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
        self.connect()
    
    def connect(self):
        self.lg = bs.login()
        
    def reconnect(self):
        self.connect()
            
    def load_code_data(self):  
        """取得所有股票代码"""
        pass     
                    
    def import_data(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value,
                    contain_institution=False,resume=False,no_total_file=False,auto_import=False):
        """
            取得所有股票历史行情数据
            Params:
                task_batch 任务批次号
                begin_date 导入数据的开始日期
                end_date 导入数据的结束日期
                period 频次类别
        """
        
        results = None  
        flag = False
        cnt = 0
        while True:
            if flag:
                break            
            try: 
                results = super().import_data(task_batch=task_batch, start_date=start_date, end_date=end_date,auto_import=auto_import,
                                               period=period,contain_institution=contain_institution,resume=resume,no_total_file=no_total_file)
                flag = True  
            except Exception as e:
                logger.exception("tdx import_data fail:")      
                self.reconnect()     
                cnt += 1
                if cnt>3:
                    return None
                logger.info("reconnect time:{}".format(cnt)) 
                continue                        
        # 任务结束时关闭连接
        self.bs.logout()
        return results
        
    def extract_item_data(self,instrument_code,start_date=None,end_date=None,period=None,market=MarketType.SH.value,institution=False):   
        """取得单个股票历史行情数据"""
        
        if period<=PeriodType.MIN1.value:
            # 暂时只使用分钟级别加载模式
            logger.warnning("not support period:{}".format(period))
            return
        
        if not institution:
            adjustflag = "3"
        else:
            adjustflag = "1"
        
        start_date = str(datetime.datetime.strptime(str(start_date),'%Y%m%d').date())
        end_date = str(datetime.datetime.strptime(str(end_date),'%Y%m%d').date())
        # 生成符合baostock规范的股票编号
        if market==MarketType.SH.value:
            real_code = "sh." + instrument_code
        else:
            real_code = "sz." + instrument_code
        rs = bs.query_history_k_data_plus(real_code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date, end_date=end_date,
            frequency=str(PeriodType.MIN5.value), adjustflag=adjustflag)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
            
        return result
    
    def compute_period_space(self,start_date=None,end_date=None,period=None):
        """取得指定日期下的间隔数
            Return:
                start_number： 起始时间的前推数
                end_number： 结束时间的前推数
        """
        
        min_number = 1
        if period==PeriodType.MIN1.value:
            min_number = 1
        if period==PeriodType.MIN5.value:
            min_number = 5    
        if period==PeriodType.MIN15.value:
            min_number = 15    
        
        # 根据不同分钟级别，计算一天需要多少K线
        day_item_number = int(4 * 60 / min_number)
        # 计算需要向前退多少天，根据当前日期以及开始日期进行计算
        now_time = datetime.datetime.now()
        today = now_time.strftime('%Y%m%d')
        days_to_begin = tradedays(start_date,today)
        # 前推数量为天数乘以每天K线数
        start_number = int(day_item_number * days_to_begin)
        # 根据结束日期，计算需要查询的K线数量
        days_to_end = tradedays(end_date,today)
        end_number = int(day_item_number * days_to_end)
        # 计算当日时间差值，并补齐
        dur_time = get_trade_min_dur(now_time,min_number)
        end_number = end_number + dur_time - day_item_number
        start_number = start_number + dur_time
        # exceed_number = before_number - item_number
        return start_number,end_number

    def get_real_data(self,instrument_code,market):  
        """取得实时数据"""
        
        api = self.api
        data = api.get_security_quotes([(market, instrument_code)])
        # 按照规范构造返回值，有几项没有数据
        np_data = np.array([[data[0]["code"],data[0]["servertime"],float(data[0]["open"]),float(data[0]["price"]),float(data[0]["high"]),
                             float(data[0]["low"]),float(data[0]["cur_vol"]),float(data[0]["amount"]),0,0,0,0]])
        df = pd.DataFrame(np_data,columns=self.busi_columns)
        df["open"] = pd.to_numeric(df["open"])
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["amount"] = pd.to_numeric(df["amount"])
        # 添加last属性
        df["last"] = pd.to_numeric(df["close"])
        return df
        
    def get_last_data_date(self,data_item,period):    
        """取得存储数据中的最近日期"""
        
        if period==PeriodType.MIN5.value:
            cur_date = data_item["datetime"].max()
            # 分钟模式下，如果最后时间不足15点，则说明数据不完整，需要排除
            if cur_date.hour<15:
                cur_date = str(cur_date.date())
                tar_date = date_string_transfer(cur_date,direction=2)
            else:
                cur_date = str(cur_date.date())
                tar_date = get_tradedays_dur(date_string_transfer(cur_date,direction=2),1)
                tar_date = tar_date.strftime("%Y%m%d")     
            return tar_date  
        else:
            return super().get_last_data_date(data_item,period)
        
    def clear_redun_data(self,ori_data_item,date):
        """清除重复部分的数据"""
        
        data_item = ori_data_item[ori_data_item["datetime"]<date]
        return data_item        
        
    def get_first_default_date(self):
        return "20220101"        
        
    