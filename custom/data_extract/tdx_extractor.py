from pytdx.hq import TdxHq_API
from .his_data_extractor import HisDataExtractor,PeriodType,MarketType

import numpy as np
import pandas as pd
import datetime

from trader.utils.date_util import tradedays

from cus_utils.log_util import AppLogger
logger = AppLogger()

class TdxExtractor(HisDataExtractor):
    """通达信数据源"""

    def __init__(self, backend_channel="tdx",savepath=None):
        
        super().__init__(backend_channel=backend_channel,savepath=savepath)
        # 初始化pytdx的调用接口对象
        self.api = TdxHq_API(auto_retry=True,raise_exception=True)
        self.host = '119.147.212.81'
        self.port = 7709
        # 一次性查询最大限额
        self.maxcnt_once_call = 800
    
    def reconnect(self):
        try:
            self.api.disconnect()
            self.api.connect(self.host,self.port)
        except Exception as e:
            logger.error("reconnect fail:",e)
            
    def load_code_data(self):  
        """取得所有股票代码"""
        pass     
    
    def imoprt_data(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY):
        """
            取得所有股票历史行情数据
            Params:
                task_batch 任务批次号
                begin_date 导入数据的开始日期
                end_date 导入数据的结束日期
                period 频次类别
        """
        
        # 任务开始时初始化连接，后续保持使用此连接
        self.api.connect(self.host,self.port)       
        try: 
            super().imoprt_data(task_batch, start_date, end_date, period)
        except Exception as e:
            logger.error("imoprt_data fail:",e)            
        # 任务结束时关闭连接
        self.api.disconnect()
        
    def load_item_data(self,instrument_code,start_date=None,end_date=None,period=None,market=MarketType.SH.value):   
        """取得单个股票历史行情数据"""
        
        if period<=PeriodType.MIN1.value:
            # 暂时只使用分钟级别加载模式
            logger.warnning("not support period:{}".format(period))
            return
        
        # 此接口只支持2004年以后的数据
        if int(str(start_date)[:4])<2004:
            start_date = 20040101
        if int(str(end_date)[:4])>2023:
            today = datetime.date.today().strftime('%Y%m%d')
            end_date = int(today)
                        
        if period==PeriodType.MIN5.value:
            category = 0
            offset = -800
        if period==PeriodType.MIN15.value:
            category = 1            
        # 计算开始节点编号（前推多少个数量），以及需要查询的K线数量
        before_number,exceed_number = self.compute_period_space(str(start_date), str(end_date), period,offset=offset)
        api = self.api
        inner_batch = 0
        item_data = None
        while True:
            inner_batch += 1
            # 滚动查询，每次减少前推间隔
            data = api.get_security_bars(category,market, instrument_code, before_number, self.maxcnt_once_call)
            before_number -= self.maxcnt_once_call
            if data is None:
                logger.warning("data none,instrument_code:{},inner_batch:{}".format(instrument_code,inner_batch))
                continue
            df_data = api.to_df(data)
            logger.debug("inner_batch:{},data size:{}".format(inner_batch,len(data)))   
            if item_data is None:
                item_data = df_data   
            else:
                item_data = pd.concat([item_data,df_data])
            total_number = inner_batch * self.maxcnt_once_call
            # 如果超出结束期限，退出循环
            if total_number>=before_number or before_number<=0:
                logger.info("exceed number,break:{}".format(instrument_code))
                break
                          
        return item_data
    
    def compute_period_space(self,start_date=None,end_date=None,period=None,offset=0):
        """取得指定日期下的间隔数"""
        
        min_number = 1
        if period==PeriodType.MIN1.value:
            min_number = 1
        if period==PeriodType.MIN5.value:
            min_number = 5    
        if period==PeriodType.MIN15.value:
            min_number = 15    
        
        # 根据不同分钟级别，计算一天需要多少K线
        day_item_number = 4 * 60 / min_number
        # 计算需要向前退多少天，根据当前日期以及开始日期进行计算
        today = datetime.date.today().strftime('%Y%m%d')
        days_to_now = tradedays(start_date,today)
        # 前推数量为天数乘以每天K线数
        before_number = int(day_item_number * days_to_now) + offset
        # 根据结束日期，计算需要查询的K线数量
        days_to_end = tradedays(start_date,end_date)
        item_number = int(day_item_number * days_to_end) + offset
        return before_number,item_number
        
    