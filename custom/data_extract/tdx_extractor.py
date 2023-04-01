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

    def __init__(self, backend_channel="tdx",savepath=None,**kwargs):
        
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
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

    def import_data_within_workflow(self,wf_task_id,start_date=19700101,end_date=20500101,period=None,
                                    is_complete=False,contain_institution=True,fill_history=False):
        
        # 任务开始时初始化连接，后续保持使用此连接
        self.api.connect(self.host,self.port)       
        try: 
            super().import_data_within_workflow(wf_task_id,start_date=start_date,end_date=end_date,period=period,
                                    is_complete=is_complete,contain_institution=contain_institution,fill_history=fill_history)
        except Exception as e:
            logger.error("import_data fail:",e)            
        # 任务结束时关闭连接
        self.api.disconnect()
                    
    def import_data(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value,contain_institution=False,resume=False):
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
        results = None  
        try: 
            results = super().import_data(task_batch=task_batch, start_date=start_date, end_date=end_date,
                                           period=period,contain_institution=contain_institution,resume=resume)
        except Exception as e:
            logger.exception("tdx import_data fail:")            
        # 任务结束时关闭连接
        self.api.disconnect()
        return results
        
    def extract_item_data(self,instrument_code,start_date=None,end_date=None,period=None,market=MarketType.SH.value,institution=False):   
        """取得单个股票历史行情数据"""
        
        if period<=PeriodType.MIN1.value:
            # 暂时只使用分钟级别加载模式
            logger.warnning("not support period:{}".format(period))
            return
        
        # 此接口只支持2004年以后的数据
        if int(str(start_date)[:4])<2004:
            start_date = 20040101
        # 分钟级别数据，最早只能从2022年开始
        if int(str(start_date)[:4])<2022 and period>=PeriodType.MIN1.value:
            start_date = 20220101            
        if int(str(end_date)[:4])>2023:
            today = datetime.date.today().strftime('%Y%m%d')
            end_date = int(today)
                        
        if period==PeriodType.MIN5.value:
            category = 0
        if period==PeriodType.MIN15.value:
            category = 1            
        # 计算开始节点编号（前推多少个数量），以及需要查询的K线数量
        before_number,exceed_number = self.compute_period_space(str(start_date), str(end_date), period)
        range_number = before_number - exceed_number
        # 计算每个轮次，请求多少个批次的K线
        loop_call_number = range_number if self.maxcnt_once_call>range_number else self.maxcnt_once_call
        # 注意，由于api接口规定的单批次查询数量是往前查，因此这里需要把before_number减去单次查询数
        before_number = before_number - loop_call_number 
        api = self.api
        inner_batch = 0
        item_data = None
        while(True):
            # 滚动查询，每次减少前推间隔.
            # category：0--为5分钟K线 market：0深圳 1上海
            data = api.get_security_bars(category,market, instrument_code, before_number, loop_call_number)
            if data is None or len(data)==0:
                logger.warning("data none,instrument_code:{},inner_batch:{}".format(instrument_code,inner_batch))
                break
            df_data = api.to_df(data)
            # 附加股票代码
            df_data["code"] = instrument_code
            df_data["volume"] = df_data["vol"] 
            if item_data is None:
                item_data = df_data   
            else:
                item_data = pd.concat([item_data,df_data])
            if before_number<=exceed_number:
                break                
            # 如果与结束间隔不足一次循环，则把其余的补上并退出
            if before_number-exceed_number<loop_call_number:
                loop_call_number = before_number-exceed_number                
            before_number -= loop_call_number
        return item_data
    
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
        day_item_number = 4 * 60 / min_number
        # 计算需要向前退多少天，根据当前日期以及开始日期进行计算
        today = datetime.date.today().strftime('%Y%m%d')
        days_to_begin = tradedays(start_date,today)
        # 前推数量为天数乘以每天K线数
        start_number = int(day_item_number * days_to_begin)
        # 根据结束日期，计算需要查询的K线数量
        days_to_end = tradedays(end_date,today)
        end_number = int(day_item_number * days_to_end)
        # exceed_number = before_number - item_number
        return start_number,end_number

        
           
    