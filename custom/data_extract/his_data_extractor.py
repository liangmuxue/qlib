# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd

from cus_utils.db_accessor import DbAccessor
from cus_utils.log_util import AppLogger

logger = AppLogger()

from enum import Enum, unique

@unique
class MarketType(Enum):
    """市场类别，0深圳 1上海"""
    SH = 1 
    SZ = 0
    
@unique
class PeriodType(Enum):
    """数据频次类别"""
    DAY = 1 
    WEEK = 2
    MONTH = 3
    MIN1 = 4
    MIN5 = 5
    MIN15 = 6

@unique
class DataTaskStatus(Enum):
    """数据任务执行状态"""
    Start = 1 
    Fail = 2
    Success = 3
    
@unique
class DataTaskType(Enum):
    """数据任务类型 1 数据导入"""
    DataImport = 1 


class HisDataExtractor:
    """历史证券数据采集"""

    def __init__(self, backend_channel="ak",savepath="./custom/data/stock_data"):
        """

        Parameters
        ----------
        backend_channel : 采集源    ak: akshare数据源
        """
        
        self.savepath = savepath
        self.item_savepath = savepath + "/" + backend_channel + "/item"
        self.backend_channel = backend_channel
        self.dbaccessor = DbAccessor({})
        
        self.busi_columns = ["code","date","open","close","high","low","volume","amount","amplitude","flu_range","flu_amount","turnover"]
           
    def create_code_data(self):
        """生成所有股票代码"""
        
        code_data = self.load_code_data()
        # 把股票列表信息保存到数据库
        for item in code_data:
            sql = "insert into instrument_info(code,name,market) values(%s,%s,%s)"
            self.dbaccessor.do_inserto_withparams(sql, tuple(item))             
        
    def load_code_data(self):  
        """取得所有股票代码，子类实现"""
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
        
        last_item_code = None
        # 任务记录处理
        if task_batch>0:
            # 如果设置了任务编号，说明需要从之前的任务继续,需要修改之前的任务表状态
            last_item_code = self.dbaccessor.do_query("select last_item_code from data_task where task_batch={}".format(task_batch))[0][0]
            self.dbaccessor.do_inserto_withparams("update data_task set status=%s where task_batch=%s", (DataTaskStatus.Start.value,task_batch))
        else:
            # 新创建一个任务记录
            task_batch = self.dbaccessor.do_query("select max(task_batch) from data_task")[0][0]
            if task_batch is None:
                task_batch = 1
            else:
                task_batch += 1
            insert_sql = "insert into data_task(task_type,task_batch,backend_channel,start_date,end_date,status) values(%s,%s,%s,%s,%s,%s)"
            self.dbaccessor.do_inserto_withparams(insert_sql, 
                        (DataTaskType.DataImport.value,task_batch,self.backend_channel,start_date,end_date,DataTaskStatus.Start.value))
        # 股票编码从数据库表中获得
        sql = "select code,market from instrument_info order by code"
        if last_item_code is not None:
            # 断点处继续
            sql = "select code,market from instrument_info where code>='{}' order by code".format(last_item_code)
        result_rows = self.dbaccessor.do_query(sql)            
        index = 0
        savepath = "{}/{}".format(self.item_savepath,period)
        if not os.path.exists(savepath):
            os.makedirs(savepath) 
            
        for row in result_rows:
            code = row[0]
            market = row[1]
            # 取得相关数据，子类实现
            item_data = self.load_item_data(code,start_date=start_date,end_date=end_date,period=period,market=market)
            if item_data is not None:
                # 每个股票分别保存csv到本地
                save_file_path = "{}/{}.csv".format(savepath,code)
                item_data.to_csv(save_file_path, index=False)   
            # 记录最后一条子任务号码，以便后续断点继续
            self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=%s where task_batch=%s", (code,task_batch))
        # 任务结束后设置状态为已成功  
        self.dbaccessor.do_inserto_withparams("update data_task set status=%s where task_batch=%s", (DataTaskStatus.Success.value,task_batch))
        
    def load_item_data(self,instrument_code,start_date=None,end_date=None,period="day"):   
        """取得单个股票历史行情数据"""
        pass


if __name__ == "__main__":    
    from data_extract.akshare_extractor import AkExtractor
    from data_extract.tdx_extractor import TdxExtractor
    # extractor = AkExtractor()   
    # extractor.create_code_data()
    # extractor.imoprt_data(task_batch=1)
    extractor = TdxExtractor(savepath="./custom/data/stock_data")
    extractor.imoprt_data(task_batch=0,period=PeriodType.MIN5.value,start_date=20220101,end_date=20221231)
    
        
