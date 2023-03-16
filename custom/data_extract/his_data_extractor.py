# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob

import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
import pickle

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
    
def get_period_name(period_value):
    if period_value==PeriodType.DAY.value:
        return "day"
    if period_value==PeriodType.WEEK.value:
        return "week"
    if period_value==PeriodType.MONTH.value:
        return "month"
    if period_value==PeriodType.MIN1.value:
        return "1m"
    if period_value==PeriodType.MIN5.value:
        return "5m"    
    if period_value==PeriodType.MIN15.value:
        return "15m"
                
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

    def __init__(self, backend_channel="ak",savepath=None):
        """

        Parameters
        ----------
        backend_channel : 采集源    ak: akshare数据源
        """
        if savepath is None:
            savepath="./custom/data/stock_data"
            
        self.savepath = savepath + "/" + backend_channel 
        self.item_savepath = self.savepath + "/item"
        self.backend_channel = backend_channel
        self.dbaccessor = DbAccessor({})
        
        self.busi_columns = ["code","datetime","open","close","high","low","volume","amount","amplitude","flu_range","flu_amount","turnover"]
           
    def create_code_data(self):
        """生成所有股票代码"""
        
        code_data = self.extract_code_data()
        # 把股票列表信息保存到数据库
        for item in code_data:
            sql = "insert into instrument_info(code,name,market) values(%s,%s,%s)"
            self.dbaccessor.do_inserto_withparams(sql, tuple(item))             
        
    def extract_code_data(self):  
        """取得所有股票代码，子类实现"""
        pass
        
              
    def imoprt_data(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value,contain_institution=True):
        """
            取得所有股票历史行情数据
            Params:
                task_batch 任务批次号
                begin_date 导入数据的开始日期
                end_date 导入数据的结束日期
                period 频次类别
                contain_institution 是否包含复权数据
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
            insert_sql = "insert into data_task(task_type,task_batch,backend_channel,start_date,end_date,status,period) values(%s,%s,%s,%s,%s,%s,%s)"
            self.dbaccessor.do_inserto_withparams(insert_sql, 
                        (DataTaskType.DataImport.value,task_batch,self.backend_channel,start_date,end_date,DataTaskStatus.Start.value,period))
        # 股票编码从数据库表中获得
        sql = "select code,market from instrument_info order by code"
        if last_item_code is not None:
            # 断点处继续
            sql = "select code,market from instrument_info where code>='{}' order by code".format(last_item_code)
        result_rows = self.dbaccessor.do_query(sql)            
        savepath = "{}/{}".format(self.item_savepath,get_period_name(period))
        if not os.path.exists(savepath):
            os.makedirs(savepath) 
        
        total_data = None  
        total_data_institution = None  
        for row in result_rows:
            code = row[0]
            market = row[1]
            # 取得相关数据，子类实现
            item_data = self.extract_item_data(code,start_date=start_date,end_date=end_date,period=period,market=market)
            if item_data is not None:
                # 每个股票分别保存csv到本地
                save_file_path = "{}/{}.csv".format(savepath,code)
                item_data.to_csv(save_file_path, index=False)   
                # 合并为一个总DataFrame，最后保存
                if total_data is None:
                    total_data = item_data
                else:
                    total_data = pd.concat([total_data,item_data],axis=0)
            # 复权模式下需要再处理一次
            if contain_institution:
                item_data_institution = self.extract_item_data(code,start_date=start_date,end_date=end_date,period=period,market=market,institution=True)
                if item_data_institution is not None:
                    # 每个股票分别保存csv到本地
                    save_file_path = "{}/{}_institution.csv".format(savepath,code)
                    item_data.to_csv(save_file_path, index=False)   
                    # 合并为一个总DataFrame，最后保存
                    if total_data_institution is None:
                        total_data_institution = item_data_institution
                    else:
                        total_data_institution = pd.concat([item_data_institution,item_data],axis=0)                  
            # 记录最后一条子任务号码，以便后续断点继续
            self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=%s where task_batch=%s", (code,task_batch))
            
        # 最后统一保存一个文件   
        self.save_total_df(total_data,period=period)
        if contain_institution:
            self.save_total_df(total_data_institution,period=period,institution=True)
        # 任务结束后设置状态为已成功  
        self.dbaccessor.do_inserto_withparams("update data_task set status=%s where task_batch=%s", (DataTaskStatus.Success.value,task_batch))
        
    def extract_item_data(self,instrument_code,start_date=None,end_date=None,period=None,institution=True):   
        """取得单个股票历史行情数据,子类实现"""
        pass
    
    def get_total_file_save_path(self,period,institution=False):
        period_name = get_period_name(period)
        if institution:
            return self.savepath + "/all_{}_institution.pickle".format(period_name)
        return self.savepath + "/all_{}.pickle".format(period_name)
        
    def save_total_df(self,df,period=None,institution=False):
        data_file = self.get_total_file_save_path(period,institution=institution)
        with open(data_file, "wb") as fout:
            pickle.dump(df, fout)           
    
    def load_item_df(self,instrument,period=PeriodType.MIN5.value,institution=False):
        """加载单个股票"""
        
        period_name = get_period_name(period)
        item_savepath = self.item_savepath + "/{}".format(period_name)
        if institution:
            f =  "{}/{}_institution.csv".format(item_savepath,instrument)
        else:
            f =  "{}/{}.csv".format(item_savepath,instrument)
        item_df = pd.read_csv(f)  
        # 对时间字段进行检查及清洗
        if self.backend_channel=="tdx":
            item_df["volume"] = item_df["vol"]            
        item_df["datetime"] = pd.to_datetime(item_df["datetime"],errors="coerce")
        item_df = item_df.dropna()
        return item_df
    
    def load_total_df(self,period=PeriodType.MIN5.value):
        """加载之前保存的数据"""
        
        data_file = self.get_total_file_save_path(period)
        # 如果没有保存，则从每个单独数据里加载
        if not os.path.exists(data_file):
            total_df = None
            item_savepath = self.item_savepath + "/{}".format(period)
            csv_files = glob.glob(os.path.join(item_savepath, "*.csv"))
            for f in csv_files:
                try:
                    instrument_code = f.split("/")[-1].split(".")[0]
                    df = self.load_item_df(instrument_code)  
                    df["instrument"] = instrument_code
                    logger.debug("item load suc:{}".format(f))  
                except Exception as e:
                    logger.warning("item load fail:{},reason:{}".format(f,e))  
                    continue
                if total_df is None:
                    total_df = df
                else:
                    total_df = pd.concat([total_df,df],axis=0)    
            # 合并以后保存
            self.save_total_df(total_df,period=period)                
        else:
            with open(data_file, "rb") as fin:
                total_df = pickle.load(fin)            
        return total_df
    
if __name__ == "__main__":    
    from data_extract.akshare_extractor import AkExtractor
    from data_extract.tdx_extractor import TdxExtractor
    # extractor = AkExtractor()   
    # extractor.create_code_data()
    # extractor.imoprt_data(task_batch=0,period=PeriodType.MIN5.value,start_date=20220101,end_date=20221231)
    # extractor = TdxExtractor(savepath="./custom/data/stock_data")
    # extractor.imoprt_data(task_batch=0,period=PeriodType.MIN5.value,start_date=20220101,end_date=20221231)
    extractor = AkExtractor(savepath="./custom/data/stock_data")
    extractor.imoprt_data(task_batch=0,period=PeriodType.DAY.value,start_date=20220101,end_date=20221231)    
        
