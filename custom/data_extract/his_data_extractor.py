# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import glob
import shutil  

import numpy as np
import datetime
from tqdm import tqdm
import pandas as pd
import pickle

from cus_utils.db_accessor import DbAccessor
from cus_utils.log_util import AppLogger
from persistence.common_dict import CommonDictType,CommonDict
from pandas._libs.tslibs import period
from trader.utils.date_util import get_tradedays_dur,date_string_transfer
from scripts.dump_bin import DumpDataAll

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

def get_period_value(period_name):
    for item in PeriodType:
        if item.name==period_name:
            return item.value
    return None
                
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

    def __init__(self, backend_channel="ak",savepath=None,**kwargs):
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
        self.common_dict = CommonDict()
        
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
        
    def create_base_info(self):  
        """生成股票的基础信息"""
        
        instrument_list = self.dbaccessor.do_query("select code from instrument_info where delete_flag=0")
        for instrument in instrument_list:
            instrument = instrument[0]
            base_data = self.extract_base_info(instrument)
            industry_dict = {"type": CommonDictType.INDUSTRY_TYPE.value,
                             "code": base_data["industry"],"value": base_data["industry"]}
            # 查询并补充数据字典
            industry_dict_db = self.common_dict.get_or_insert_dict(industry_dict)
            sql = "update instrument_info set industry=%s,total_capital=%s,tradable_shares=%s where code=%s"
            self.dbaccessor.do_inserto_withparams(sql, (industry_dict_db["id"],base_data["total_capital"],base_data["tradable_shares"],instrument))   
            logger.debug("create_base_info instrument ok:{}".format(instrument))

    def extract_base_info(self,instrument):  
        """取得股票的基础信息,子类实现"""
        pass
        
    def export_item_data(self,code,item_data,period=None,is_complete=False,savepath=None,institution=False):
        """生成单个文件，兼顾全量和增量"""
        
        save_file_path = "{}/origin/{}.csv".format(savepath,code)
        if institution:
            save_file_path = "{}/institution/{}.csv".format(savepath,code)        
        if is_complete:
            # 如果是全量，则覆盖
            item_data.to_csv(save_file_path, index=False)   
        else:
            # 如果是增量，则取出原来文件，合并后覆盖
            ori_item_data = self.load_item_df(code,period=period,institution=institution)
            item_data = pd.concat([ori_item_data,item_data],axid=0)
            item_data.to_csv(save_file_path, index=False)        

    def export_whole_item_data(self,period=None,institution=False):
        """把全量df文件拆成每个股票分别存储为csv格式的单独文件"""
        
        df_total = self.load_total_df(period, institution)
        path = self.get_whole_item_datapath(period,institution=institution)
        if not os.path.exists(path):
            os.makedirs(path) 
            
        group_column = "code"
        for group_name,group_data in df_total.groupby(group_column):
            if isinstance(group_name, int):
                group_name = self.code_transfer_to_string(group_name)
            
            save_file_path = "{}/{}.csv".format(path,group_name)  
            group_data.to_csv(save_file_path, index=False)     
    
    def get_whole_item_datapath(self,period,institution=False):
        period_name = get_period_name(period)
        if institution:
            return "{}/item/{}/institution".format(self.savepath,period_name)
        return "{}/item/{}/origin".format(self.savepath,period_name)
    
    def export_to_qlib(self,qlib_dir,period,file_name="all.txt",institution=False):
        """csv格式的单独文件导入到qlib"""
        
        source_path = self.get_whole_item_datapath(period,institution=institution)
        dump_all = DumpDataAll(csv_path=source_path,qlib_dir=qlib_dir,date_field_name="datetime")
        # 设置文件名
        dump_all.INSTRUMENTS_FILE_NAME = file_name
        dump_all.dump()
                    
    def get_last_local_data_date(self,code,savepath=None,period=None):            
        """取得本地数据里最后一天"""
        
        item_data = self.load_item_df(code,period=period)        
        if item_data is None or item_data.shape[0]==0:
            return self.get_init_begin_date()
        return item_data["datetime"].max()
    
    def get_init_begin_date(self):    
        return "19700101"
    
    def prepare_import_batch(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value):    
        """任务记录处理"""
        
        if task_batch>0:
            # 如果设置了任务编号，说明需要从之前的任务继续,需要修改之前的任务表状态
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
        return task_batch
 
    def import_data(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value,is_complet=False,
                    contain_institution=False,resume=False,no_total_file=False,auto_import=False):
        """
            取得所有股票历史行情数据
            Params:
                task_batch 任务批次号
                begin_date 导入数据的开始日期
                end_date 导入数据的结束日期
                period 频次类别
                contain_institution 是否包含复权数据
                auto_import 是否自动计算日期
                ori_data 原数据
        """
                
        task_batch = self.prepare_import_batch(task_batch, start_date, end_date, period)
       
        if auto_import:
            return self.import_data_auto(task_batch=task_batch,end_date=end_date,period=period,
                                     contain_institution=contain_institution,no_total_file=no_total_file)
        return self.import_data_part(task_batch=task_batch,start_date=start_date,end_date=end_date,period=period,is_complet=is_complet,
                                     contain_institution=contain_institution,resume=resume,no_total_file=no_total_file)

    def import_data_auto(self,task_batch=0,end_date=20500101,period=PeriodType.DAY.value,
                         contain_institution=False,no_total_file=False):
        """取得所有股票历史行情数据,自动根据原有数据日期匹配导入日期范围"""
        
        # 股票编码从数据库表中获得
        sql = "select code,market from instrument_info where delete_flag=0 order by code "
        result_rows = self.dbaccessor.do_query(sql)            
        savepath = "{}/{}".format(self.item_savepath,get_period_name(period))
        total_data = None
        total_data_institution = None
        if not os.path.exists(savepath):
            os.makedirs(savepath) 
        
        # 准备全量文件，用于后续日期筛选
        if not no_total_file:
            ori_data = self.load_total_df(period) 
            if contain_institution:
                ori_data_institution = self.load_total_df(period,institution=True) 
        for row in result_rows:
            code = row[0]
            market = row[1]       
            if not no_total_file:
                # 从全量文件里摘出当前股票数据
                ori_data_item = ori_data[ori_data["code"]==code]
                if contain_institution:
                    ori_data_item_institution = ori_data_institution[ori_data_institution["code"]==code]   
            else:
                # 从单独数据文件中加载
                try:
                    ori_data_item = self.load_item_df(code, period=period) 
                except Exception as e:
                    logger.warning("load_item_df fail:{}".format(e)) 
                    ori_data_item = None
                if contain_institution:
                    try:
                        ori_data_item_institution = self.load_item_df(code, period=period,institution=True)   
                    except Exception as e:
                        logger.warning("load_item_df fail:{}".format(e)) 
                        ori_data_item_institution = None                    
            # 单个股票数据抽取         
            item_data = self.data_part_auto_process(code,end_date=end_date,period=period,market=market,savepath=savepath,
                                                     institution=False,ori_data_item=ori_data_item)
            if item_data is None:
                continue
            logger.debug("item data ok:{}".format(code))
            self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=%s where task_batch=%s", (code,task_batch))
            # 合并所有的股票数据
            if total_data is None:
                total_data = item_data
            else:
                total_data = pd.concat([total_data,item_data])
            if contain_institution:
                item_data_institution = self.data_part_auto_process(code,end_date=end_date,period=period,market=market,savepath=savepath,
                                                    institution=True,ori_data_item=ori_data_item_institution)
                if total_data_institution is None:
                    total_data_institution = item_data_institution
                else:
                    total_data_institution = pd.concat([total_data_institution,item_data_institution])     
        # 保存全量文件    
        if not no_total_file: 
            self.save_total_df(total_data,period=period)    
            if contain_institution:
                self.save_total_df(total_data_institution,period=period,institution=True)      
        self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=%s where task_batch=%s", (code,task_batch))
        return (total_data,total_data_institution)
        
    def data_part_auto_process(self,code,end_date=None,period=None,savepath=None,market=None,institution=False,ori_data_item=None):
        """数据抽取，根据原有数据自行计算开始日期"""
 
        if ori_data_item is None:
            start_date = self.get_first_default_date()
            origin_data = None
        else:
            # 根据原有数据，取得最近日期，并从下一天作为开始日期
            start_date = self.get_last_data_date(ori_data_item,period)
            # 清除原来数据冗余的部分
            origin_data = self.clear_redun_data(ori_data_item,start_date)
        # 取得相关数据，子类实现
        item_data = self.extract_item_data(code,start_date=start_date,end_date=end_date,period=period,market=market,institution=institution)
        if item_data is None:
            logger.info("item_data None:{}".format(code))
            return origin_data
        if ori_data_item is None:    
            total_data = item_data
        else:
            total_data = pd.concat([origin_data,item_data],axis=0)
        # 保存csv数据文件
        self.export_item_data(code,total_data,is_complete=True,savepath=savepath,period=period,institution=institution)  
        return total_data
    
    def get_first_default_date(self):
        return "20080101"
        
    def get_last_data_date(self,data_item,period):    
        """取得存储数据中的最近日期"""
        
        cur_date = data_item["datetime"].max()
        tar_date = get_tradedays_dur(date_string_transfer(cur_date,direction=2),1)
        tar_date = tar_date.strftime("%Y%m%d")
        return tar_date     
     
    def clear_redun_data(self,ori_data_item,date):
        return ori_data_item
           
    def import_data_part(self,task_batch=0,start_date=19700101,end_date=20500101,period=PeriodType.DAY.value,is_complete=False,
                         contain_institution=False,resume=False,no_total_file=False):
        """取得所有股票历史行情数据,去除批次号部分"""
        
        last_item_code = self.dbaccessor.do_query("select last_item_code from data_task where task_batch={}".format(task_batch))[0][0]
        # 股票编码从数据库表中获得
        sql = "select code,market from instrument_info where delete_flag=0 order by code "
        if last_item_code is not None:
            # 断点处继续
            sql = "select code,market from instrument_info where delete_flag=0 and code>='{}' order by code".format(last_item_code)
        result_rows = self.dbaccessor.do_query(sql)            
        savepath = "{}/{}".format(self.item_savepath,get_period_name(period))
        if not os.path.exists(savepath):
            os.makedirs(savepath) 
            
        # 恢复模式下，需要先加载之前的已生成数据
        if resume and not no_total_file:
            total_data = self.load_total_df(period,force_load_item=True)
            total_data_institution = self.load_total_df(period, institution=True,force_load_item=False)
        else:
            total_data = None  
            total_data_institution = None  
        for row in result_rows:
            code = row[0]
            market = row[1]
            # 如果本地记录里包含，则跳过
            if total_data is None or np.sum(total_data["code"]==int(code))==0:
                total_data = self.data_part_process(code,total_data, start_date=start_date,
                                    end_date=end_date,period=period,market=market,savepath=savepath,institution=False,no_total_file=no_total_file)
            else:
                logger.debug("has data:{}".format(code))
            # 复权模式下需要再处理一次
            if contain_institution:
                if total_data_institution is None or np.sum(total_data_institution["code"]==int(code))==0:
                    total_data_institution = self.data_part_process(code,total_data_institution,start_date=start_date,
                                    end_date=end_date,period=period,market=market,savepath=savepath,institution=True,no_total_file=no_total_file)
                else:
                    logger.debug("has institution data:{}".format(code))
            logger.info("import data loop:{}".format(code))               
            # 记录最后一条子任务号码，以便后续断点继续
            self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=%s where task_batch=%s", (code,task_batch))
            
        if not no_total_file:
            # 最后统一保存一个文件   
            self.save_total_df(total_data,period=period)
            if contain_institution:
                self.save_total_df(total_data_institution,period=period,institution=True)                
        # 任务结束后设置状态为已成功  
        self.dbaccessor.do_inserto_withparams("update data_task set status=%s where task_batch=%s", (DataTaskStatus.Success.value,task_batch))
        return (total_data,total_data_institution)
    
    def data_part_process(self,code,total_data,start_date=None,end_date=None,period=None,market=None,savepath=None,institution=False,no_total_file=False):
        # 取得相关数据，子类实现
        item_data = self.extract_item_data(code,start_date=start_date,end_date=end_date,period=period,market=market,institution=institution)
        # 如果使用股票文件独立加载，则在此加载csv文件,并合并刚刚下载的数据
        if no_total_file:
            try:
                item_df = self.load_item_df(code,period=period,institution=institution) 
                # 去重，如果之前的日期和之后导入的数据有日期有重合，则删除之前的重复部分
                item_filter_df = item_df[~item_df["datetime"].isin(item_data["datetime"])]
                item_data = pd.concat([item_filter_df,item_data])
            except Exception as e:
                logger.error("load_item_df fail:{},err:{}".format(code,e))
            
        if item_data is not None:
            # 每个股票分别保存csv到本地
            self.export_item_data(code,item_data,is_complete=True,savepath=savepath,period=period,institution=institution)  
            # 合并为一个总DataFrame，最后保存
            if total_data is None:
                total_data = item_data
            else:
                total_data = pd.concat([total_data,item_data],axis=0)
        return total_data
                    
    def clear_local_data(self,period,data_task_batch):  
        """清除本地数据""" 
        
        # 清除子目录所有单独文件
        item_savepath = self.item_savepath + "/{}".format(get_period_name(period))
        if os.path.exists(item_savepath):
            shutil.rmtree(item_savepath)
        # 清除主文件
        total_file = self.get_total_file_save_path(period)
        institution_total_file = self.get_total_file_save_path(period,institution=True)
        if os.path.exists(total_file):
            os.remove(total_file)
        if os.path.exists(institution_total_file):
            os.remove(institution_total_file)    
        # 清除数据库中的子记录标识
        self.dbaccessor.do_inserto_withparams("update data_task set last_item_code=0 where task_batch=%s", (data_task_batch))
                
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
            f = "{}/institution/{}.csv".format(item_savepath,instrument)
        else:
            f = "{}/origin/{}.csv".format(item_savepath,instrument)
        item_df = pd.read_csv(f)  
        # 对时间字段进行检查及清洗
        if self.backend_channel=="tdx":
            item_df["volume"] = item_df["vol"]            
        item_df["datetime"] = pd.to_datetime(item_df["datetime"],errors="coerce")
        item_df = item_df.dropna()
        return item_df
    
    def load_total_df(self,period=PeriodType.MIN5.value,institution=False,force_load_item=False):
        """加载之前保存的数据"""
        
        data_file = self.get_total_file_save_path(period,institution=institution)
        period_name = get_period_name(period)
        # 如果全灵没有保存，则从每个单独数据里加载，如果设置了强制加载，则也从单项记录中加载
        if not os.path.exists(data_file) or force_load_item:
            total_df = None
            item_savepath = self.item_savepath + "/{}".format(period_name)
            csv_files = glob.glob(os.path.join(item_savepath, "*.csv"))
            for f in csv_files:
                try:
                    instrument_code = f.split("/")[-1].split(".")[0]
                    if "_institution" in instrument_code:
                        continue                    
                    df = self.load_item_df(instrument_code,period=period,institution=institution)  
                    df["instrument"] = instrument_code
                    if institution:
                        logger.debug("item load suc:{}_institution".format(f))  
                    else:
                        logger.debug("item load suc:{}".format(f)) 
                except Exception as e:
                    logger.warning("item load fail:{},reason:{}".format(f,e))  
                    continue
                if total_df is None:
                    total_df = df
                else:
                    total_df = pd.concat([total_df,df],axis=0)    
            # 合并以后保存
            self.save_total_df(total_df,period=period,institution=institution)                
        else:
            with open(data_file, "rb") as fin:
                total_df = pickle.load(fin)            
        return total_df
    
    def code_transfer_to_string(self,int_code):
        return str(int_code).zfill(6)
        
if __name__ == "__main__":    
    
    from data_extract.akshare_extractor import AkExtractor
    from data_extract.tdx_extractor import TdxExtractor
    
    # 导入股票代码
    # extractor = AkExtractor()   
    # extractor.create_code_data()
    # extractor.import_data(task_batch=0,period=PeriodType.MIN5.value,start_date=20220101,end_date=20221231)
    # 导入分钟数据
    extractor = TdxExtractor(savepath="./custom/data/stock_data")
    extractor.import_data(task_batch=0,period=PeriodType.MIN5.value,start_date=20220101,end_date=20221231)
    # 导入日K数据
    # extractor = AkExtractor(savepath="./custom/data/stock_data")
    # extractor.import_data(task_batch=0,period=PeriodType.DAY.value,start_date=20220101,end_date=20221231)    
    # 导入基础信息
    # extractor.create_base_info()
        
