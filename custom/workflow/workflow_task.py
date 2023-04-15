# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import glob
import yaml
import time
import datetime as dt
from datetime import datetime

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

from cus_utils.db_accessor import DbAccessor
from cus_utils.log_util import AppLogger
from persistence.common_dict import CommonDictType,CommonDict
from persistence.wf_task_store import WfTaskStore
from workflow.constants_enum import WorkflowStatus,WorkflowSubStatus,FrequencyType,WorkflowType
from workflow.busi_process.train_processor import TrainProcessor
from workflow.busi_process.predict_processor import PredictProcessor
from workflow.busi_process.pred_result_processor import PredResultProcessor
from workflow.busi_process.backtest_processor import BacktestProcessor
from workflow.busi_process.offer_processor import OfferProcessor
from workflow.busi_process.data_processor import DataProcessor
from workflow.busi_process.classify_processor import ClassifyProcessor
from trader.utils.date_util import get_tradedays_dur,get_tradedays
from cus_utils.data_filter import list_split

logger = AppLogger()

from enum import Enum, unique


class WorkflowTask(object):
    """工作流任务类"""

    def __init__(self, task_batch=0,workflow_name=None,resume=False):
        """

        Parameters
        ----------
        task_batch : 任务批次号,如果是0，则新建任务，否则继续之前的任务
        workflow_id: 对应的工作流配置标识
        resume: 是否延续之前的任务
        """
        
        self.task_batch = task_batch
        self.workflow_name = workflow_name
        self.resume = resume
        self.db_accessor = DbAccessor({})
        self.task_store = WfTaskStore()
        # 任务数据准备
        self.prepare_task_data()
        
    def prepare_task_data(self): 
        
        # 取得工作流配置
        self.workflow_id = self.task_store.get_workflow_id_by_name(self.workflow_name)
        self.config = self.task_store.get_workflow_config(self.workflow_id)
        # 准备任务数据
        if self.task_batch==0:
            # 如果从新开始，则创建新的任务流程
            self.task_obj = self.task_store.create_workflow_task(self.workflow_id)
            self.task_batch = self.task_obj["task_batch"]
        else:
            # 使用原任务流程
            self.task_obj = self.task_store.get_workflow_task(self.task_batch)
            if not self.resume:
                # 如果重新开始原来的流程，则需要清空相关记录
                self.task_store.clear_task_state(self.task_batch)
    
    def get_work_dir(self):
        return self.config["work_dir"]
    
    def get_task_work_dir(self):
        twd = "{}/task".format(self.get_work_dir())
        return twd
        
    def start_task(self):
        """开始执行任务"""
        
        # 生成任务目录
        work_dir = self.get_task_work_dir()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)         
        # 修改状态为已运行
        self.task_store.update_workflow_task_status(self.task_obj["task_batch"],WorkflowStatus.running.value)
        # 取得所有子任务，并依次执行
        detail_configs = self.task_store.get_workflow_details_config(self.workflow_id)
        if not self.resume:
            # 如果不是任务恢复，则初始化数据库任务表中的当前工作日，设置为工作流计划中的开始日期
            self.task_store.update_workflow_working_day(self.task_obj["task_batch"],self.config["start_date"])
            # 生成周任务的工作日历
            self.split_period_region(self.task_obj["id"],self.config["start_date"],self.config["end_date"])
            # 还需要清除相关配置文件，以及产生的中间数据     
            self.clear_data()   
        # 以当前日期为基准，循环执行任务
        start_date = self.task_store.get_task_working_day(self.task_obj["task_batch"])
        # 初始化子任务定义，只在开始的时候进行
        subtask_list = []
        for sub_config in detail_configs:
            sub_task = WorkflowSubTask(self,config=sub_config)
            self.prepare_subtask_env(sub_task,sub_config,start_date)
            subtask_list.append(sub_task)
        # 记录到本地变量，后续可以提取
        self.subtask_list = subtask_list
        g_flag = True
        next_working_day = 0
        while True:
            if next_working_day>0:
                working_day = next_working_day
                # 更新数据库记录中的当前工作日期
                self.task_store.update_workflow_working_day(self.task_obj["task_batch"],working_day)                
            else:
                # 开始的时候使用起始日期作为工作日
                working_day = start_date
            logger.info("working today:{}".format(working_day))
            for sub_task in subtask_list:
                if next_working_day>0:
                    if sub_task.config["dynamic_config"]==1:
                        # 如果为动态配置模式，则每次都需要再次生成配置文件，以及进行数据初始化
                        self.prepare_subtask_env(sub_task,sub_task.config,working_day)
                    else:
                        # 非动态配置，每次只需要初始化相关工作环境，不需要生成配置文件
                        task_entity = self.subtask_working_init(sub_task,sub_task.config,working_day)   
                        if self.status_judge(task_entity["status"]):
                            continue
                else:        
                    task_entity = self.task_store.get_subtask(main_task_id=self.task_obj["id"],config_id=sub_task.config["id"],working_day=working_day)
                    # 如果之前的状态是已成功，则跳过
                    if self.status_judge(task_entity["status"]):
                        continue    
                # 启动子任务         
                flag = sub_task.start_task(working_day,resume=self.resume)
                # 出错后终止进程
                if flag==WorkflowSubStatus.fail.value:
                    g_flag = False
                    break  
            if not g_flag:
                break
            # 任务执行完毕后，进入下一天
            next_working_day = self.next_working_day(working_day)
            # 如果还是同一天，则等待
            if next_working_day<=working_day:
                logger.debug("wait for next day:",datetime.now())
                time.sleep(60)
            # 如果超出结束日期，则退出
            if next_working_day>self.config["end_date"]:
                break
    
    def status_judge(self,status):
        if status==WorkflowSubStatus.success.value or status==WorkflowSubStatus.freq_ignore.value or status==WorkflowSubStatus.busi_ignore.value:
            return True
        return False
    
    def get_subtask_by_type(self,task_type_id):
        """根据任务类型，获取对应的子任务"""
        
        for sub_task in self.subtask_list:
            if sub_task.config["type"]==task_type_id:
                return sub_task
    
    def next_working_day(self,working_day):
        """取得下一工作日"""
        
        if self.config["type"]==WorkflowType.backtest.value:
            # 回测模式，直接取下一天
            working_day = datetime.strptime(str(working_day),'%Y%m%d').date()
            next_day = working_day + dt.timedelta(days=1)
            # next_day = get_tradedays_dur(str(working_day),1)
            next_day = next_day.strftime("%Y%m%d")  
        else:
            # 实盘模式，取当前时间
            next_day = datetime.now().strftime("%Y%m%d")  
        return int(next_day)

    def prepare_subtask_env(self,sub_task,sub_config,working_day):
        """准备子任务环境，使用开始日期作为工作日进行初始化设置"""
        
        # 检查是否已存在子任务，如果已存在，则续用.如不存在则生成
        task_entity = self.task_store.get_subtask(main_task_id=self.task_obj["id"],config_id=sub_config["id"],working_day=working_day)
        if task_entity is None:
            # 生成子任务
            task_entity = self.task_store.create_subtask(self.task_obj,sub_config,working_day)
            sub_task.task_entity = task_entity      
            # 生成实际处理类
            sub_task.build_processor()    
            # 处理配置文件，置换可配置项
            sub_task.process_config_file(sub_config,working_day=working_day)                   
        else:
            if task_entity["sequence"]==0:
                task_entity["sequence"] = 1
                init_sequence = True
            else:
                init_sequence = False
            sub_task.task_entity = task_entity
            # 重新设置状态
            if task_entity["status"]==WorkflowSubStatus.success.value:     
                # 如果之前的状态是已成功，则不处理
                pass
            else:
                self.task_store.update_workflow_subtask_status(task_entity["id"],WorkflowSubStatus.created.value,init_sequence=init_sequence)  
            # 生成实际处理类
            sub_task.build_processor()
            if self.resume:
                # 如果是恢复模式，仍然需要生成配置文件
                sub_task.process_config_file(sub_config,working_day=working_day)                    
        
    def subtask_working_init(self,sub_task,sub_config,working_day):
        """每日工作环境准备"""
        
        # 检查是否已存在子任务，如果已存在，则续用.如不存在则生成
        task_entity = self.task_store.get_subtask(main_task_id=self.task_obj["id"],config_id=sub_config["id"],working_day=working_day)
        if task_entity is None:
            # 生成子任务数据库记录
            task_entity = self.task_store.create_subtask(self.task_obj,sub_config,working_day)   
            sub_task.task_entity = task_entity                   
        else:
            # 如果之前的状态是已成功，则不处理
            if task_entity["status"]!=WorkflowSubStatus.success.value:
                # 重新设置数据库状态
                self.task_store.update_workflow_subtask_status(task_entity["id"],WorkflowSubStatus.created.value)  
                sub_task.task_entity = task_entity    
        return task_entity

    def split_period_region(self,task_id,start_date,end_date,batch_size=5):
        """根据工作流日期范围，切割为指定长度的多个任务批次,长度一般为5个交易日"""
        
        # 取得区间内所有交易日
        cal_list = get_tradedays(str(start_date),str(end_date))
        # 按照周长度切分为多个批次
        batch_list = list_split(cal_list, batch_size)
        # 入库
        self.task_store.create_workflow_task_calendar(task_id,batch_list)
        return batch_list
            
    def clear_data(self):
        """清除目录文件"""
        
        task_path = "{}/task/{}".format(self.config["work_dir"],self.task_batch) 
        if os.path.exists(task_path):
            shutil.rmtree(task_path)
                
class WorkflowSubTask(object):
    """工作流子任务类"""

    def __init__(self, main_task,config=None):
        """

        Parameters
        ----------
        main_task : 主任务对象
        """
        
        self.main_task = main_task
        self.config = config
        self.common_dict = CommonDict()
        self.processor = None

    def get_main_dir(self):
        return self.main_task.config["work_dir"]
    
    def get_template_filepath(self,template_name):
        """模板文件路径获取"""
        
        # 拼接规范，主目录+子配置模板目录
        template_filepath = "{}/template/{}".format(self.get_main_dir(),template_name) 
        return template_filepath
    
    def get_task_config_filepath(self):
        """任务配置文件路径获取"""
        
        # 拼接规范，主目录+主任务批次
        config_filepath = "{}/task/{}/config".format(self.get_main_dir(),self.main_task.task_batch) 
        his_filepath = config_filepath + "/history"
        return config_filepath,his_filepath


    def get_task_config_file(self,file_name=None,working_day=None):
        """任务配置文件获取"""
        
        config_path,his_filepath = self.get_task_config_filepath()
        if file_name is None:
            file_name = self.config["name"]
        if working_day is not None:
            config_file = "{}/{}_{}.yaml".format(his_filepath,file_name,working_day)
        else:
            config_file = "{}/{}.yaml".format(config_path,file_name)
        return config_file
           
    def process_config_file(self,config,working_day=None):      
        """处理配置文件"""
        
        template_name = config["main_yaml_template"]
        template_filepath = self.get_template_filepath(template_name)
        with open(template_filepath, "r") as f:
            template = yaml.safe_load(f)    
            # 生成实际配置文件
            template_real = self.processor.build_real_template(template,config=config,working_day=working_day)
            # 保存到任务路径中
            config_path,his_filepath = self.get_task_config_filepath()
            if not os.path.exists(config_path):
                os.makedirs(config_path)
            if not os.path.exists(his_filepath):   
                os.makedirs(his_filepath)
            config_file = self.get_task_config_file()   
            config_file_working_day = self.get_task_config_file(working_day=working_day) 
            # 保存修改后的配置文件 
            with open(config_file, "w") as f:
                yaml.dump(template_real, f)     
            # 同时保留当天的配置文件           
            with open(config_file_working_day, "w") as f:
                yaml.dump(template_real, f)  
                
    def get_dumpdata_path(self):
        """内部数据文件存储路径"""
        
        filepath = "{}/task/{}/dump_data".format(self.get_main_dir(),self.main_task.task_batch) 
        return filepath

    def get_dumpdata_part_path(self):
        """内部数据文件存储路径"""
        
        filepath = "{}/task/{}/dump_data/pred_part".format(self.get_main_dir(),self.main_task.task_batch) 
        return filepath

    def get_pred_data_part_path(self,base_file_name,working_day):
        """预测结果存储文件相对路径"""
        
        return "pred_part/{}".format(base_file_name)
       
    def get_pred_data_part_filepath(self,base_file_name,working_day):
        """预测结果存储文件路径"""
        
        return "{}/pred_part/{}".format(self.get_dumpdata_path(),self.get_pred_data_file_part(base_file_name,working_day))
       
    def get_pred_data_file_part(self,base_file_name,working_day):
        """预测结果存储文件名"""
        
        # 把原文件名扩展为包含日期后缀的文件名
        base_file = base_file_name.split(".")[0]
        base_file_ext = base_file_name.split(".")[-1]
        return "{}_{}.{}".format(base_file,working_day,base_file_ext)
    
    def get_stock_data_path(self):
        """数据文件路径"""
        
        filepath = "{}/stock_data".format(self.get_main_dir()) 
        return filepath 
    
    def get_trader_data_path(self):
        """数据文件路径"""
        
        filepath = "{}/trader_data".format(self.get_main_dir()) 
        return filepath 
    
       
    def get_model_path(self):
        """模型文件存储路径"""
        
        filepath = "{}/task/{}/model".format(self.get_main_dir(),self.main_task.task_batch) 
        return filepath
    
    def get_model_name(self,working_day=None):
        """模型文件名称,命名规范中添加当前日期"""
        
        month_str = str(working_day)[:6]
        name = self.config["name"] + "_{}".format(month_str)
        return name
     
    def get_matched_model_file_name(self,working_day,task_type=None): 
        """取得与指定工作日匹配的模型文件名称"""
        
        month_str = str(working_day)[:6]
        if task_type is not None:
            type_name = task_type
        else:
            type_name = self.config["name"]
        name = type_name + "_{}".format(month_str)
        return name

    def get_subtask_by_type(self,task_type_code):
        """根据任务类型，获取对应的子任务"""
        
        task_type_dict = self.common_dict.get_dict_by_type_and_code(CommonDictType.WORK_TYPE.value,task_type_code)
        return self.main_task.get_subtask_by_type(task_type_dict["id"])
    
    def get_subtask_by_seq(self,sequence):
        
        main_task_id = self.main_task.task_obj["id"]
        config_id = self.config["id"]
        sub_task = self.main_task.task_store.get_subtask_by_type_and_seq(main_task_id,config_id,sequence)
        return sub_task

    def get_calendar_by_seq(self,sequence):
        
        main_task_id = self.main_task.task_obj["id"]
        day_list = self.main_task.task_store.get_workflow_task_calendar(main_task_id,sequence)
        return day_list
            
    ########################################################   流程处理部分   #########################################################                                 
    def build_processor(self):    
        """生成实际处理类"""
        
        if self.processor is not None:
            return self.processor
        
        dict_code = self.common_dict.get_dict_by_id(self.config["type"])["code"]
        
        if dict_code=="data":
            processor = DataProcessor(self)        
        if dict_code=="train":
            processor = TrainProcessor(self)
        if dict_code=="predict":
            processor = PredictProcessor(self)
        if dict_code=="pred_result":
            processor = PredResultProcessor(self)
        if dict_code=="classify":
            processor = ClassifyProcessor(self)
        if dict_code=="backtest":
            processor = BacktestProcessor(self)            
        if dict_code=="offer":
            processor = OfferProcessor(self)
        
        self.processor = processor          
    
    def start_task(self,working_day,resume=True):
        """开始执行任务"""
        
        self.working_day = working_day
        # 任务频次类型筛选，如果不符合当天的任务频次，则忽略此任务
        frequency_types = self.get_frequency_types(str(working_day))
        if self.config["frequency"] not in frequency_types:
            self.task_ignore_handler(ignore_status=WorkflowSubStatus.freq_ignore.value)
            return WorkflowSubStatus.freq_ignore.value        

        # 在任务主表中设置当前子任务
        self.main_task.task_store.update_workflow_current_task(self.main_task.task_obj["task_batch"],self.task_entity["id"])           
        # 根据配置文件，执行实际任务
        flag = self.processor.run(self.get_task_config_file(),working_day=working_day,resume=resume)
        return flag
    
    def get_frequency_types(self,working_day):   
        """判断频次类型"""
        
        working_day_date = datetime.strptime(working_day, '%Y%m%d')
        season = [1,4,7,10]
        rtn_list = []
        # 肯定是日任务
        rtn_list.append(FrequencyType.DAY.value)
        # 根据工作流日历，判断当天是否开启周任务
        cal_obj = self.main_task.task_store.get_workflow_task_firstday_calendar(self.main_task.task_obj["id"],working_day)
        if cal_obj is not None:
            rtn_list.append(FrequencyType.WEEK.value)
        # 每个与1号执行月任务
        if working_day_date.day==1:
            rtn_list.append(FrequencyType.MONTH.value)      
        # 每个季度1号执行季度任务
        if working_day_date.month in season and working_day_date.day==1:
            rtn_list.append(FrequencyType.QUARTER.value)
        return rtn_list          
    
    
    ########################################################   业务处理部分   #########################################################
    def attach_busi_task(self,busi_task_id):
        """挂接对应工作流子任务"""
        
        self.main_task.task_store.attach_busi_task_id(self.task_entity["id"],busi_task_id)
    
    def get_relate_dt_batch(self):
        busi_task_id = self.main_task.task_store.get_busi_task_id(self.task_entity["id"])
        if busi_task_id is None:
            busi_task_id = 0
        return busi_task_id
            
    ########################################################   Hook回调部分   #########################################################
    
    def task_start_handler(self,processor=None):
        """任务开始事件回调"""
        
        # 修改状态为已运行
        self.main_task.task_store.update_workflow_subtask_status(self.task_entity["id"],WorkflowSubStatus.running.value)         

    def task_sucess_handler(self,processor=None):
        """任务完成事件回调"""
        
        # 修改状态为已完成
        self.main_task.task_store.update_workflow_subtask_status(self.task_entity["id"],WorkflowSubStatus.success.value) 

    def task_ignore_handler(self,processor=None,ignore_status=WorkflowSubStatus.freq_ignore.value):
        """任务完忽略事件回调"""
        
        # 修改状态为已忽略,如果是周期原因，需要重置序号
        reset_sequence = False
        if ignore_status==WorkflowSubStatus.freq_ignore.value:
            reset_sequence = True
        self.main_task.task_store.update_workflow_subtask_status(self.task_entity["id"],ignore_status,reset_sequence=reset_sequence) 
        # 删除之前生成的历史配置文件
        config_file_working_day = self.get_task_config_file(working_day=self.working_day) 
        if os.path.exists(config_file_working_day):
            os.remove(config_file_working_day)         
        
    def task_fail_handler(self,processor=None):
        """任务失败事件回调"""
        
        # 修改状态为已失败
        self.main_task.task_store.update_workflow_subtask_status(self.task_entity["id"],WorkflowSubStatus.fail.value) 
                   
if __name__ == "__main__":    
    
    # For Test
    # task = WorkflowTask(task_batch=73,workflow_name="wf_test",resume=True)
    # task = WorkflowTask(task_batch=0,workflow_name="wf_test",resume=False)
    
    # 全量导入，任务只进行一次
    # task = WorkflowTask(task_batch=0,workflow_name="wf_data_import_complete",resume=False)
    # task = WorkflowTask(task_batch=1,workflow_name="wf_data_import_complete",resume=True)
    
    # 回测工作流
    # task = WorkflowTask(task_batch=0,workflow_name="wf_backtest_flow",resume=False)
    task = WorkflowTask(task_batch=75,workflow_name="wf_backtest_flow",resume=True)    
    
    task.start_task()
        
