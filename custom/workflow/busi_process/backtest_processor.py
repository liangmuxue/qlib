import copy
import datetime
from datetime import timedelta
import os

from rqalpha import run_file

from .base_processor import BaseProcessor
from trader.utils.date_util import get_first_and_last_day,get_tradedays_dur
from workflow.constants_enum import WorkflowStatus,WorkflowSubStatus,FrequencyType,WorkflowType
import cus_utils.global_var as global_var

from cus_utils.log_util import AppLogger
logger = AppLogger()

class BacktestProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        
    def build_real_template(self,template,config=None,working_day=None):
        """根据原配置模板，生成实际配置文件"""
        
        real_template = copy.deepcopy(template)
        backtest_template = real_template["task"]["backtest"]
        model_template = real_template["task"]["model"]
        dataset_template = real_template["task"]["dataset"]
        
        # 设置预测数据路径
        model_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        # 计算开始结束日期
        start_date,end_date = self.get_first_and_last_day(working_day)  
        # start_date = datetime.date(2021,1,day=5)  
        # 回测开始和结束日期为本期（月、季、年）第一天和最后一天
        backtest_template["rqalpha"]["base"]["start_date"] = start_date

        backtest_template["rqalpha"]["base"]["end_date"] = end_date
        dataset_template["kwargs"]["segments"]["classify_range"] = [start_date,end_date]
        # 给回测进程植入任务号
        backtest_template["rqalpha"]["extra"]["task_id"] = self.wf_task.task_entity["id"]
        # config_path为当前文件路径--Cancel
        config_file_path = self.wf_task.get_task_config_file()   
        # backtest_template["rqalpha"]["extra"]["context_vars"]["strategy_class"]["config_path"] = config_file_path
        # 相关文件路径
        parent_path = self.wf_task.get_trader_data_path()
        cur_period_path = parent_path + "/" + str(working_day)[4:6]
        backtest_template["rqalpha"]["extra"]["report_save_path"] = cur_period_path
        backtest_template["rqalpha"]["mod"]["sys_analyser"]["report_save_path"] = cur_period_path
        backtest_template["rqalpha"]["mod"]["ext_ds_mod"]["report_save_path"] = cur_period_path
        # 映射数据文件路径
        backtest_template["rqalpha"]["extra"]["stock_data_path"] = self.wf_task.get_stock_data_path()
        # 映射预测数据文件路径
        backtest_template["rqalpha"]["extra"]["dump_path"] = self.wf_task.get_dumpdata_path()
        # 根据当前序号，如果不够则不进行此次任务
        cur_sequence = self.wf_task.task_entity["sequence"]
        if cur_sequence<1:
            # 设置忽略标志
            self.task_ignore = True        
        return real_template
    
    def get_first_and_last_day(self,working_day):
        month = int(str(working_day)[4:6])
        year = int(str(working_day)[:4])     
        # 周频率，取得本周数据
        if self.wf_task.config["frequency"]==FrequencyType.WEEK.value:
            main_task = self.wf_task.main_task
            today = datetime.datetime.strptime(str(working_day), "%Y%m%d")
            start_date = (today - timedelta(today.weekday())).date()
            end_date = today.date()
                    
        # 月度频率，取得上个月数据
        if self.wf_task.config["frequency"]==FrequencyType.MONTH.value:
            # 回测上个月数据
            month = month - 1
            if month==0:
                month = 12
                year = year - 1     
            start_date,end_date = get_first_and_last_day(year,month)    
        # 季度频率，取得本季度所有数据
        if self.wf_task.config["frequency"]==FrequencyType.QUARTER.value:
            if month<3:
                start_date = datetime.date(year,1,day=1)
                end_date = datetime.date(year,1,day=1)
            else:
                start_month = month - 2
                end_month = month
                start_date = datetime.date(year,start_month,day=1)
                end_date = datetime.date(year,end_month,day=30)                
        # 年度频率，取得本年所有数据
        if self.wf_task.config["frequency"]==FrequencyType.YEAR.value:
            start_date = datetime.date(year,1,day=1)
            end_date = datetime.date(year,12,day=31)         
        # 实时频率，开始和结束都是当天
        if self.wf_task.config["frequency"]==FrequencyType.REAL.value:
            start_date = datetime.datetime.now()
            end_date = datetime.datetime.now()               
        return start_date,end_date     
                           
    def sub_run(self,working_day=None,results=None,resume=True):
        """回测入口，整合glib的工作流配置方式，与rqalpha结合"""

        # rqalpha配置属于整个工作流配置文件的一部分
        rq_config = self.config["task"]["backtest"]["rqalpha"]
        # 统一一个运行策略文件，后面通过不同的实现类来进行策略区分
        strategy_file_path = self.config["task"]["backtest"]["run_file"]
        # 透传总体任务配置
        global_var.set_value("task_config", self.config)
        run_file(strategy_file_path, rq_config)   
        
        