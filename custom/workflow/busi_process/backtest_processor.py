import copy
import datetime
import os

from rqalpha import run_file

from .base_processor import BaseProcessor
from trader.utils.date_util import get_first_and_last_day

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
        # 回测部分
        start_date,end_date = get_first_and_last_day(str(working_day)[:4],str(working_day)[4:6])    
        # 回测开始和结束日期为本月第一天和最后一天
        backtest_template["rqalpha"]["base"]["start_date"] = start_date
        backtest_template["rqalpha"]["base"]["end_date"] = end_date
        # 给回测进程植入任务号
        backtest_template["rqalpha"]["extra"]["task_id"] = self.wf_task.task_obj["id"]
        # config_path为当前文件路径
        config_file_path = self.wf_task.get_task_config_file()   
        backtest_template["rqalpha"]["extra"]["context_vars"]["strategy_class"]["config_path"] = config_file_path
        # 相关文件路径
        parent_path = self.wf_task.get_trader_data_path()
        cur_period_path = parent_path + "/" + str(working_day)[4:6]
        backtest_template["rqalpha"]["mod"]["sys_analyser"]["report_save_path"] = cur_period_path
        backtest_template["rqalpha"]["mod"]["ext_ds_mod"]["report_save_path"] = cur_period_path
        return real_template
                            
    def sub_run(self,working_day=None,results=None,resume=True):
        """回测入口，整合glib的工作流配置方式，与rqalpha结合"""

        # rqalpha配置属于整个工作流配置文件的一部分
        rq_config = self.config["task"]["backtest"]["rqalpha"]
        # 统一一个运行策略文件，后面通过不同的实现类来进行策略区分
        strategy_file_path = self.config["task"]["backtest"]["run_file"]
        run_file(strategy_file_path, rq_config)   

        
        