import copy
import pandas as pd
from datetime import datetime
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config

import json

from .base_processor import BaseProcessor
from cus_utils.db_accessor import DbAccessor
from workflow.constants_enum import LocalDataSourceType
from data_extract.his_data_extractor import get_period_value
from trader.utils.date_util import get_tradedays_dur

import warnings

class FuturesDataProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        self.db_accessor = DbAccessor({})
        # 设置全量导入完成标志
        self.has_finish_complete_import = False
        # warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
     
    def build_real_template(self,template,working_day=None,config=None):
        """根据原配置模板，生成实际配置文件"""
        
        real_template = copy.deepcopy(template)
        extend_config = self.wf_task.config["extend_config"]
        self.auto_import = False
        # 查看附加配置
        if extend_config is not None:
            cfg = json.loads(extend_config)
            # 自动导入的选项
            self.auto_import = bool(cfg["auto_import"])
            # 失败后是否重新进行
            self.restart_after_fail = bool(cfg["restart_after_fail"])

        return real_template
                            
    def sub_run(self,working_day=None,results=None,resume=True):
        """执行数据导入业务"""
 
        task_config = self.config.get("task")
        import_tasks = task_config.get("import_task", [])
        if isinstance(import_tasks, dict):  # prevent only one dict
            import_tasks = [import_tasks]
        # 逐个执行导入任务
        for config in import_tasks:
            period = get_period_value(config["period"])
            # 从工作流子任务中查找关联业务任务编号
            data_task_batch = self.wf_task.get_relate_dt_batch()
            # 存储路径使用数据库中的记录定义
            t = init_instance_by_config(
                config,
            )           
            # 如果自动导入模式，则从原数据中找到最近日期作为开始日期，以当前日工作日前一天作为结束日期
            if self.auto_import:
                start_date = working_day
                end_date = working_day       
            
            # 动态调用导入数据对应的类方法 
            import_func_names = config["kwargs"]["data_import_func_names"]
            for import_func_name in import_func_names:
                args = (start_date,end_date)
                method = getattr(t,import_func_name)
                method(args) 
            
        
    def get_local_data_info(self,code,period):    
        """取得本地数据里最后一天作为开始日期""" 
        
        sql = "select id,start_date,end_date,data_path from local_data_source where code=%s and period_type=%s"
        rows = self.db_accessor.do_query(sql,(code,period))
        if len(rows)==0:
            return None
        data_info = {"id":rows[0][0],"start_date":rows[0][1],"end_date":rows[0][2],"data_path":rows[0][3]}
        return data_info
 
    def update_last_local_data_date(self,code,period,end_date):    
        """更新本地数据里的结束日期字段""" 
        
        sql = "update local_data_source set end_date=%s where code=%s and period_type=%s"
        self.db_accessor.do_inserto_withparams(sql,(end_date,code,period))
    
               
    