import copy
import pandas as pd
from datetime import datetime
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config

import json

from .base_processor import BaseProcessor
from cus_utils.db_accessor import DbAccessor
from data_extract.his_data_extractor import get_period_value
from trader.utils.date_util import get_previous_day

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
            # 如果自动导入模式，则以当前日工作日前一天作为开始结束日期
            if self.auto_import:
                prev_day = datetime.strptime(str(working_day), '%Y%m%d').date()
                start_date = int(get_previous_day(prev_day).strftime('%Y%m%d'))
                end_date = int(get_previous_day(prev_day).strftime('%Y%m%d'))
            
            # 动态调用导入数据对应的类方法 
            import_func_names = config["kwargs"]["data_import_func_names"]
            for import_func_name in import_func_names:
                args = (start_date,end_date)
                method = getattr(t,import_func_name)
                method(args) 
            
        
    
               
    