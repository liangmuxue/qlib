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

class DataProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        self.db_accessor = DbAccessor({})
        # 设置全量导入完成标志
        self.has_finish_complete_import = False
     
    def build_real_template(self,template,working_day=None,config=None):
        """根据原配置模板，生成实际配置文件"""
        
        real_template = copy.deepcopy(template)
        extend_config = self.wf_task.config["extend_config"]
        self.auto_import = False
        # 附加配置中，查看是否有自动导入的选项
        if extend_config is not None:
            cfg = json.loads(extend_config)
            self.auto_import = bool(cfg["auto_import"])
        else:
            self.auto_import = False
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
            is_complete = config["kwargs"]["is_complete"]
            fill_history = config["kwargs"]["fill_history"]
            contain_institution = config["kwargs"]["contain_institution"]
            local_data_info = self.get_local_data_info(LocalDataSourceType.dataframe.value,period) 
            # 存储路径使用数据库中的记录定义
            config["kwargs"]["savepath"] = local_data_info["data_path"]
            no_total_file = config["kwargs"]["no_total_file"]
            t = init_instance_by_config(
                config,
            )           
            # 如果自动导入模式，则从原数据中找到最近日期作为开始日期，以当前日工作日前一天作为结束日期
            if self.auto_import:
                start_date = working_day
                end_date = working_day       
                is_complete = False
                fill_history = True
            else:         
                # 如果已完成全量导入，再次进行任务的时候则恢复为增量
                if self.has_finish_complete_import:
                    is_complete = False
                    fill_history = False
                if is_complete:
                    # 如果是全量导入，则根据配置项，取得导入开始日期，并使用当前工作日期为结束日期，进行导入
                    start_date = config["kwargs"]["start_date"]
                    end_date = working_day
                elif fill_history:
                    # 根据标志,追加以前没有导入的记录
                    start_date = config["kwargs"]["start_date"]  
                    # 如果需要追溯，则取得本地数据里最后一天作为开始日期,使用dataframe类别查询
                    if data_task_batch>0:
                        start_date_indb = local_data_info["end_date"]
                        start_date = start_date_indb
                    end_date = working_day
                else:
                    # 只导入当天记录
                    start_date = working_day  
                    end_date = working_day        
            
            task_batch = t.prepare_import_batch(data_task_batch, start_date, end_date, period)  
            # 根据批次号挂接对应工作流子任务
            self.wf_task.attach_busi_task(task_batch)
            if is_complete:
                # 如果是全量,并且是非恢复模式，需要清除之前的数据
                if not resume:
                    self.clear_local_data(LocalDataSourceType.dataframe.value,period,processor=t)
            results = t.import_data(task_batch,period=period, start_date=start_date,end_date=end_date,auto_import=self.auto_import,
                            contain_institution=contain_institution,resume=resume,no_total_file=no_total_file) 
            if results is None:
                raise Exception("import_data fail")
            # 标志全量导入已完成              
            if is_complete:
                self.has_finish_complete_import = True
            # 执行后，处理任务相关信息   
            self.update_last_local_data_date(LocalDataSourceType.dataframe.value,period,end_date)
            
        # 导出数据任务
        export_tasks = task_config.get("export_task", [])
        if isinstance(export_tasks, dict):  # prevent only one dict
            export_tasks = [export_tasks]
        # 逐个执行任务
        for config in export_tasks:            
            period = get_period_value(config["period"])
            institution = config["kwargs"]["institution"]
            end_date = working_day
            target = config["kwargs"]["target"]
            # 路径使用数据库中的记录定义
            local_data_info = self.get_local_data_info(LocalDataSourceType.qlib.value,period) 
            config["kwargs"]["savepath"] = local_data_info["data_path"]            
            t = init_instance_by_config(
                config,
            )            
            # 首先以股票为单位，导出多个csv
            t.export_whole_item_data(period=period,institution=institution)
            # 然后使用qlib的dump功能从上一步生成的cvs文件中导入
            if target==LocalDataSourceType.qlib.name:
                qlib_dir = config["kwargs"]["qlib_provider_uri"]
                t.export_to_qlib(qlib_dir,period,file_name=config["kwargs"]["dump_file_name"],institution=institution)
                # 修改qlib的数据源导入日期
                self.update_last_local_data_date(LocalDataSourceType.qlib.value,period,end_date)
        
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
    
    def clear_local_data(self,ds_type,period,processor=None):    
        """清除本地数据""" 
        
        if ds_type==LocalDataSourceType.dataframe.value:
            # dataframe模式，清空本地df文件
            data_task_batch = self.wf_task.get_relate_dt_batch()
            processor.clear_local_data(period,data_task_batch)
               
    