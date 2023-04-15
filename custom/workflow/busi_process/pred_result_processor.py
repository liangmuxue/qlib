import copy
from datetime import datetime
import pickle
import os

from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config

from .base_processor import BaseProcessor
from trader.utils.date_util import get_tradedays_dur,date_string_transfer
from persistence.common_dict import CommonDictEnum,CommonDict,CommonDictType

class PredResultProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        
    def build_real_template(self,template,config=None,working_day=None):
        """根据原配置模板，生成实际配置文件"""
        
        real_template = copy.deepcopy(template)
        model_template = real_template["task"]["model"]
        dataset_template = real_template["task"]["dataset"]
        record_template = real_template["task"]["record"][0]
        
        start_date = str(config["start_date"])
        start_date = date_string_transfer(start_date)
        # 如果没有设置结束时间，则取当前日期
        if "end_date" in config:
            end_date = str(config["end_date"])
        else:
            end_date = str(working_day)
        end_date = date_string_transfer(end_date) 
        # 总数据集定义的开始结束时间
        real_template["data_handler_config"]["start_date"] = start_date
        real_template["data_handler_config"]["end_date"] = end_date
        
        # dataset数据集定义的开始结束时间
        dataset_template["kwargs"]["segments"]["train_total"] = [start_date,end_date]
        # 验证集结束日期前推3个月
        valid_start_date = get_tradedays_dur(date_string_transfer(end_date,direction=2),-30*3)
        # 验证集结束日期前推1个月
        valid_end_date = get_tradedays_dur(date_string_transfer(end_date,direction=2),-30*1)
        dataset_template["kwargs"]["segments"]["valid"] = [valid_start_date,valid_end_date]
        # 预测开始和结束日期，为当前指定日期（workding day）的下一周(间隔5天)
        pred_len = model_template["kwargs"]["optargs"]["forecast_horizon"]
        working_day_list = self.wf_task.get_calendar_by_seq(self.wf_task.task_entity["sequence"])
        pred_begin_date = str(working_day_list[0])
        pred_begin_date = datetime.strptime(pred_begin_date,"%Y%m%d")       
        pred_end_date = str(working_day_list[-1])
        pred_end_date = datetime.strptime(pred_end_date,"%Y%m%d")
        real_template["port_analysis_config"]["backtest"]["pred_start_time"] = pred_begin_date
        real_template["port_analysis_config"]["backtest"]["pred_end_time"] = pred_end_date
        # 设置内部数据存储路径
        model_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        record_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        # 设置模型路径
        model_template["kwargs"]["optargs"]["work_dir"] = self.wf_task.get_model_path()
        # 设置模型名称
        model_template["kwargs"]["optargs"]["model_name"] = self.wf_task.get_matched_model_file_name(working_day,
                                                                    task_type=CommonDictEnum.WORK_TYPE__TRAIN.value)  
        return real_template

    def before_run(self,working_day=None):
        """子任务运行的前处理"""
        
        # 开始前生成数据库记录
        params = (self.wf_task.task_entity["id"])
        # 先删除，保证唯一性
        self.db_accessor.do_inserto_withparams("delete from pred_result where task_id=%s", params)
        sql = "insert into pred_result(task_id,start_time) values(%s,now())"
        self.db_accessor.do_inserto_withparams(sql, params)   
        # 创建存储目录
        dumpdata_path = self.wf_task.get_dumpdata_path()
        if not os.path.exists(dumpdata_path):
            os.makedirs(dumpdata_path)   
            os.makedirs(self.wf_task.get_dumpdata_part_path())
                                    
    def sub_run(self,working_day=None,results=None,resume=True):
        """个性化内容"""
        
        df_result = results[0]
        model_template = self.config["task"]["model"]
        pred_data_file = model_template["kwargs"]["pred_data_file"]
        file_path = self.wf_task.get_pred_data_part_filepath(pred_data_file,working_day)
        file_name = os.path.split(file_path)[1]
        # 保留每次的预测记录
        with open(file_path, "wb") as fout:
            pickle.dump(df_result, fout)       
        
        # 保存文件名到数据库      
        sql = "update pred_result set end_time=now(),pred_result_file=%s where task_id=%s"
        params = (file_name,self.wf_task.task_entity["id"])
        self.db_accessor.do_inserto_withparams(sql, params)        
    
    