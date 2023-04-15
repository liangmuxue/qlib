import copy
from datetime import datetime
import pickle

from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config

from .base_processor import BaseProcessor
from trader.utils.date_util import get_tradedays_dur,date_string_transfer
from persistence.common_dict import CommonDictEnum,CommonDict,CommonDictType
from workflow.busi_exception import WorkflowException
from workflow.constants_enum import WorkflowExceptionType

class ClassifyProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
        self.common_dict = CommonDict()
        
    def build_real_template(self,template,config=None,working_day=None):
        """根据原配置模板，生成实际配置文件"""
        
        self.task_ignore = False
        
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
        real_template["data_handler_config"]["start_time"] = start_date
        real_template["data_handler_config"]["end_time"] = end_date
        
        # dataset数据集定义的开始结束时间
        dataset_template["kwargs"]["segments"]["train_total"] = [start_date,end_date]
        # 验证集结束日期前推3个月
        valid_start_date = get_tradedays_dur(date_string_transfer(end_date,direction=2),-30*3)
        # 验证集结束日期前推1个月
        valid_end_date = get_tradedays_dur(date_string_transfer(end_date,direction=2),-30*1)
        dataset_template["kwargs"]["segments"]["valid"] = [valid_start_date,valid_end_date]
        
        # 设置内部数据存储路径
        model_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        record_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        # 映射预测文件
        cur_sequence = self.wf_task.task_entity["sequence"]
        # 根据当前序号，前推2个周期，如果不够则不进行此次任务
        if cur_sequence<=2:
            # 设置忽略标志
            self.task_ignore = True
        else:
            pred_sequence = cur_sequence - 2
            pred_type_id = self.common_dict.get_dict_by_type_and_code(CommonDictType.WORK_TYPE.value,CommonDictEnum.WORK_TYPE__PRED.value)["id"]
            sql = "select pred_result_file from pred_result pr,workflow_task_detail " \
                "wtd,workflow_detail wd where pr.task_id=wtd.id and wtd.workflow_detail_id=wd.id and " \
                "wtd.main_task_id={} and wd.type={} and wtd.sequence={}".format(self.wf_task.main_task.task_obj["id"],pred_type_id,pred_sequence)
            # 关联到预测处理器，并取得对应的文件名
            pred_result_file = self.db_accessor.do_query(sql, ())[0][0]
            model_template["kwargs"]["pred_data_file"] = self.wf_task.get_pred_data_part_path(pred_result_file,working_day)
            record_template["kwargs"]["pred_data_file"] = model_template["kwargs"]["pred_data_file"]
            # 数据集定义预测分类评估的开始时间，为当前指定日期（workding day）的两周前,即上2个日历批次的第一天
            working_day_list = self.wf_task.get_calendar_by_seq(pred_sequence)
            classify_begin_date = str(working_day_list[0])
            classify_begin_date = datetime.strptime(classify_begin_date,"%Y%m%d")
            # 数据集定义预测分类评估的结束时间，为当前指定日期（workding day）的上一个交易日，即上一个日历批次的最后一天
            next_working_day_list = self.wf_task.get_calendar_by_seq(pred_sequence+1)
            classify_end_date = str(next_working_day_list[-1])        
            classify_end_date = datetime.strptime(classify_end_date,"%Y%m%d")
            dataset_template["kwargs"]["segments"]["classify_range"] = [classify_begin_date,classify_end_date]
        return real_template
    
    
    def before_run(self,working_day=None):
        """子任务运行的前处理"""

        # 保存验证结果到数据库,其中开始日期和结束日期是在开始的时候动态计算的（滚动计算，2周倩到1周前）
        dataset_template = self.config["task"]["dataset"]
        classify_range = dataset_template["kwargs"]["segments"]["classify_range"]
        classify_begin = int(classify_range[0].strftime("%Y%m%d"))
        classify_end = int(classify_range[1].strftime("%Y%m%d"))
        # 开始前生成记录
        rows = self.db_accessor.do_query("select * from classify_result where task_id={}".format(self.wf_task.task_entity["id"]))
        if len(rows)>0:
            sql = "update classify_result set start_time=now(),classify_begin_date=%s,classify_end_date=%s where task_id=%s"
            params = (classify_begin,classify_end,self.wf_task.task_entity["id"])
        else:
            sql = "insert into classify_result(task_id,start_time,classify_begin_date,classify_end_date) values(%s,now(),%s,%s)"
            params = (self.wf_task.task_entity["id"],classify_begin,classify_end)
        self.db_accessor.do_inserto_withparams(sql, params)   
                                    
    def sub_run(self,working_day=None,results=None,resume=True):
        """个性化内容"""
        
        df = results[0]
        corr_2_rate = df[df["correct"]==2].shape[0]/df.shape[0]
        corr_1_rate = df[df["correct"]==1].shape[0]/df.shape[0]
        corr_0_rate = df[df["correct"]==0].shape[0]/df.shape[0]  
        total_cnt = df.shape[0]
        # 预测判别中已经生成了相关结果，在此进行结果入库
        sql = "update classify_result set end_time=now(),corr2_rate=%s,corr1_rate=%s,corr0_rate=%s,total_cnt=%s where task_id=%s"
        # 保存验证结果到数据库
        params = (round(corr_2_rate,5),round(corr_1_rate,5),round(corr_0_rate,5),total_cnt,self.wf_task.task_entity["id"])
        self.db_accessor.do_inserto_withparams(sql, params)                    
    
    
    