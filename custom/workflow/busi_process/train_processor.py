import copy

from .base_processor import BaseProcessor
from trader.utils.date_util import get_tradedays_dur

class TrainProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
     
    def build_real_template(self,template,config=None):
        """根据原配置模板，生成实际配置文件"""
        
        real_template = copy.deepcopy(template)
        model_template = real_template["task"]["model"]
        dataset_template = real_template["task"]["dataset"]
        
        start_date = str(config["start_date"])
        start_date = start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:]
        end_date = str(config["end_date"])
        end_date = end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:]
        # 总数据集定义的开始结束时间
        real_template["data_handler_config"]["start_date"] = start_date
        real_template["data_handler_config"]["end_date"] = end_date
        
        # dataset数据集定义的开始结束时间
        dataset_template["kwargs"]["segments"]["train_total"] = [start_date,end_date]
        # 验证集结束日期前推3个月
        valid_start_date = get_tradedays_dur(str(config["end_date"]),-30*3)
        # 验证集结束日期前推1个月
        valid_end_date = get_tradedays_dur(str(config["end_date"]),-30*1)
        dataset_template["kwargs"]["segments"]["valid"] = [valid_start_date,valid_end_date]
        # 设置内部数据存储路径
        model_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        # 设置模型路径
        model_template["kwargs"]["optargs"]["work_dir"] = self.wf_task.get_model_path()
        # 设置模型名称
        model_template["kwargs"]["optargs"]["model_name"] = self.wf_task.get_model_name()  
        
        return real_template
                            
    def sub_run(self,working_day=None,results=None,resume=True):
        pass
    
    