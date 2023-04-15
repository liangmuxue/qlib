import copy
import datetime
import os

from pytorch_lightning.callbacks import Callback

from .base_processor import BaseProcessor
from trader.utils.date_util import get_tradedays_dur
from cus_utils.db_accessor import DbAccessor

class TrainProcessor(BaseProcessor):
    
    def __init__(self, workflow_subtask):
        super().__init__(workflow_subtask)
     
    def build_real_template(self,template,config=None,working_day=None):
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
        
        # 如果当前序号大于1，则直接使用以前的数据文件
        cur_sequence = self.wf_task.task_entity["sequence"]
        # 只在第一次保存全量数据，后续直接加载
        if cur_sequence>1:
            # 设置忽略标志
            model_template["kwargs"]["load_dataset_file"] = True
            model_template["kwargs"]["save_dataset_file"] = False
            dataset_template["kwargs"]["load_dataset_file"] = True
                   
        # dataset数据集定义的开始结束时间
        dataset_template["kwargs"]["segments"]["train_total"] = [start_date,end_date]
        # 验证集结束日期前推3个月
        valid_start_date = get_tradedays_dur(str(working_day),-22*3)
        # 验证集结束日期前推1个月
        valid_end_date = get_tradedays_dur(str(working_day),-22*1)
        dataset_template["kwargs"]["segments"]["valid"] = [valid_start_date,valid_end_date]
        # 设置内部数据存储路径
        model_template["kwargs"]["pred_data_path"] = self.wf_task.get_dumpdata_path()
        # 设置模型路径
        model_template["kwargs"]["optargs"]["work_dir"] = self.wf_task.get_model_path()
        # 设置模型名称
        model_template["kwargs"]["optargs"]["model_name"] = self.wf_task.get_model_name(working_day=working_day)  
        return real_template
                            
    def sub_run(self,working_day=None,results=None,resume=True):
        """子任务运行的后处理"""

        sql = "update train_val_result set end_time=now() where id={}".format(self.main_id)
        self.db_accessor.do_inserto(sql)         

    def before_run(self,working_day=None):
        """子任务运行的前处理"""
        
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 清除之前的多余数据
        # self.db_accessor.do_inserto("delete from train_val_result_detail where main_id=(select id from train_val_result where task_id={})".format(self.wf_task.task_entity["id"]))
        # self.db_accessor.do_inserto("delete from train_val_result where task_id={}".format(self.wf_task.task_entity["id"]))
        sql = "insert into train_val_result(task_id,start_time) values(%s,%s)"
        # 生成一条训练记录
        self.db_accessor.do_inserto_withparams(sql, (self.wf_task.task_entity["id"],dt)) 
        self.main_id = self.db_accessor.do_query("select id from train_val_result where task_id={}".format(self.wf_task.task_entity["id"]))[0][0]
        # 创建存储目录
        dumpdata_path = self.wf_task.get_dumpdata_path()
        if not os.path.exists(dumpdata_path):
            os.makedirs(dumpdata_path)   
            os.makedirs(self.wf_task.get_dumpdata_part_path())        
            
        # 给回调类设置当前训练结果主表的标识
        model_template = self.config["task"]["model"]
        lightning_callbacks = model_template["kwargs"].get("lightning_callbacks", [])
        for callback_config in lightning_callbacks:
            callback_config["kwargs"]["main_id"] = self.main_id        
      
class TrainProcessorCallback(Callback):
    """用于训练回调，数据处理"""

    def __init__(self, main_id,**kwargs):
        self.db_accessor = DbAccessor({})
        self.main_id = main_id
        
    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_validation_epoch_end(self, trainer, pl_module):
        """每个epoch结束时的回调"""
        
        epoch = pl_module.current_epoch
        results = pl_module.val_results[epoch]
        # 把当前批次内的累加数据取平均值，并入库
        val_loss = round(results["val_loss"]/results["time"],5)
        val_corr_loss = round(results["val_corr_loss"]/results["time"],5)
        value_range_loss = round(results["value_range_loss"]/results["time"],5)
        ce_loss = round(results["ce_loss"]/results["time"],5)
        value_diff_loss = round(results["value_diff_loss"]/results["time"],5)
        vr_acc = round(results["vr_acc"]/results["time"],5)
        import_vr_acc = round(results["import_vr_acc"]/results["time"],5)    
        
        sql = "insert into train_val_result_detail(main_id,batch_idx,val_loss,val_corr_loss,value_range_loss,ce_loss,value_diff_loss,vr_acc,import_vr_acc)" \
            "values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        params = (self.main_id,epoch,val_loss,val_corr_loss,value_range_loss,ce_loss,value_diff_loss,vr_acc,import_vr_acc)
        self.db_accessor.do_inserto_withparams(sql, params)         
        # 清除已经统计过的数据
        # del pl_module.val_results[epoch]            
        
        