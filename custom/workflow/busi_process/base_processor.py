import qlib
from qlib.model.trainer import task_train
from qlib.config import C
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.model.trainer import fill_placeholder


import time
import sys, os
from pathlib import Path
import yaml
import multiprocessing as mp
from multiprocessing import Queue

from workflow.constants_enum import WorkflowStatus,WorkflowSubStatus,FrequencyType,WorkflowType
from cus_utils.db_accessor import DbAccessor
from cus_utils.log_util import AppLogger

logger = AppLogger()

class BaseProcessor(object):
    
    def __init__(self, workflow_subtask):
        self.wf_task = workflow_subtask
        self.db_accessor = DbAccessor({})
        self.task_ignore = False
        self.restart_after_fail = False
        
    def build_real_template(self,config=None,working_day=None):
        """生成实际配置文件，子类实现"""
        pass
    
    def run(self,yaml_file,working_day=None,resume=True): 
        """运行主要逻辑"""
        
        self.wf_task.task_start_handler(self) 
        #  如果有忽略标志，则直接略过 
        if self.task_ignore:
            logger.info("ignore for task:{}".format(self.wf_task.task_entity["id"]))
            self.wf_task.task_ignore_handler(self,ignore_status=WorkflowSubStatus.busi_ignore.value)
            return WorkflowSubStatus.busi_ignore.value      
        # 读取配置文件
        with open(yaml_file) as fp:
            config = yaml.safe_load(fp)    
            self.config = config     
        # 多进程异步方式执行
        self.process_worker = self.start_worker(self.run_worker_task,args=(self.wf_task,config,working_day,resume))    
        while True:
            # 一直轮询当前状态，如果发生改变则退出主进程
            status = self.wf_task.get_task_status()
            if status==WorkflowSubStatus.running.value:
                # logger.debug("keep running")
                time.sleep(3)
            else:
                break
        return status

    def _exe_task(self,task_config: dict):
        rec = R.get_recorder()
        # model & dataset initiation
        model = init_instance_by_config(task_config["model"])
        dataset = init_instance_by_config(task_config["dataset"])
        # 执行实际运行入口，具体运行内容取决于yaml配置文件中的任务type字段
        model.fit(dataset)
        # fill placehorder
        placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
        task_config = fill_placeholder(task_config, placehorder_value)
        # generate records: prediction, backtest, and analysis
        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        rtn_list = []
        for record in records:
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_ext",
                try_kwargs={"model": model, "dataset": dataset},
            )
            rtn = r.generate()
            rtn_list.append(rtn)
        return rtn_list

    def before_run(self,working_day=None):
        """子任务运行的前处理，子类实现"""
        pass   
              
    def sub_run(self,working_day=None,results=None,resume=True):
        """子任务运行，子类实现"""
        pass   
    
    def record(self,results):
        """运行结果记录，子类实现"""
        pass   
        
    ########################################################   多进程部分   #########################################################     
       
    def start_worker(self, target,args):
        p = mp.Process(target=target, args=args)
        p.start()
        return p     
    
    def run_worker_task(self,wf_task,config,working_day=None,resume=True): 
        """在单独进程内运行"""

        # 初始化qlib数据
        experiment_name = "workflow"
        qlib_init_config = config["qlib_init"]
        qlib.init(provider_uri=qlib_init_config["provider_uri"], region=qlib_init_config["region"])  
        with R.start(experiment_name=experiment_name, recorder_name=None):  
            try:
                # 根据配置，决定是否全部个性化运行
                if "indiv" in config and config["indiv"] is True:
                    self.sub_run(working_day=working_day,resume=resume)
                else:
                    exp_manager = C["exp_manager"]
                    uri_folder = "mlruns"
                    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)  
                    self.before_run(working_day=working_day)  
                    # 执行任务    
                    results = self._exe_task(config.get("task"))
                    # 执行个性化内容             
                    self.sub_run(working_day=working_day,results=results,resume=resume)
            except Exception as e:
                # 修改为错误状态
                wf_task.task_fail_handler(self)
                logger.exception("sub_run fail:{}".format(e))
            else:
                # 执行回调修改状态为已成功
                wf_task.task_sucess_handler(self)
                                     