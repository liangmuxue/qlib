import qlib
from qlib.model.trainer import task_train
from qlib.config import C
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.model.trainer import fill_placeholder

import sys, os
from pathlib import Path
import yaml

from workflow.constants_enum import WorkflowStatus,WorkflowSubStatus,FrequencyType,WorkflowType
from cus_utils.db_accessor import DbAccessor
from cus_utils.log_util import AppLogger

logger = AppLogger()

class BaseProcessor(object):
    
    def __init__(self, workflow_subtask):
        self.wf_task = workflow_subtask
        self.db_accessor = DbAccessor({})
        self.task_ignore = False
        self.model = None
        self.dataset = None
        
    def build_real_template(self,config=None,working_day=None):
        """生成实际配置文件，子类实现"""
        pass
    
    def run(self,yaml_file,working_day=None,resume=True): 
        """运行"""

        # 读取配置文件
        with open(yaml_file) as fp:
            config = yaml.safe_load(fp)    
            self.config = config
        # 初始化qlib数据
        experiment_name = "workflow"
        qlib_init_config = config["qlib_init"]
        qlib.init(provider_uri=qlib_init_config["provider_uri"], region=qlib_init_config["region"])  
        with R.start(experiment_name=experiment_name, recorder_name=None):  
            self.wf_task.task_start_handler(self)     
            #  如果有忽略标志，则直接略过 
            if self.task_ignore:
                logger.info("ignore for task:{}".format(self.wf_task.task_entity["id"]))
                self.wf_task.task_ignore_handler(self,ignore_status=WorkflowSubStatus.busi_ignore.value)
                return WorkflowSubStatus.busi_ignore.value  
            try:
                # 根据配置，决定是否全部个性化运行
                if "indiv" in config and config["indiv"] is True:
                    self.sub_run(working_day=working_day)
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
                logger.exception("sub_run fail:{}".format(e))
                self.wf_task.task_fail_handler(self)
                return WorkflowSubStatus.fail.value
        # 执行回调修改状态
        self.wf_task.task_sucess_handler(self)
        return WorkflowSubStatus.success.value        

    def _exe_task(self,task_config: dict):
        rec = R.get_recorder()
        # model & dataset initiation
        if self.model is None:
            self.model = init_instance_by_config(task_config["model"])
        self.dataset = init_instance_by_config(task_config["dataset"])
        # FIXME: resume reweighter after merging data selection
        # reweighter: Reweighter = task_config.get("reweighter", None)
        # model training
        # auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
        self.model.fit(self.dataset)
        R.save_objects(**{"params.pkl": self.model})
        # this dataset is saved for online inference. So the concrete data should not be dumped
        self.dataset.config(dump_all=False, recursive=True)
        R.save_objects(**{"dataset": self.dataset})
        # fill placehorder
        placehorder_value = {"<MODEL>": self.model, "<DATASET>": self.dataset}
        task_config = fill_placeholder(task_config, placehorder_value)
        # generate records: prediction, backtest, and analysis
        records = task_config.get("record", [])
        if isinstance(records, dict):  # prevent only one dict
            records = [records]
        rtn_list = []
        for record in records:
            # Some recorder require the parameter `model` and `dataset`.
            # try to automatically pass in them to the initialization function
            # to make defining the tasking easier
            r = init_instance_by_config(
                record,
                recorder=rec,
                default_module="qlib.workflow.record_ext",
                try_kwargs={"model": self.model, "dataset": self.dataset},
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
        
      