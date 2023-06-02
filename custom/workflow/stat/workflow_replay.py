"""复盘功能"""
import yaml

from workflow.workflow_task import WorkflowTask
import qlib
from qlib.model.trainer import task_train
from qlib.config import C
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.model.trainer import fill_placeholder

class Replay():
    def __init__(self,config_file):
        with open(config_file) as fp:
            config = yaml.safe_load(fp)    
            self.config = config          
    
    def sim_replay(self):
        """仿真数据复盘"""
        
        config = self.config
        # 初始化qlib数据
        experiment_name = "workflow"
        qlib_init_config = config["qlib_init"]
        qlib.init(provider_uri=qlib_init_config["provider_uri"], region=qlib_init_config["region"])  
        with R.start(experiment_name=experiment_name, recorder_name=None):  
            rq_config = self.config["task"]["backtest"]["rqalpha"]
            # 统一一个运行策略文件，后面通过不同的实现类来进行策略区分
            strategy_file_path = self.config["task"]["backtest"]["run_file"]
            # 改写run file，用于使用自定义strategy,解决execution context问题
            from trader.rqalpha import run_file
            run_file(strategy_file_path, rq_config)           

    def _exe_task(self,task_config: dict):
        rec = R.get_recorder()
        # model & dataset initiation
        model = init_instance_by_config(task_config["model"])
        dataset = init_instance_by_config(task_config["dataset"])
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
        
if __name__ == "__main__":    
    # config_file = "/home/qdata/workflow/wf_backtest_flow_2023/task/118/config/history/simulation_20230531.yaml"
    # replay = Replay(config_file=config_file)
    # replay.sim_replay()
    task = WorkflowTask(task_batch=118,workflow_name="wf_backtest_flow_2023",resume=True)  
    task.start_single_task("replay")
    
    