import pandas as pd
import pickle
import yaml
import sys, os
from pathlib import Path

import qlib
from qlib.model.trainer import task_train
from qlib.config import C
from qlib.workflow import R
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.model.trainer import fill_placeholder

from cus_utils.log_util import AppLogger

logger = AppLogger()

class ResultView(object):
    """工作流任务结果查看"""
    
    def __init__(self):
        pass
    
    def view_qlib_data(self,yaml_file):
        # 读取配置文件
        with open(yaml_file) as fp:
            config = yaml.safe_load(fp)    
        # 初始化qlib数据
        experiment_name = "workflow"
        qlib_init_config = config["qlib_init"]
        qlib.init(provider_uri=qlib_init_config["provider_uri"], region=qlib_init_config["region"])  
        with R.start(experiment_name=experiment_name, recorder_name=None):          
            exp_manager = C["exp_manager"]
            uri_folder = "mlruns"
            exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)    
            task_config = config.get("task")
            dataset = init_instance_by_config(task_config["dataset"])
            dataset.build_series_data()
            logger.info("dataset shape:{}".format(dataset.df_all.shape))
            logger.info("dataset begin date:{}".format(dataset.df_all.datetime.min()))
            logger.info("dataset last date:{}".format(dataset.df_all.datetime.max()))
            
    def view_whole_data(self,data_file):
        with open(data_file, "rb") as fin:
            total_df = pickle.load(fin)           
            logger.info("df total:{}".format(total_df.code.unique()))     
            logger.info("df datetime max:{}".format(total_df.datetime.max())) 
                
    def view_csv_data(self,csv_file):
        with open(csv_file, "rb") as f:
            item_df = pd.read_csv(f)        
            logger.info("df date min:{} and max:{}".format(item_df.datetime.min(),item_df.datetime.max()))     
    
    def view_pred_data(self,pred_data_file):
        with open(pred_data_file, "rb") as fin:
            total_df = pickle.load(fin)           
            logger.info("df pred_date:{}".format(total_df.pred_date.unique()))         
        
if __name__ == "__main__":    
    # task = WorkflowTask(task_batch=2,workflow_id=1,resume=True)
    view = ResultView()
    csv_file = "/home/qdata/stock_data/ak/whole_data/day/600008_institution.csv"
    csv_file = "/home/qdata/stock_data/ak/item/day/600009_institution.csv"
    csv_file = "/home/qdata/stock_data/tdx/item/5m/000539.csv"
    view.view_csv_data(csv_file)
    # data_file = "/home/qdata/stock_data/ak/all_day_institution.pickle"
    # data_file = "/home/qdata/stock_data/tdx/all_5m.pickle"
    # view.view_whole_data(data_file)
    # view.view_qlib_data("custom/config/stat/dataset.yaml")    
    pred_data_file = "/home/qdata/workflow/wf_backtest_flow/task/20/dump_data/pred_part/pred_df_total_20220201.pkl"
    pred_data_file = "/home/qdata/workflow/wf_test/task/73/dump_data/pred_part/pred_df_total_20220118.pkl"
    # view.view_pred_data(pred_data_file)   
    