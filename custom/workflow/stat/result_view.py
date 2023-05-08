import pandas as pd
import numpy as np
import pickle
import yaml
import sys, os
from pathlib import Path
from datetime import datetime
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
            item_df["datetime_dt"] = pd.to_datetime(item_df.datetime)
            test_df = item_df[item_df["datetime_dt"].dt.strftime("%Y%m%d").astype(int)>=20230401]
            # logger.info("test_df datetime:{}".format(test_df.datetime.unique()))   
    
    def view_pred_data(self,pred_data_file):
        with open(pred_data_file, "rb") as fin:
            total_df = pickle.load(fin)           
            logger.info("df pred_date:{}".format(total_df.pred_date.unique()))         
   
    def view_trade_data(self):
        file_path = "/home/qdata/workflow/wf_backtest_flow_2023/trader_data/05/trade_data.csv"
        trade_data = pd.read_csv(file_path,parse_dates=['trade_date'],infer_datetime_format=True)  
        print("trade_data:",trade_data)
        tar_data = trade_data["trade_date"].dt.strftime('%Y%m%d')=="20230505"
        order_id = 16832845180002
        # trade_data[trade_data["order_id"]==order_id] = np.array([datetime.datetime(2023, 5, 5, 9, 31), '600533.XSHG', 1, 3.4, 4600, 15640.0])
        from rqalpha.const import ORDER_STATUS
        print("ORDER_STATUS.ACTIVE:{}".format(ORDER_STATUS.ACTIVE))
        
if __name__ == "__main__":    
    # task = WorkflowTask(task_batch=2,workflow_id=1,resume=True)
    view = ResultView()
    csv_file = "/home/qdata/stock_data/ak/whole_data/day/600008_institution.csv"
    csv_file = "/home/qdata/stock_data/ak/item/day/002461_institution.csv"
    # csv_file = "/home/qdata/stock_data/tdx/item/5m/600678.csv"
    view.view_csv_data(csv_file)
    data_file = "/home/qdata/stock_data/ak/all_day_institution.pickle"
    # data_file = "/home/qdata/stock_data/tdx/all_5m.pickle"
    # view.view_whole_data(data_file)
    # view.view_qlib_data("custom/config/stat/dataset.yaml")    
    pred_data_file = "/home/qdata/workflow/wf_backtest_flow/task/20/dump_data/pred_part/pred_df_total_20220201.pkl"
    pred_data_file = "/home/qdata/workflow/wf_test/task/73/dump_data/pred_part/pred_df_total_20220118.pkl"
    # view.view_pred_data(pred_data_file)   
    # view.view_trade_data()
    
    
    