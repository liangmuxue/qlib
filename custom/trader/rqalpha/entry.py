import argparse
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

from rqalpha import run_file

import qlib
import fire
import pandas as pd
import ruamel.yaml as yaml
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN
from trader.rqalpha.ml_context import MlIntergrate
from trader.data_viewer import DataViewer
    
def start_rqalpha(config_file):
    """"回测入口，整合glib的工作流配置方式，与rqalpha结合"""
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)   
        
    # rqalpha配置属于整个工作流配置文件的一部分
    rq_config = config["task"]["backtest"]["rqalpha"]
    # 统一一个运行策略文件，后面通过不同的实现类来进行策略区分
    strategy_file_path = config["task"]["backtest"]["run_file"]
    run_file(strategy_file_path, rq_config) 

def analysis_result():
    ext_length = 25
    ml_context = MlIntergrate(ext_length=ext_length)
    ml_context.create_bt_env()
    pred_date = 20210525
    instrument = 600030
    date_pred_df = ml_context.pred_df[(ml_context.pred_df["pred_date"]==pred_date)]
    complex_item_df = ml_context.combine_complex_df_data(pred_date,instrument,pred_df=date_pred_df,df_ref=ml_context.dataset.df_all,ext_length=ext_length)  
    data_viewer = DataViewer(env_name="result analysis")
    data_viewer.show_single_complex_pred_data_visdom(complex_item_df,dataset=ml_context.dataset)
           
 
# analysis_result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_file", type=str, default="custom/config/workflow/backtest_day.yaml", help="workflow config file")
    args = parser.parse_args()
    start_rqalpha(args.config_file)
      
    