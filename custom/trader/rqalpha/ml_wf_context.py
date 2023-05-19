import pickle
import numpy as np
import pandas as pd

from qlib.utils import init_instance_by_config
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.model.trainer import fill_placeholder
import qlib

import ruamel.yaml as yaml

from .ml_context import MlIntergrate
from darts_pro.data_extension.custom_model import TFTExtModel
from cus_utils.common_compute import normalization,compute_series_slope,compute_price_range,slope_classify_compute,comp_max_and_rate
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,CLASS_SIMPLE_VALUE_MAX
from trader.busi_compute import slope_status
from persistence.wf_task_store import WfTaskStore
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
logger = AppLogger()

class MlWorkflowIntergrate(MlIntergrate):
    """继承MlIntergrate,获取相关上下文数据和环境,用于工作流模式"""
    
    def __init__(
        self,
        config_path,
        **kwargs,
    ):
        with open(config_path) as fp:
            config = yaml.safe_load(fp)    
            
        # 配置文件内容
        self.model_cfg =  config["task"]["model"]
        self.dataset_cfg = config["task"]["dataset"]
        self.backtest_cfg = config["task"]["backtest"]
        
        # 初始化
        self.pred_data_path = self.model_cfg["kwargs"]["pred_data_path"]
        self.load_dataset_file = self.model_cfg["kwargs"]["load_dataset_file"]
        self.save_dataset_file = self.model_cfg["kwargs"]["save_dataset_file"]
        
        # 初始化dataset
        dataset = init_instance_by_config(self.dataset_cfg)
        df_data_path = self.pred_data_path + "/df_all.pkl"
        dataset.build_series_data(df_data_path,no_series_data=True)  
          
        # 生成recorder,用于后续预测数据处理
        record_cfg = config["task"]["record"]
        optargs = self.model_cfg["kwargs"]["optargs"]
        model = TFTExtModel.load_from_checkpoint(optargs["model_name"],work_dir=optargs["work_dir"],best=False)
        placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
        record_cfg = fill_placeholder(record_cfg, placehorder_value)   
        rec = R.get_recorder()
        recorder = init_instance_by_config(
            record_cfg,
            recorder=rec,
        )       
             
        self.dataset = dataset
        self.pred_recorder = recorder
        self.kwargs = kwargs
               
        self.task_id = kwargs["task_id"]
        self.task_store = WfTaskStore()
        self.dbaccess = DbAccessor({})

    def prepare_data(self,pred_date):   
        
        # 根据日期，动态找到对应的预测文件，并加载
        date_pred_df_file = self.task_store.get_pred_result_by_task_and_working_day(self.task_id,pred_date)
        dump_path = self.kwargs["dump_path"]
        total_path = "{}/{}".format(dump_path,date_pred_df_file)
        with open(total_path, "rb") as fin:
            self.pred_df = pickle.load(fin)      
                 
    def filter_buy_candidate(self,pred_date):
        """根据预测计算，筛选可以买入的股票"""
        
        can_list = self.filter_buy_candidate_data(pred_date,self.pred_df) 
        return can_list 
    
    def record_results(self,result_df):
        """保存回测结果"""

        total_gain = result_df["gain"].sum()
        total_transactions = result_df.shape[0]
        avg_duration = result_df["duration"].sum()/total_transactions
        # 保存到数据库
        self.dbaccess.do_inserto("delete from backtest_result where task_id={}".format(self.task_id))
        sql = "insert into backtest_result(task_id,total_gain,total_transactions,avg_duration) values(%s,%s,%s,%s)"
        self.dbaccess.do_inserto_withparams(sql,(self.task_id,total_gain,total_transactions,avg_duration))
        
    
