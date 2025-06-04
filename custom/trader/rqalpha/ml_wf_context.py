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

from darts_pro.data_extension.custom_nor_model import TFTCluSerModel
from .ml_context import MlIntergrate
from workflow.main_fitter import MainFitter
from cus_utils.common_compute import normalization,compute_series_slope,compute_price_range,slope_classify_compute,comp_max_and_rate
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,CLASS_SIMPLE_VALUE_MAX
from trader.busi_compute import slope_status
from trader.data_viewer import DataViewer
from persistence.wf_task_store import WfTaskStore
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
logger = AppLogger()

class MlWorkflowIntergrate(MlIntergrate):
    """继承MlIntergrate,获取相关上下文数据和环境,用于工作流模式"""
    
    def __init__(
        self,
        task_config,
        **kwargs,
    ):
            
        # 配置文件内容
        self.model_cfg =  task_config["task"]["model"]
        self.dataset_cfg = task_config["task"]["dataset"]
        self.backtest_cfg = task_config["task"]["backtest"]
        # 调试的股票列表
        if "verbose_list" in self.backtest_cfg:
            self.verbose_list = self.backtest_cfg["verbose_list"]
        else:
            self.verbose_list = None
        # 初始化
        self.pred_data_path = self.model_cfg["kwargs"]["pred_data_path"]
        self.pred_data_file = self.model_cfg["kwargs"]["pred_data_file"]
        self.load_dataset_file = self.model_cfg["kwargs"]["load_dataset_file"]
        self.save_dataset_file = self.model_cfg["kwargs"]["save_dataset_file"]
        
        self.kwargs = kwargs
        self.pred_df = None
        self.task_id = kwargs["task_id"]
        self.task_store = WfTaskStore()
        self.dbaccess = DbAccessor({})

    def prepare_data(self,pred_date):   
        """数据准备"""
        
        # 根据日期，累加当前上下文的预测数据
        # pred_data_file = self.model_cfg["kwargs"]["pred_data_file"]
        date_pred_df_file = "{}/{}".format(self.pred_data_path,self.pred_data_file)
        with open(date_pred_df_file, "rb") as fin:
            pred_df = pickle.load(fin)     
            self.pred_df = pred_df[pred_df['date']==pred_date]
        
    def filter_buy_candidate(self,pred_date):
        """根据预测计算，筛选可以买入的品种"""
        
        inst_list = self.pred_df.values[:,-1]  
        return inst_list

    def filter_futures_buy_candidate(self,pred_date):
        """根据预测计算，筛选可以买入的品种"""
        
        inst_list = self.pred_df[["top_flag","instrument"]]  
        return inst_list.values
                       
    def filter_buy_candidate_old(self,pred_date):
        """根据预测计算，筛选可以买入的股票"""
        
        date_pred_df = self.pred_df[(self.pred_df["pred_date"]==pred_date)]
        can_list = []
        can_list = self.filter_buy_candidate_data(pred_date,self.pred_df)            
        # 如果设置了调试内容，则在此处理
        if self.verbose_list is not None:
            data_viewer_correct = DataViewer(env_name="replay_pred_classify_correct")
            ext_length = 25
            for code in self.verbose_list:
                instrument = int(code)
                complex_df = self.combine_complex_df_data(pred_date,instrument,pred_df=date_pred_df,df_ref=self.dataset.df_all,ext_length=ext_length)   
                if complex_df is None:
                    continue
                # 根据预测数据综合判断，取得匹配标志
                (match_flag,pred_class_real,vr_class) = self.pred_data_jud(complex_df, dataset=self.dataset,ext_length=ext_length)     
                data_viewer_correct.show_single_complex_pred_data_visdom(complex_df,correct=0,dataset=self.dataset)       
                data_viewer_correct.show_single_complex_pred_data(complex_df,dataset=self.dataset,save_path=self.pred_data_path+"/plot")
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
        
    
