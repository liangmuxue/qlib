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

from cus_utils.log_util import AppLogger
from trader import pred_recorder
logger = AppLogger()

class MlWorkflowIntergrate(MlIntergrate):
    """继承MlIntergrate,获取相关上下文数据和环境,用于工作流模式"""
    
    def __init__(
        self,
        provider_uri = "/home/qdata/qlib_data/custom_cn_data",
        config_path="custom/config/darts/workflow_backtest.yaml",
        **kwargs,
    ):
        super().__init(provider_uri=provider_uri,config_path=config_path,kwargs=kwargs)
        self.task_id = kwargs["task_id"]
         
    def filter_buy_candidate(self,pred_date):
        """根据预测计算，筛选可以买入的股票"""
        
        # 根据日期，动态找到对应的预测文件，并加载
        date_pred_df = self.pred_df[(self.pred_df["pred_date"]==pred_date)]
        return self.filter_buy_candidate_data(pred_date,date_pred_df)   
    
    
    
