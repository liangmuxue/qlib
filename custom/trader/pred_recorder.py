import logging
import warnings
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
from typing import Union, List, Optional
import backtrader as bt
from datetime import datetime
from qlib.utils.exceptions import LoadObjectError
from qlib.contrib.evaluate import risk_analysis, indicator_analysis

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.backtest import backtest as normal_backtest
from qlib.log import get_module_logger
from qlib.utils import flatten_dict, class_casting
from qlib.utils.time import Freq
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec
from qlib.utils import init_instance_by_config

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from .rl_strategy import RLStrategy,ResultStrategy

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from rl.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from .tft_recorder import TftRecorder

from cus_utils.log_util import AppLogger
logger = AppLogger()

class PortAnaRecord(TftRecorder):
    """
    自定义recorder，实现策略应用以及回测，使用深度学习框架模式
    """

    artifact_path = "portfolio_analysis"

    def __init__(
        self,
        recorder,
        config,
        model = None,
        dataset = None,
        **kwargs,
    ):
        """
        config["strategy"] : dict
            define the strategy class as well as the kwargs.
        config["executor"] : dict
            define the executor class as well as the kwargs.
        config["backtest"] : dict
            define the backtest kwargs.
        risk_analysis_freq : str|List[str]
            risk analysis freq of report
        indicator_analysis_freq : str|List[str]
            indicator analysis freq of report
        indicator_analysis_method : str, optional, default by None
            the candidated values include 'mean', 'amount_weighted', 'value_weighted'
        """
        super().__init__(recorder=recorder, **kwargs)

        self.strategy_config = config["strategy"]
        self.backtest_config = config["backtest"]
        self.model = model
        self.dataset = dataset
        
        self.df_ref = dataset.df_all     
               
    def _get_strategy_clazz(self,class_name):
        if class_name=="RLStrategy":
            return RLStrategy     
        if class_name=="ResultStrategy":
            return ResultStrategy              
        
    def _get_report_freq(self, executor_config):
        ret_freq = []
        if executor_config["kwargs"].get("generate_portfolio_metrics", False):
            _count, _freq = Freq.parse(executor_config["kwargs"]["time_per_step"])
            ret_freq.append(f"{_count}{_freq}")
        if "inner_executor" in executor_config["kwargs"]:
            ret_freq.extend(self._get_report_freq(executor_config["kwargs"]["inner_executor"]))
        return ret_freq

    def generate(self, **kwargs):
        """执行预测数据生成的过程"""
        
        train_start_time = self.backtest_config["train_start_time"]
        train_end_time = self.backtest_config["train_end_time"]
        trade_start_time = self.backtest_config["trade_start_time"]
        trade_end_time = self.backtest_config["trade_end_time"]
 
        # 生成预测数据
        self.pred_data = self.build_pred_result(train_start_time,trade_end_time)
    
    def build_pred_result(self,start_time,end_time):
        """逐天生成预测数据"""
        
        df = self.df_ref[(self.df_ref["datetime"]>=pd.to_datetime(str(start_time)))&(self.df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        pred_data = {}
        
        date_range = df["datetime"].dt.strftime('%Y%m%d').unique()
        # 取得日期范围，并遍历生成预测数据
        for item in date_range:
            cur_date = item
            # 利用strategy的方法生成数据
            pred_series_list = self.predict_process(cur_date,outer_df=self.df_ref)
            # 存储数据
            data_path = self._get_pickle_path(cur_date)
            with open(data_path, "wb") as fout:
                pickle.dump(pred_series_list, fout)              
        return pred_data

    def _get_pickle_path(self,cur_date):
        pred_data_path = self.model.pred_data_path
        data_path = pred_data_path + "/" + str(cur_date) + ".pkl"
        return data_path
    
    def predict_process(self,cur_date,outer_df=None):
        """执行预测过程"""
        
        # 根据时间点，取得对应的输入时间序列范围
        total_range,val_range = self.dataset.get_part_time_range(cur_date,ref_df=self.df_ref)
        # 如果不满足预测要求，则返回空
        if total_range is None:
            self.log("pred series none")
            return None
        
        # 从执行器模型中取得已经生成好的模型变量
        my_model = self.model.model
        # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
        series_transformed,val_series_transformed,past_convariates,future_convariates = self.dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=outer_df)
        my_model.fit(series_transformed,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,num_loader_workers=8,verbose=True,epochs=-1)            
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=self.dataset.pred_len, series=val_series_transformed,num_samples=200,
                                            past_covariates=past_convariates,future_covariates=future_convariates)  
        # 归一化反置，恢复到原值
        pred_series_list = self.dataset.reverse_transform_preds(pred_series_list)
        return pred_series_list    