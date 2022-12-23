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
        
        if self.model.load_dataset_file:
            # 使用之前保存的数据作为当前全集参考数据
            self.df_ref =  self.model.df_ref
        else:
            # 使用上一步骤中dataset对象保存的数据集，作为当前全集参考数据
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
        """执行回测进程"""
        
        # 根据配置，取得对应的策略类
        qlib_strategy_clazz = self._get_strategy_clazz(self.strategy_config["class"])
        trade_strategy_obj = init_instance_by_config(self.strategy_config, accept_types=qlib_strategy_clazz)
        
        work_dir = self.backtest_config["work_dir"]
        build_data_flag = self.backtest_config["build_data"]
        train_start_time = self.backtest_config["train_start_time"]
        train_end_time = self.backtest_config["train_end_time"]
        trade_start_time = self.backtest_config["trade_start_time"]
        trade_end_time = self.backtest_config["trade_end_time"]
        # 从回测数据集中再次切分为训练数据集和交易数据集
        group_column = trade_strategy_obj.dataset.get_group_rank_column()
        logger.info("begin recorder")
        df_train = self.df_ref[(self.df_ref["datetime"]>=pd.to_datetime(str(train_start_time)))&(self.df_ref["datetime"]<pd.to_datetime(str(train_end_time)))]
        df_trade = self.df_ref[(self.df_ref["datetime"]>=pd.to_datetime(str(trade_start_time)))&(self.df_ref["datetime"]<pd.to_datetime(str(trade_end_time)))]
        # 根据标志决定是否生成预测数据
        if build_data_flag:
            self.pred_data = self.build_pred_result(trade_strategy_obj,train_start_time,trade_end_time)
        # else:
            # self.pred_data = self.load_pred_result(trade_strategy_obj,train_start_time,trade_end_time)
        logger.info("begin transfer_to_finrl_format")            
        # 转化为FinRL的数据格式
        df_train = self.transfer_to_finrl_format(df_train,type="train")
        df_trade = self.transfer_to_finrl_format(df_trade,type="trade")            
        # 股票数量空间
        stock_dimension = len(df_train[self.dataset.get_group_column()].unique())
        # 总状态空间数量
        state_space = 1 + 2*stock_dimension + self.dataset.pred_len*stock_dimension      
        buy_cost_list = sell_cost_list = [0.001] * stock_dimension
        num_stock_shares = [0] * stock_dimension
        # 每个预测数值作为一个字段
        INDICATORS = self.dataset.ind_column_names(self.dataset.pred_len)
        env_kwargs = {
            "hmax": 100,
            "initial_amount": 1000000,
            "num_stock_shares": num_stock_shares,
            "buy_cost_pct": buy_cost_list,
            "sell_cost_pct": sell_cost_list,
            "state_space": state_space,
            "stock_dim": stock_dimension,
            "tech_indicator_list": INDICATORS,
            "action_space": stock_dimension,
            "reward_scaling": 1e-4
        }
        # 定义执行环境
        e_train_gym = StockTradingEnv(df = df_train, **env_kwargs)
        e_trade_gym = StockTradingEnv(df = df_trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        agent = DRLAgent(env = env_train)
        model_ddpg = agent.get_model("ddpg")
        
        tmp_path = work_dir + '/ddpg'
        new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model_ddpg.set_logger(new_logger_ddpg)
        trained_ddpg = agent.train_model(model=model_ddpg, 
                                     tb_log_name='ddpg',
                                     total_timesteps=50000)     
        
        # df_account_value, df_actions = DRLAgent.DRL_prediction(
        #     model=trained_ddpg, 
        #     environment = e_trade_gym)     
        
          
        # data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
        # insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])        
            
    def transfer_to_finrl_format(self,ori_df,type="train"):
        """转化为FinRL的数据格式"""
        
        target_df = pd.DataFrame()
        target_df["open"] = ori_df["OPEN"]
        target_df["close"] = ori_df["CLOSE"]
        target_df["high"] = ori_df["HIGH"]
        target_df["low"] = ori_df["LOW"]
        target_df["volume"] = ori_df["VOLUME_CLOSE"]
        target_df["time_idx"] = ori_df["time_idx"]
        target_df["time_idx"] = target_df["time_idx"] - target_df["time_idx"].min()
        target_df["datetime"] = ori_df["datetime"]
        target_df["day"] = ori_df["dayofweek"]
        target_df[self.dataset.get_group_column()] = ori_df[self.dataset.get_group_column()]
        
        date_range = ori_df["datetime"].dt.strftime('%Y%m%d').unique()
        # 把连续预测结果数据，生成为dataframe格式
        combine_df = self.dataset.build_df_data_for_pred_list(date_range, 
                                self.dataset.pred_len,pred_data_path=self.model.pred_data_path,type=type,load_cache=True)
        target_df = pd.merge(combine_df, target_df, on=["datetime",self.dataset.get_group_column()])
        target_df["tic"] = target_df[self.dataset.get_group_column()] 
        target_df["date"] = target_df["datetime"]
        # 鉴于一些股票在某些交易日期不开盘，因此需要填充空值
        target_df = self.dataset.fill_miss_data(target_df)     
        # 使用日期编号，重新设置索引  
        target_df["datetime_seq"] = target_df["datetime"].rank(method='dense',ascending=True).astype("int") - 1      
        target_df.set_index('datetime_seq',drop=True,inplace=True)
        return target_df
        
    def build_pred_result(self,trade_strategy_obj,start_time,end_time):
        """逐天生成预测数据"""
        
        df = self.df_ref[(self.df_ref["datetime"]>=pd.to_datetime(str(start_time)))&(self.df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        pred_data = {}
        
        date_range = df["datetime"].dt.strftime('%Y%m%d').unique()
        # 取得日期范围，并遍历生成预测数据
        for item in date_range:
            cur_date = item
            # 利用strategy的方法生成数据
            pred_series_list = trade_strategy_obj.predict_process(cur_date,outer_df=self.df_ref)
            pred_series_list = self.dataset.reverse_transform_preds(pred_series_list)
            # 存储到本地变量，后续使用
            # pred_data[cur_date] = pred_series_list
            data_path = trade_strategy_obj._get_pickle_path(cur_date)
            with open(data_path, "wb") as fout:
                pickle.dump(pred_series_list, fout)              
        return pred_data
       
    def load_pred_result(self,trade_strategy_obj,start_time,end_time):
        """加载之前生成的预测数据"""
        
        df = self.df_ref[(self.df_ref["datetime"]>=pd.to_datetime(str(start_time)))&(self.df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        pred_data = {}
        
        date_range = df["datetime"].dt.strftime('%Y%m%d').unique()
        # 取得日期范围，并遍历生成预测数据
        for item in date_range:
            cur_date = item
            pred_data_path = trade_strategy_obj._get_pickle_path(cur_date)
            with open(pred_data_path, "rb") as fin:
                pred_series_list = pickle.load(fin)
                # 存储到本地变量，后续使用
                # pred_data[cur_date] = pred_series_list                

        return pred_data
    
    