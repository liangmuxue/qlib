import logging
import warnings
import numpy as np
import pandas as pd
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

from .tft_recorder import TftRecorder
from .bt_strategy import Strategy,ResultStrategy,QlibStrategy

logger = get_module_logger("workflow", logging.INFO)

class PortAnaRecord(TftRecorder):
    """
    自定义recorder，实现策略应用以及回测，使用backtrader框架模式
    """

    artifact_path = "portfolio_analysis"

    def __init__(
        self,
        recorder,
        config,
        model = None,
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

    def _get_report_freq(self, executor_config):
        ret_freq = []
        if executor_config["kwargs"].get("generate_portfolio_metrics", False):
            _count, _freq = Freq.parse(executor_config["kwargs"]["time_per_step"])
            ret_freq.append(f"{_count}{_freq}")
        if "inner_executor" in executor_config["kwargs"]:
            ret_freq.extend(self._get_report_freq(executor_config["kwargs"]["inner_executor"]))
        return ret_freq
    
    def _get_strategy_clazz(self,class_name):
        if class_name=="Strategy":
            return Strategy
        if class_name=="ResultStrategy":
            return ResultStrategy        
        if class_name=="QlibStrategy":
            return QlibStrategy   
        
    def generate(self, **kwargs):
        """执行回测进程"""
        
        # 根据配置，取得对应的策略类
        strategy_clazz = self._get_strategy_clazz(self.strategy_config["bt_class"])
        qlib_strategy_clazz = self._get_strategy_clazz(self.strategy_config["class"])
        trade_strategy_obj = init_instance_by_config(self.strategy_config, accept_types=qlib_strategy_clazz)
        cerebro = bt.Cerebro()
        # bt模式下，只能直接放置类,参数需要从已经初始化中的对象里再拿一次
        cerebro.addstrategy(strategy_clazz,model=trade_strategy_obj.model,dataset=trade_strategy_obj.dataset,topk=trade_strategy_obj.topk)
        
        # 从初始化对象中，取得数据及配置，并插入bt的cerebro中
        start_time = self.backtest_config["start_time"]
        end_time = self.backtest_config["end_time"]
        # 这里使用valid数据集,每个股票系列单独添加
        group_column = trade_strategy_obj.dataset.get_group_rank_column()
        df_val = self.model.df_ref[(self.model.df_ref["datetime"]>=pd.to_datetime(str(start_time)))&(self.model.df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        group = df_val.groupby(group_column)
        for group_name, item_df in group:
            dataname = self.transfer_to_bt_format(item_df)
            data = bt.feeds.PandasData(dataname=dataname, name=str(group_name),
                                       fromdate=start_time, todate=end_time,datetime="datetime")
            cerebro.adddata(data)
    
        cerebro.broker.setcash(100000.0)
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        cerebro.run()        
    
    def transfer_to_bt_format(self,ori_df):
        target_df = ori_df
        target_df["open"] = target_df["OPEN"]
        target_df["close"] = target_df["CLOSE"]
        target_df["high"] = target_df["HIGH"]
        target_df["low"] = target_df["LOW"]
        target_df["volume"] = target_df["VOLUME_CLOSE"]
        return target_df
        
    def list(self):
        list_path = []
        for _freq in self.all_freq:
            list_path.extend(
                [
                    f"report_normal_{_freq}.pkl",
                    f"positions_normal_{_freq}.pkl",
                ]
            )
        for _analysis_freq in self.risk_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(f"port_analysis_{_analysis_freq}.pkl")
            else:
                warnings.warn(f"risk_analysis freq {_analysis_freq} is not found")

        for _analysis_freq in self.indicator_analysis_freq:
            if _analysis_freq in self.all_freq:
                list_path.append(f"indicator_analysis_{_analysis_freq}.pkl")
            else:
                warnings.warn(f"indicator_analysis freq {_analysis_freq} is not found")
        return list_path