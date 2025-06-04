import os
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

import torch
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from datetime import datetime

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from .tft_recorder import TftRecorder

from darts_pro.data_extension.series_data_utils import get_pred_center_value
from darts_pro.tft_series_dataset import TFTSeriesDataset
from trader.utils.date_util import get_tradedays,get_tradedays_dur
from cus_utils.log_util import AppLogger
from trader.data_viewer import DataViewer
logger = AppLogger()

class FuturesPredRecorder(TftRecorder):
    """自定义recorder，针对期货行情进行预测，并生成结果"""

    artifact_path = "portfolio_analysis"

    def __init__(
        self,
        recorder,
        config,
        model = None,
        dataset = None,
        pred_data_path = None,
        pred_data_file = None,
        **kwargs,
    ):
        """predict result build"""
        super().__init__(recorder=recorder, **kwargs)
        
        if isinstance(config,dict):
            self.strategy_config = config["strategy"]
            self.backtest_config = config["backtest"]
        self.model = model
        self.dataset = dataset
        self.pred_data_path = pred_data_path
        self.pred_data_file = pred_data_file
        
        # self.df_ref = dataset.df_all     
        self.pred_result_columns = ['pred_date','trend','item_index','name','industry','code'] 
        # self.data_viewer_correct = DataViewer(env_name="stat_pred_classify_correct")
        # self.data_viewer_incorrect = DataViewer(env_name="stat_pred_classify_incorrect")
        
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
        
        start_date = self.backtest_config["pred_start_date"]
        end_date = self.backtest_config["pred_end_date"]
 
        # 生成预测数据
        return self.build_pred_result(start_date,end_date)
    
    def load_pred_data(self,pred_data_file=None):
        if pred_data_file is None:
            pred_data_file = "pred_df_total.pkl"
        with open(pred_data_file, "rb") as fin:
            df_pred = pickle.load(fin)     
        df_pred["trend_class"] = df_pred["trend_class"].astype(int) 
        df_pred["vr_class"] = df_pred["vr_class"].astype(int) 
        return df_pred
                    
    def build_pred_result(self,start_date,end_date):
        """逐天生成预测数据"""
        
        dataset = self.dataset
        trade_dates = np.array(get_tradedays(str(start_date),str(end_date))).astype(np.int)  
        pred_result_list = None
        # 进行预测，取得预测结果
        for pred_date in trade_dates:
            pred_result = self.model.build_pred_result(str(pred_date),dataset=dataset)
            if pred_result_list is None:
                pred_result_list = pred_result[pred_date]
            else:
                pred_result_list = pd.concat([pred_result_list,pred_result[pred_date]])
        if pred_result_list is None or pred_result_list.shape[0]==0:
            return None 
        # 保存数据      
        pred_data_path = self.pred_data_path + "/" + self.pred_data_file
        # 首先取得之前已经有的数据，把此次数据按照日期进行覆盖
        if not os.path.exists(pred_data_path):
            pred_data_df = None
        else:
            with open(pred_data_path, "rb") as fin:
                pred_data_df = pickle.load(fin)     
        pred_data_result = None  
        if pred_data_df is None:
            pred_data_result = pred_result_list
        else:
            pred_data_result_filter = pred_data_df[~pred_data_df['date'].isin(pred_result_list['date'])]
            pred_data_result = pd.concat([pred_data_result_filter,pred_result_list])
        with open(pred_data_path, "wb") as fout:
            pickle.dump(pred_data_result, fout)     
        return pred_data_result
    
    def build_pred_data(self,pred_combine_data,df_ref=None):
        """预测数据生成,pandas格式"""
        
        pred_dates = np.array(list(pred_combine_data.keys())).astype(np.int)
        pred_combine_result = None
        for date in pred_dates:
            pred_res = pred_combine_data[date]
            pred_data = pred_res[0]
            pred_trend = pred_res[1]
            if pred_data.shape[0]==0:
                continue
            # 拼接日期和趋势字段，整合为数据表格式
            pred_trend_arr = np.array([[pred_trend] for _ in range(pred_data.shape[0])])
            pred_trend_date = np.array([[date] for _ in range(pred_data.shape[0])])
            pred_combine_item = np.concatenate([pred_trend_date,pred_trend_arr,pred_data],axis=1)
            if pred_combine_result is None:
                pred_combine_result = pred_combine_item
            else:
                pred_combine_result = np.concatenate((pred_combine_result,pred_combine_item),axis=0)
        pred_combine_result = pd.DataFrame(pred_combine_result,columns=self.pred_result_columns)
        return pred_combine_result
    
    def _get_pickle_path(self,cur_date):
        pred_data_path = self.pred_data_path
        data_path = pred_data_path + "/" + str(cur_date) + ".pkl"
        return data_path
    
    def combine_complex_df_data(self,pred_date,instrument,pred_df=None,df_ref=None,ext_length=25,type="combine"):
        """合并预测数据,实际行情数据,价格数据等
           Params:
                pred_df 预测数据集
                df_ref 全量数据集
                pred_date 预测日期
                instrument 股票代码
                ext_length 显示扩展长度，指定显示前面多少天的相关指标数据
        """
        
        group_column = self.dataset.get_group_column()
        time_index_column = self.dataset.get_time_column()
        # 目标预测数据
        df_pred_item = pred_df[(pred_df["instrument"]==instrument)&(pred_df["pred_date"]==int(pred_date))]
        time_index = df_pred_item[time_index_column].values[0]   
        # 时间序号,向前延展到指定长度，向后延展到预测长度
        time_index_range = [time_index-ext_length,time_index+self.dataset.pred_len]
        # 目标范围数据
        df_item = df_ref[(df_ref[group_column]==instrument)&
                            (df_ref[time_index_column]>=time_index_range[0])&
                            (df_ref[time_index_column]<time_index_range[1])]
        # 如果全量数据里不包括当前股票，则返回空
        if  df_item.shape[0]==0:
            return None   
        # 新增补充的列值
        new_columns = df_item.columns.tolist() + ["pred_date","pred_data","trend_class","vr_class"]
        df_item = df_item.reindex(columns=new_columns)
        df_item["pred_date"] = [pred_date for i in range(df_item.shape[0])] 
        # 预测数据,前面补0
        pred_data = df_pred_item["pred_data"].values
        pad_len = df_item.shape[0]-pred_data.shape[0]
        data_line = np.pad(pred_data,(pad_len,0),'constant',constant_values=(0,0))          
        df_item["pred_data"] = data_line
        # 走势分类信息处
        trend_class = df_pred_item["trend_class"].values[0]
        # 为了方便，在每一行中都放入同样的分类信息
        df_item["trend_class"] = [trend_class for i in range(df_item.shape[0])]
        # 涨跌幅分类信息
        vr_class = df_pred_item["vr_class"].values[0]
        df_item["vr_class"] = [vr_class for i in range(df_item.shape[0])]

        return df_item

