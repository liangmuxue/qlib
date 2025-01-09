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
from cus_utils.common_compute import normalization,compute_series_slope,slope_classify_compute,compute_price_class,comp_max_and_rate
from tft.class_define import SLOPE_SHAPE_SMOOTH,CLASS_SIMPLE_VALUE_SEC,CLASS_SIMPLE_VALUE_MAX
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
        self.pred_result_columns = ['pred_date','time_idx','instrument','trend_class','vr_class','pred_data'] 
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
        
        start_time = self.backtest_config["pred_start_time"]
        end_time = self.backtest_config["pred_end_time"]
 
        # 生成预测数据
        self.pred_data = self.build_pred_result(start_time,end_time)
        return self.pred_data
    
    def load_pred_data(self,pred_data_file=None):
        if pred_data_file is None:
            pred_data_file = "pred_df_total.pkl"
        with open(pred_data_file, "rb") as fin:
            df_pred = pickle.load(fin)     
        df_pred["trend_class"] = df_pred["trend_class"].astype(int) 
        df_pred["vr_class"] = df_pred["vr_class"].astype(int) 
        return df_pred
                    
    def build_pred_result(self,start_time,end_time):
        """逐天生成预测数据"""
        
        # 取得日期范围，并遍历生成预测数据
        date_range = get_tradedays(start_time,end_time)
        date_range = [start_time.strftime('%Y%m%d')]
        data_total = {}
        dataset = self.dataset
        for cur_date in date_range:
            # 进行预测，取得预测结果
            logger.debug("begin predict_process")  
            pred_combine_data = self.predict_process(cur_date=cur_date)
            logger.debug("begin build_pred_data")  
            # # 生成加工后的预测数据
            # data_pred = self.build_pred_data(cur_date,pred_combine_data,df_ref=dataset.df_all)
            data_total[cur_date] = pred_combine_data
            logger.debug("build df_pred,data:{}".format(cur_date))   
             
        # pred_data_path = self.pred_data_path + "/" + self.pred_data_file
        # with open(pred_data_path, "wb") as fout:
        #     pickle.dump(data_total, fout)     
        return data_total
    
    def build_pred_data(self,pred_date,pred_combine_data,df_ref=None):
        """预测数据生成,numpy格式"""
        
        group_column = self.dataset.get_group_rank_column()        
        time_index_column = self.dataset.get_time_column()
        (pred_series_list,pred_class_total,vr_class_total) = pred_combine_data
        data_pred = None
        for index,series in enumerate(pred_series_list):
            group_rank = series.static_covariates[group_column].values[0]
            group_item = self.dataset.get_group_code_by_rank(group_rank)  
            time_index = series.time_index.values
            # 根据概率数据取得唯一性数据
            pred_center_data = get_pred_center_value(series).data
            # 取得分类值
            # pred_class = pred_class_total[index]
            # pred_class_max = self.dataset.combine_pred_class(pred_class)   
            # pred_class_real = pred_class_max[1].item()
            vr_class_data = vr_class_total[index]
            vr_class,vr_class_confidence = comp_max_and_rate(np.array(vr_class_data))
            data_item = np.array([[int(pred_date) for i in range(time_index.shape[0])],
                                  time_index.tolist(),
                                 [group_item for i in range(time_index.shape[0])],
                                 [1 for i in range(time_index.shape[0])],
                                 [vr_class for i in range(time_index.shape[0])]])
            data_item = np.concatenate((data_item,np.expand_dims(pred_center_data,0)),axis=0).transpose(1,0)
            # 图像验证
            # df_item = pd.DataFrame(data_item,columns=self.pred_result_columns)
            # df_item["pred_date"] = df_item["pred_date"].astype("int")
            # complex_df = self.combine_complex_df_data(pred_date,group_item,pred_df=df_item,df_ref=df_ref)      
            # self.data_viewer.show_single_complex_pred_data(complex_df,dataset=self.dataset,save_path=self.pred_data_path+"/plot")
            # self.data_viewer.show_single_complex_pred_data_visdom(complex_df,dataset=self.dataset)                    
            if data_pred is None:
                data_pred = data_item
            else:
                data_pred = np.concatenate((data_pred,data_item),axis=0)
        return data_pred
    
    def _get_pickle_path(self,cur_date):
        pred_data_path = self.pred_data_path
        data_path = pred_data_path + "/" + str(cur_date) + ".pkl"
        return data_path
    
    def predict_process(self,cur_date):
        """根据日期进行预测，得到预测结果 """
        
        # 开始结束日期都是给定日期当天
        cur_date = datetime.strptime(cur_date, '%Y%m%d').date()
        pred_range=[cur_date,cur_date]
        # 调用模型进行预测
        pred_result = self.model.predict(pred_range=pred_range)
        # 返回股票编码
        instrument_rank_arr = [ts["instrument"] for ts in pred_result]
        return instrument_rank_arr

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

