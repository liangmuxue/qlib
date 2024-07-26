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
import datetime


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

class PortAnaRecord(TftRecorder):
    """自定义recorder，主要用于生成一系列的预测数据"""

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
        
        dataset = self.dataset
        expand_length = 10
        
        # 根据时间点，取得对应的输入时间序列范围
        total_range = dataset.segments["train_total"]
        valid_range = dataset.segments["valid"]    
        # 需要延长集合时间
        last_day = get_tradedays_dur(total_range[1],expand_length)
        # 以当天为数据时间终点
        total_range[1] = cur_date
        valid_range[1] = cur_date    
        # 生成未扩展的真实数据
        dataset.build_series_data_step_range(total_range,valid_range)  
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 重置数据区间以满足生成第一阶段数据的足够长度
        total_range[1] = last_day
        valid_range[1] = last_day            
        # 生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_step_range(total_range,valid_range,outer_df=df_expands)            
        # 每次使用前置模型生成一阶段数据          
        self.model.model_exp.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=1,verbose=True,num_loader_workers=6)               
        
        
        # 根据时间点，取得对应的输入时间序列范围
        total_range,val_range,missing_instruments = dataset.get_part_time_range(cur_date,ref_df=dataset.df_all)
        # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
        _,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=df_expands)            
        # 进行第二阶段预测           
        pred_result = self.model.model.predict(n=dataset.pred_len, series=val_series_transformed,num_samples=10,cur_date=cur_date,
                                            past_covariates=past_convariates,future_covariates=future_convariates)
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
        
class ClassifyRecord(PortAnaRecord):
    """用于预测数据的再次分析和筛选"""
    
    def __init__(
        self,
        recorder,
        config,
        model = None,
        dataset = None,
        entity_mode=False,
        **kwargs,
    ):
        """classify result analysis"""
        super().__init__(recorder=recorder, config=config,model=model,dataset=dataset,**kwargs)
        self.classify_range = self.dataset.kwargs["segments"]["classify_range"] 
        self.entity_mode = entity_mode
        
    def generate(self, **kwargs):
        if self.entity_mode:
            return self
        
        df = self.stat_complex_pred_data(self.dataset)
        return df

    def classify_analysis(self, dataset: TFTSeriesDataset):
        """对预测数据进行分类训练"""
        ext_length = 25
        self.stat_complex_pred_data(dataset=dataset,ext_length=ext_length,load_cache=False) 
        
    def stat_complex_pred_data(self,dataset=None,ext_length=25,load_cache=False):
        """统计预测信息准确度，可用性"""
        
        pred_data_path = self.model.kwargs["pred_data_path"]
        pred_file = self.model.kwargs["pred_data_file"]
        if len(pred_file)==0:
            pred_file = None
        else:
            pred_file = "{}/{}".format(pred_data_path,pred_file)
        df,pred_df_total = self.filter_cancidate_result(dataset, pred_file=pred_file,ext_length=ext_length)        
        corr_2_rate = df[df["correct"]==CLASS_SIMPLE_VALUE_MAX].shape[0]/df.shape[0]
        corr_1_rate = df[df["correct"]==CLASS_SIMPLE_VALUE_SEC].shape[0]/df.shape[0]
        corr_danger_rate = df[df["correct"]==0].shape[0]/df.shape[0]
        print("corr_rate:{},corr_sec:{},corr_danger:{}".format(corr_2_rate,corr_1_rate,corr_danger_rate))   
        self.show_correct_pred(df,pred_df_total,dataset=dataset) 
        return df
    
    def filter_cancidate_result(self,dataset=None,ext_length=25,pred_file=None):
        """筛选出合适的品种"""

        date_list = get_tradedays(self.classify_range[0],self.classify_range[1])
        match_list = []
        match_columns = ["date","instrument","correct","vr_class"]
        data_viewer_correct = DataViewer(env_name="stat_pred_classify_correct")
        for pred_date in date_list:   
            # 每日动态加载预测结果数据
            if pred_file is None:
                pred_data_file = self.get_pred_data_file(pred_date)
            else:
                pred_data_file = pred_file
            pred_df_total = self.load_pred_data(pred_data_file=pred_data_file)
            pred_date = int(pred_date)  
            # 筛选出分类上涨类股票
            date_pred_df = pred_df_total[(pred_df_total["vr_class"]==CLASS_SIMPLE_VALUE_MAX)&(pred_df_total["pred_date"]==pred_date)]            
            # 整体需要一定涨幅
            date_pred_df = date_pred_df.groupby('instrument').filter(
                lambda x: ((x["pred_data"].values[-1] - x["pred_data"].values[0])/x["pred_data"].values[0]>(1/100)  
                           # and (x["pred_data"].values[-1] - x["pred_data"].values[0])/x["pred_data"].values[0]<(5/100)
                        )
            )    
            # 最后一段需要上涨
            date_pred_df = date_pred_df.groupby('instrument').filter(
                lambda x: (x["pred_data"].values[-1] - x["pred_data"].values[-2])>0
            )          
            match_cnt = 0                     
            for instrument,group_data in date_pred_df.groupby("instrument"):
                # 生成对应日期的单个股票的综合数据
                complex_df = self.combine_complex_df_data(pred_date,instrument,pred_df=group_data,df_ref=dataset.df_all,ext_length=ext_length)   
                if complex_df is None:
                    logger.warning("data None,code:{}".format(instrument))
                    continue
                if complex_df.shape[0]<dataset.step_len:
                    logger.warning("not enough len,code:{},len:{}".format(instrument,complex_df.shape[0]))
                    continue                

                # 根据预测数据综合判断，取得匹配标志
                [match_flag,vr_class] = self.pred_data_jud(complex_df, dataset=dataset,ext_length=ext_length)
                if not match_flag:
                    continue
                
                # 使用收盘价格进行衡量         
                price_values = complex_df["label_ori"].values                
                # 取得实际价格信息，进行准确率判断
                price_list = price_values[-(dataset.pred_len):]
                # 以昨日收盘价为基准
                cur_price = price_values[-(dataset.pred_len+1)]
                price_data = np.array([cur_price] + price_list.tolist())
                correct = compute_price_class(price_data)                                                   
                match_item = [pred_date,instrument,correct,vr_class]
                match_list.append(match_item)
                # if correct!=2:
                #     self.data_viewer_correct.show_single_complex_pred_data(complex_df,dataset=dataset,save_path=self.pred_data_path+"/plot")
                # data_viewer_correct.show_single_complex_pred_data_visdom(complex_df,dataset=dataset)
                match_cnt += 1
            logger.debug("date:{} and match_cnt:{}".format(pred_date,match_cnt))
        df = pd.DataFrame(np.array(match_list),columns=match_columns)     
        return df,pred_df_total
    
    def get_pred_data_file(self,pred_date):
        file_path = "{}/pred_part/pred_df_total_{}.pkl".format(self.pred_data_path,pred_date)
        return file_path
           
    def show_correct_pred(self,stat_df,pred_df_total,dataset=None,show_num=10):
        
        incorrect_df = stat_df[stat_df["correct"]==0].iloc[:show_num]
        correct_df = stat_df[stat_df["correct"]==CLASS_SIMPLE_VALUE_MAX].iloc[:show_num]
        show_df = pd.concat([incorrect_df,correct_df])
        # show_df = stat_df[stat_df["instrument"]==66]
        data_viewer_correct = DataViewer(env_name="stat_pred_classify_correct")
        data_viewer_incorrect = DataViewer(env_name="stat_pred_classify_incorrect")
        for index,group_data in show_df.groupby(["instrument","date"]):
            instrument = int(group_data["instrument"].values[0])
            date = int(group_data["date"].values[0])
            correct = group_data["correct"].values[0]
            complex_item_df = self.combine_complex_df_data(date,instrument,pred_df=pred_df_total,df_ref=dataset.df_all) 
            if correct==CLASS_SIMPLE_VALUE_MAX:
                data_viewer_correct.show_single_complex_pred_data_visdom(complex_item_df,correct=correct,dataset=dataset)
            else:
                data_viewer_incorrect.show_single_complex_pred_data_visdom(complex_item_df,correct=correct,dataset=dataset)
            # self.data_viewer_correct.show_single_complex_pred_data(complex_item_df,correct=correct,dataset=dataset,save_path=self.pred_data_path+"/plot")
            # logger.debug("correct:{}".format(correct))
        
    def label_data_jud(self,row,dataset=None):
        """均线价值判断"""
        
        from visdom import Visdom
        # viz = Visdom(env="test",port=8098)  
        
        each_diff = []
        label_arr = []
        for i in range(2*dataset.pred_len):
            label_arr.append(row["label_{}".format(i)])
            if i<dataset.pred_len-1:
                each_diff.append(row["label_{}".format(i+1)] - row["label_{}".format(i)]) 
                
        label_arr_nor = normalization(np.array(label_arr))    
        # 计算均线斜率
        slope_arr = compute_series_slope(label_arr_nor)  
        max_value = np.max(label_arr)
        raise_range = (max_value - label_arr[0])/label_arr[0]
        # viz.line(
        #             X=[i for i in range(label_arr.shape[0])],
        #             Y=label_arr,
        #             win="test_label",
        #             name="label",
        #             update=None,
        #             opts={
        #                 'showlegend': True, 
        #                 'title': "test_label",
        #                 'xlabel': "step", 
        #                 'ylabel': "values", 
        #             },
        # )
        match_flag = False
        # 前平后起，符合
        if slope_arr[-1]>0.2 and slope_arr[-2]>0.2:
            if abs(slope_arr[-3])<0.1 and abs(slope_arr[-4])<0.1 and abs(slope_arr[-5])<0.1:
                # 根据断涨跌幅类别进行筛选
                if raise_range> 0.05:
                    match_flag = True
        # 前起后平，符合
   
        return match_flag        
    
    def pred_data_jud(self,ins_data,dataset=None,ext_length=25,head_range=1):
        """预测均线价值判断"""
        
        label_arr = ins_data["label"].values.tolist()
        ref_price_arr = ins_data["label_ori"].values[:-dataset.pred_len]
        price_arr = ins_data["label_ori"].values[-2*dataset.pred_len:-dataset.pred_len]
        pred_arr = ins_data["pred_data"].values[-dataset.pred_len:]
        
        # 分类信息判断
        vr_class = ins_data["vr_class"].values[0]
        match_flag = False
        rtn_obj = [match_flag,vr_class]

        # 前期走势需要比价平稳
        label_target = np.array([label_arr[-2*dataset.pred_len:-dataset.pred_len]]).transpose(1,0)
        label_slope_class = slope_classify_compute(label_target,threhold=2)
        if label_slope_class!=SLOPE_SHAPE_SMOOTH: 
            return rtn_obj
        #
        # 之前的价格涨幅筛选判断
        price_arr_slope = (price_arr[1:] - price_arr[:-1])/price_arr[:-1]
        # # 最近价格连续上涨不可信
        # if price_arr_slope[-1]>0 and price_arr_slope[-2]>0:
        #     return rtn_obj
        # 最近一天上涨幅度不能过高
        if (price_arr_slope[-1]*100)>6:
            return rtn_obj   
        # # 最后一天需要创出近期新高 
        # if ((ref_price_arr.max()-price_arr[-1])/price_arr[-1])>0.0001:
        #     return rtn_obj              
        # 与近期高点比较，不能差太多
        recent_max= ins_data["label"].values[:ext_length].max()
        if pred_arr[0]<recent_max and (recent_max-pred_arr[0])/recent_max>head_range/100:
            return rtn_obj          
        rtn_obj[0] = True  
        return rtn_obj


