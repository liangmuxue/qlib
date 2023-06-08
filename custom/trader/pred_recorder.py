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
from cus_utils.common_compute import normalization,compute_series_slope,compute_price_range,slope_classify_compute,comp_max_and_rate
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,CLASS_SIMPLE_VALUE_MAX
from trader.utils.date_util import get_tradedays
from cus_utils.tensor_viz import TensorViz
from cus_utils.log_util import AppLogger
from trader.busi_compute import slope_status
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
        
        self.df_ref = dataset.df_all     
        self.pred_result_columns = ['pred_date','time_idx','instrument','class1','class2','vr_class','pred_data'] 
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
        df_pred["class1"] = df_pred["class1"].astype(int) 
        df_pred["class2"] = df_pred["class2"].astype(int) 
        df_pred["vr_class"] = df_pred["vr_class"].astype(int) 
        return df_pred
                    
    def build_pred_result(self,start_time,end_time):
        """逐天生成预测数据"""
        
        # 取得日期范围，并遍历生成预测数据
        date_range = get_tradedays(start_time,end_time)
        data_total = None
        for cur_date in date_range:
            # 动态生成数据
            logger.debug("begin predict_process")  
            pred_combine_data = self.predict_process(cur_date,outer_df=self.df_ref)
            logger.debug("begin build_pred_data")  
            data_pred = self.build_pred_data(cur_date,pred_combine_data,df_ref=self.df_ref)
            if data_total is None:
                data_total = data_pred
            else:
                data_total = np.concatenate((data_total,data_pred),axis=0) 
                 
            logger.debug("build df_pred,data:{}".format(cur_date))   
             
        # 一并生成DataFrame
        df_total = pd.DataFrame(data_total,columns=self.pred_result_columns)
        # 存储数据
        pred_data_path = self.pred_data_path + "/" + self.pred_data_file
        with open(pred_data_path, "wb") as fout:
            pickle.dump(df_total, fout)     
        return df_total
    
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
            pred_class = pred_class_total[index]
            pred_class_max = self.dataset.combine_pred_class(pred_class)   
            pred_class_real = pred_class_max[1].numpy()
            vr_class_data = vr_class_total[index]
            vr_class,vr_class_confidence = comp_max_and_rate(np.array(vr_class_data))
            data_item = np.array([[int(pred_date) for i in range(time_index.shape[0])],
                                  time_index.tolist(),
                                 [group_item for i in range(time_index.shape[0])],
                                 [pred_class_real[0] for i in range(time_index.shape[0])],
                                 [pred_class_real[1] for i in range(time_index.shape[0])],
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
    
    def predict_process(self,cur_date,outer_df=None):
        """执行预测过程"""
        
        # 根据时间点，取得对应的输入时间序列范围
        total_range,val_range,missing_instruments = self.dataset.get_part_time_range(cur_date,ref_df=self.df_ref)
        # 如果不满足预测要求，则返回空
        if total_range is None:
            self.log("pred series none")
            return None
        # 如果包含不符合的数据，再次进行过滤
        if len(missing_instruments)>0:
            outer_df_filter = outer_df[~outer_df[self.dataset.get_group_column()].isin(missing_instruments)]
        else:
            outer_df_filter = outer_df     
        # 从执行器模型中取得已经生成好的模型变量
        my_model = self.model.model
        # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            self.dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=outer_df_filter)
        my_model.fit(series_total,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,num_loader_workers=8,verbose=True,epochs=-1)            
        # 对验证集进行预测，得到预测结果   
        pred_combine = my_model.predict(n=self.dataset.pred_len, series=val_series_transformed,num_samples=200,num_loader_workers=8,
                                            past_covariates=past_convariates,future_covariates=future_convariates)
        pred_series_list = [item[0] for item in pred_combine]
        pred_class_total = [item[1] for item in pred_combine]
        vr_class_total = [item[2] for item in pred_combine] 
        # 归一化反置，恢复到原值
        # pred_series_list = self.dataset.reverse_transform_preds(pred_series_list)
        for series in pred_series_list:
            group_rank = series.static_covariates["instrument_rank"].values[0]
            code = self.dataset.get_group_code_by_rank(group_rank)
        return pred_series_list,pred_class_total,vr_class_total

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
        new_columns = df_item.columns.tolist() + ["pred_date","pred_data","class1","class2","vr_class"]
        df_item = df_item.reindex(columns=new_columns)
        df_item["pred_date"] = [pred_date for i in range(df_item.shape[0])] 
        # 预测数据,前面补0
        pred_data = df_pred_item["pred_data"].values
        pad_len = df_item.shape[0]-pred_data.shape[0]
        data_line = np.pad(pred_data,(pad_len,0),'constant',constant_values=(0,0))          
        df_item["pred_data"] = data_line
        # 走势分类信息处
        class1 = df_pred_item["class1"].values[0]
        class2 = df_pred_item["class2"].values[0]
        # 为了方便，在每一行中都放入同样的分类信息
        df_item["class1"] = [class1 for i in range(df_item.shape[0])]
        df_item["class2"] = [class2 for i in range(df_item.shape[0])]
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
        corr_2_rate = df[df["correct"]==2].shape[0]/df.shape[0]
        corr_1_rate = df[df["correct"]==1].shape[0]/df.shape[0]
        corr_0_rate = df[df["correct"]==0].shape[0]/df.shape[0]
        print("corr_rate:{},corr_1:{},corr_0:{}".format(corr_2_rate,corr_1_rate,corr_0_rate))   
        self.show_correct_pred(df,pred_df_total,dataset=dataset) 
        return df
    
    def filter_cancidate_result(self,dataset=None,ext_length=25,pred_file=None):
        """筛选出合适的品种"""

        date_list = get_tradedays(self.classify_range[0],self.classify_range[1])
        match_list = []
        match_cnt = 0        
        match_columns = ["date","instrument","correct","pred_class","vr_class","price_raise_range","price_down_range"]
        data_viewer_correct = DataViewer(env_name="stat_pred_classify_correct")
        pred_df_total = None
        for pred_date in date_list:   
            # 每日动态加载预测结果数据
            if pred_file is None:
                pred_data_file = self.get_pred_data_file(pred_date)
            else:
                pred_data_file = pred_file
            date_pred_df_ori = self.load_pred_data(pred_data_file=pred_data_file)
            if pred_df_total is None:
                pred_df_total = date_pred_df_ori
            else:
                pred_df_total = pd.concat([pred_df_total,date_pred_df_ori])
            # 筛选出分类上涨类股票
            date_pred_df = date_pred_df_ori[(date_pred_df_ori["vr_class"]==2)]            
            pred_date = int(pred_date)  
            # 走势符合前平后起，或先起后平
            date_pred_df = date_pred_df.groupby('instrument').filter(
                lambda x: (x.iloc[0]["class1"]==SLOPE_SHAPE_SMOOTH and x.iloc[0]["class2"]==SLOPE_SHAPE_RAISE) or 
                (x.iloc[0]["class1"]==SLOPE_SHAPE_RAISE and x.iloc[0]["class2"]==SLOPE_SHAPE_SMOOTH) or  
                (x.iloc[0]["class1"]==SLOPE_SHAPE_RAISE and x.iloc[0]["class2"]==SLOPE_SHAPE_RAISE)
            )             
            for instrument,group_data in date_pred_df.groupby("instrument"):
                # 生成对应日期的单个股票的综合数据
                complex_df = self.combine_complex_df_data(pred_date,instrument,pred_df=group_data,df_ref=dataset.df_all,ext_length=ext_length)   
                if complex_df is None:
                    continue
                # 使用收盘价格进行衡量         
                price_values = complex_df["label_ori"].values
                # 根据预测数据综合判断，取得匹配标志
                (match_flag,pred_class_real,vr_class) = self.pred_data_jud(complex_df, dataset=dataset,ext_length=ext_length)
                pred_class_real = pred_class_real.tolist()
                if not match_flag:
                    continue
                
                # 取得实际价格信息，进行准确率判断
                price_list = price_values[-(dataset.pred_len):]
                # 以昨日收盘价为基准
                cur_price = price_values[-(dataset.pred_len+1)]
                price_raise_range = (price_list.max() - cur_price)/cur_price
                price_down_range = (price_list.min() - cur_price)/cur_price
                if price_raise_range > 0.05:
                    correct = 2
                elif price_raise_range > 0.03:
                    correct = 1
                elif price_raise_range > 0.01:
                    correct = 0         
                elif price_down_range > -0.01:
                    correct = 0        
                elif price_down_range > -0.03:
                    correct = -1
                else:
                    correct = -2                                                     
                match_item = [pred_date,instrument,correct,pred_class_real,vr_class,price_raise_range,price_down_range]
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
        
        incorrect_df = stat_df[stat_df["correct"]<0].iloc[:show_num]
        correct_df = stat_df[stat_df["correct"]==2].iloc[:show_num]
        show_df = pd.concat([incorrect_df,correct_df])
        show_df = stat_df[stat_df["instrument"]==66]
        data_viewer_correct = DataViewer(env_name="stat_pred_classify_correct")
        data_viewer_incorrect = DataViewer(env_name="stat_pred_classify_incorrect")
        for index,group_data in show_df.groupby(["instrument","date"]):
            instrument = int(group_data["instrument"].values[0])
            date = int(group_data["date"].values[0])
            correct = group_data["correct"].values[0]
            complex_item_df = self.combine_complex_df_data(date,instrument,pred_df=pred_df_total,df_ref=dataset.df_all) 
            if correct==2:
                data_viewer_correct.show_single_complex_pred_data_visdom(complex_item_df,correct=correct,dataset=dataset)
            else:
                data_viewer_incorrect.show_single_complex_pred_data_visdom(complex_item_df,correct=correct,dataset=dataset)
            # self.data_viewer_correct.show_single_complex_pred_data(complex_item_df,correct=correct,dataset=dataset,save_path=self.pred_data_path+"/plot")
            logger.debug("correct:{}".format(correct))
        
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
    
    def pred_data_jud(self,ins_data,dataset=None,ext_length=25):
        """预测均线价值判断"""
        
        label_arr = ins_data["label"].values.tolist()
        # 计算均线涨跌幅度,取预测日期前n天的（n为预测长度））
        label_slope_arr = compute_price_range(label_arr[-2*dataset.pred_len:-dataset.pred_len])
                     
        pred_arr = ins_data["pred_data"].values[-dataset.pred_len:]
        # 计算预测均线涨跌幅度
        slope_arr = compute_price_range(pred_arr)  
        
        # 分类信息判断
        pred_class = np.array([ins_data["class1"].values[0],ins_data["class2"].values[0]])
        vr_class = ins_data["vr_class"].values[0]
                
        match_flag = False
        rtn_obj = [match_flag,pred_class,vr_class]
        
        # RSI指标需要在50以上
        # rsi = ins_data["RSI20"].values[-2*dataset.pred_len:-dataset.pred_len]
        # if np.any(rsi<50):
        #     return rtn_obj
        # # MACD指标需要在0以上
        # macd = ins_data["MACD"].values[-2*dataset.pred_len:-dataset.pred_len]
        # if np.any(macd<0):
        #     return rtn_obj
        
        # 检查之前的实际均线形态，要求是比较平
        # status = slope_status(label_slope_arr)
        # if status != SLOPE_SHAPE_SMOOTH:
        #     return rtn_obj
        
        # 预测最后一个时间段数据需要上涨
        # if slope_arr[-1]< 0:
        #     return rtn_obj
        # 整体需要上涨
        # if pred_arr[-1] - pred_arr[0] < 0:
        #     return rtn_obj         

        # 需要符合涨跌幅类别
        # if vr_class<2:
        #     return rtn_obj
        # 一直起，符合
        # if pred_class[0]==0 and pred_class[1]==0:
        #     rtn_obj[0] = True           
        # # 前平后起，符合
        # if pred_class[0]==2 and pred_class[1]==0:
        #     rtn_obj[0] = True
        # 前起后平，符合
        # if pred_class_real[0]==0 and pred_class_real[1]==2:
        #     match_flag = True   
        rtn_obj[0] = True  
        return rtn_obj


