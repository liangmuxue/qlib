import pickle
import numpy as np
import pandas as pd
import os

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

from darts_pro.data_extension.custom_model import TFTExtModel
from cus_utils.common_compute import normalization,compute_series_slope,compute_price_range,slope_classify_compute,comp_max_and_rate
from tft.class_define import SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,CLASS_SIMPLE_VALUE_MAX
from trader.busi_compute import slope_status

from cus_utils.log_util import AppLogger
from trader import pred_recorder
logger = AppLogger()

class MlIntergrate():
    """工作流程类，以RQALPHA串接Qlib,获取相关上下文数据和环境"""
    
    def __init__(
        self,
        provider_uri = "/home/qdata/qlib_data/custom_cn_data",
        config_path="custom/config/darts/workflow_backtest.yaml",
        **kwargs,
    ):
        # 初始化qlib数据
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        with open(config_path) as fp:
            config = yaml.safe_load(fp)    
            
        # 配置文件内容
        self.model_cfg =  config["task"]["model"]
        self.dataset_cfg = config["task"]["dataset"]
        self.backtest_cfg = config["task"]["backtest"]
        
        # 初始化
        self.pred_data_path = self.model_cfg["kwargs"]["pred_data_path"]
        self.dynamic_file = self.model_cfg["kwargs"]["dynamic_file"]
        self.load_dataset_file = self.model_cfg["kwargs"]["load_dataset_file"]
        self.save_dataset_file = self.model_cfg["kwargs"]["save_dataset_file"]
        # 初始化model和dataset
        optargs = self.model_cfg["kwargs"]["optargs"]
        model = TFTExtModel.load_from_checkpoint(optargs["model_name"],work_dir=optargs["work_dir"],best=False)
        dataset = init_instance_by_config(self.dataset_cfg)
        df_data_path = self.pred_data_path + "/df_all.pkl"
        dataset.build_series_data(df_data_path,no_series_data=True)    
        # 生成recorder,用于后续预测数据处理
        record_cfg = config["task"]["record"]
        placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
        record_cfg = fill_placeholder(record_cfg, placehorder_value)        
        
        with R.start(experiment_name="workflow", recorder_name=None):
            rec = R.get_recorder()
            recorder = init_instance_by_config(
                record_cfg,
                recorder=rec,
            )      
             
        self.model = model
        self.dataset = dataset
        self.pred_recorder = recorder
        self.kwargs = kwargs

    def prepare_data(self,pred_date):   
        # 加载预测文件
        
        if self.dynamic_file:
            # 如果动态加载模式，则从目录中找出最接近此预测日期的文件
            pred_part_path = self.pred_data_path + "/pred_part"
            files = os.listdir(pred_part_path)
            match_file = None
            for file in files:
                file_data_part = int(file.split(".")[0].split("_")[-1])
                if file_data_part>pred_date:
                    continue
                if match_file is None:
                    match_file = file_data_part
                    continue
                if file_data_part>match_file:
                    match_file = file_data_part
            pred_data_path = "{}/pred_part/pred_df_total_{}.pkl".format(self.pred_data_path,match_file)
        else:
            pred_data_path = self.pred_data_path + "/pred_df_total.pkl"
        
        with open(pred_data_path, "rb") as fin:
            df_pred = pickle.load(fin)
            self.pred_df = self.load_pred_data(df_pred) 
      
    def load_pred_data(self,df_pred):
     
        df_pred["pred_date"] = df_pred["pred_date"].astype(int)
        df_pred["class1"] = df_pred["class1"].astype(int) 
        df_pred["class2"] = df_pred["class2"].astype(int) 
        df_pred["vr_class"] = df_pred["vr_class"].astype(int) 
        return df_pred
 
    def filter_buy_candidate(self,pred_date):
        """根据预测计算，筛选可以买入的股票"""
        
        date_pred_df = self.pred_df[(self.pred_df["pred_date"]==pred_date)]
        return self.filter_buy_candidate_data(pred_date,date_pred_df)

    def filter_buy_candidate_data(self,pred_date,date_pred_df):
        """根据预测计算，筛选可以买入的股票"""
        
        ext_length = self.kwargs["ext_length"]
        candidate_list = []
        for instrument,group_data in date_pred_df.groupby("instrument"):
            instrument = int(instrument)
            # 生成对应日期的单个股票的综合数据
            complex_df = self.combine_complex_df_data(pred_date,instrument,pred_df=date_pred_df,df_ref=self.dataset.df_all,ext_length=ext_length)   
            if complex_df is None:
                continue
            # 根据预测数据综合判断，取得匹配标志
            (match_flag,pred_class_real,vr_class) = self.pred_data_jud(complex_df, dataset=self.dataset,ext_length=ext_length)
            pred_class_real = pred_class_real.tolist()
            if not match_flag:
                continue   
            candidate_list.append(instrument)     
        return candidate_list   
       
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
        df_pred_item = pred_df[(pred_df["instrument"]==instrument)&(pred_df["pred_date"]==pred_date)]
        if df_pred_item.shape[0]==0:
            return None
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
        if pad_len<0:
            print("ggg") 
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
        
    def pred_data_jud(self,ins_data,dataset=None,ext_length=25):
        """预测均线价值判断"""
        
        label_arr = ins_data["label"].values.tolist()
        # 计算均线涨跌幅度,取预测日期前n天的（n为预测长度））
        label_slope_arr = compute_price_range(label_arr[-2*dataset.pred_len:-dataset.pred_len])
                     
        pred_arr = ins_data["label"].values[-dataset.pred_len:]
        # 计算预测均线涨跌幅度
        slope_arr = compute_price_range(pred_arr)  
        
        # 分类信息判断
        pred_class = np.array([ins_data["class1"].values[0],ins_data["class2"].values[0]])
        vr_class = ins_data["vr_class"].values[0]
                
        match_flag = False
        rtn_obj = [match_flag,pred_class,vr_class]
        
        # RSI指标需要在50以上
        rsi = ins_data["RSI20"].values[-2*dataset.pred_len:-dataset.pred_len]
        if np.any(rsi<50):
            return rtn_obj
        # MACD指标需要在0以上
        macd = ins_data["MACD"].values[-2*dataset.pred_len:-dataset.pred_len]
        if np.any(macd<0):
            return rtn_obj
                
        # 检查之前的实际均线形态，要求是比较平
        status = slope_status(label_slope_arr)
        if status != SLOPE_SHAPE_SMOOTH:
            return rtn_obj
        
        # 预测最后一个时间段数据需要上涨
        if slope_arr[-1]< 0:
            return rtn_obj
        # 整体需要上涨
        if pred_arr[-1] - pred_arr[0] < 0:
            return rtn_obj         

        # 需要符合涨跌幅类别
        if vr_class<2:
            return rtn_obj
        # 一直起，符合
        if pred_class[0]==0 and pred_class[1]==0:
            rtn_obj[0] = True           
        # 前平后起，符合
        if pred_class[0]==2 and pred_class[1]==0:
            rtn_obj[0] = True
        # 前起后平，符合
        # if pred_class_real[0]==0 and pred_class_real[1]==2:
        #     match_flag = True   
        return rtn_obj
    
    def measure_pos(self,pred_date,instrument):
        """对持仓对象进行衡量"""
        
        ext_length = self.kwargs["ext_length"]
        date_pred_df = self.pred_df[(self.pred_df["pred_date"]==pred_date)]
        if date_pred_df.shape[0]==0:
            logger.warning("pred df empty:{}".format(pred_date))
        complex_item_df = self.combine_complex_df_data(pred_date,instrument,pred_df=date_pred_df,df_ref=self.dataset.df_all,ext_length=ext_length)  
        if complex_item_df is None:
            return False
        # 取得预测趋势分类信息
        pred_class = np.array([complex_item_df["class1"].values[0],complex_item_df["class2"].values[0]])
        vr_class = complex_item_df["vr_class"].values[0]
        # 如果趋势分类分数小于0，并且走势阶段至少有一个阶段是下跌类型，则卖出
        if vr_class==0 and np.sum(pred_class==SLOPE_SHAPE_FALL)>=1:
            return True
        return False
    
    def record_results(self,result_df):
        pass
        