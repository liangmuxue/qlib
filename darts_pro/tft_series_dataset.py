from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES
from trader.utils.date_util import tradedays,get_tradedays_dur

import pandas as pd
import numpy as np
import pickle
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.tft_dataset import TFTDataset
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.data_filter import DataFilter
from numba.core.types import none
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
logger = AppLogger()

class TFTSeriesDataset(TFTDataset):
    """
    自定义数据集，使用darts封装的TimeSeries类型数据
    """

    def __init__(self, step_len = 30,pred_len = 5,low_threhold=-5,high_threhold=5,
                 over_time=2,aug_type="no",model_type="lstm",col_def={},load_dataset_file=False,**kwargs):
        
        # 基层数据处理器
        self.data_extractor = StockDataExtractor() 
        super().__init__(col_def=col_def,step_len=step_len,pred_len=pred_len,load_dataset_file=load_dataset_file,**kwargs)
        
        self.future_covariate_col = col_def['future_covariate_col']
        self.past_covariate_col = col_def['past_covariate_col']
        self.static_covariate_col = col_def['static_covariate_col']    
        self.low_threhold = low_threhold
        self.high_threhold = high_threhold
        self.over_time = over_time        
        self.aug_type = aug_type
        self.model_type = model_type
        self.columns = self.get_seq_columns()
        self.kwargs = kwargs
        
        self.dbaccessor = DbAccessor({})
        self.emb_size = 0
        self.transform_inner = kwargs["transform_inner"]
        
    def _create_target_scalers(self,df):
        scaler_dict = {}
        group_column = self.get_group_rank_column()
        for item in df[group_column].unique():
            scaler_dict[item] = Scaler()
        return scaler_dict
            
    def get_emb_size(self):
        return self.emb_size
            
    def _pre_process_df(self,df,val_range=None):
        """数据预处理"""
        
        # 从数据库表中读取股票基础信息
        instrument_list = self.dbaccessor.do_query("select code,industry,tradable_shares from instrument_info where delete_flag=0")
        instrument_base_info = {}
        ext_info_arr = []
        for item in instrument_list:
            code = item[0]
            industry = item[1]
            tradable_shares = item[2]
            instrument_base_info[code] = {"industry":industry,"tradable_shares":tradable_shares}
            ext_info_arr.append([code,industry,tradable_shares])
        ext_info = pd.DataFrame(np.array(ext_info_arr),columns=["instrument","industry","tradable_shares"]).astype(
            {"instrument":str,"industry":int,"tradable_shares":float})   
        data_filter = DataFilter()
        # 清除序列长度不够的股票
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"]       
        df = data_filter.data_clean(df, self.step_len,valid_range=val_range,group_column=group_column,time_column=time_column)        
        # 生成时间字段
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
                
        # 删除空值，并重新编号
        df = df.dropna()    
        logger.debug("begin group process")
        columns = df.columns.values.tolist()
        columns = columns + ["industry","tradable_shares"]
        df["min_time"] = df.groupby(group_column)[time_column].transform("min")
        df[time_column] = df[time_column] - df["min_time"]
        df = df.drop(['min_time'], axis=1)
        # 放入数据库中的股票基础信息
        df = pd.merge(df,ext_info,on=["instrument"])        
        logger.debug("end group process")
        # group字段需要转换为数值型
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')   
        # 归一化指定字段
        df_rev = df.groupby(group_column,as_index=False).apply(lambda x: (x["REV5"] - x["REV5"].mean()) / x["REV5"].std()).reindex()
        df["REV5"] = df_rev.reset_index(level=0, drop=True)
        # 按照股票代码，新增排序字段，用于后续embedding
        rank_group_column = self.get_group_rank_column()
        df[rank_group_column] = df[group_column].rank(method='dense',ascending=False).astype("int")  
        self.build_group_rank_map(df)
        return df    
    
    def filter_by_indicator(self,df):
        """通过指标，进一步筛选数据集"""
        pass
    
    def stat_value_range(self,df):
        """统计单位时间内上涨下跌幅度情况"""
        
        def rl_apply(df_values):
            values = df_values.values
            raise_range = (values.max() - values[0])/values[0]
            fall_range = (values[0] - values.min())/values.min()
            if raise_range>fall_range:
                return raise_range
            return -(values[0] - values.min())/values[0]
        
        group_column = self.get_group_column()
        df_stat = df.groupby(group_column)["label"].rolling(window=self.step_len).apply(rl_apply)
        raise_desc = df_stat[df_stat>0].describe([.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]).values
        fall_desc = df_stat[df_stat<0].describe([.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99])
        max_value = raise_desc[-2]
        min_value = -fall_desc.values[2]
        print("max_value:{},min_value:{}".format(max_value,min_value))
    
    def check_clear_valid_df(self):
        """清理验证集合长度不符合的数据"""

        group_column = self.get_group_rank_column()
        time_column = self.get_time_column()
        accord_groups = []
        # 遍历全集，找出时间序号不足的数据
        for group_name,group_data in self.df_val.groupby(group_column):
            # 取得验证集最大时间序号，加上一定的长度（预测长度2倍），全集中的时间序号需要大于这个序号
            total_time_index = group_data[time_column].max() + self.pred_len * 2
            df_target = self.df_all[(self.df_all[group_column]==group_name)&(self.df_all[time_column]>=total_time_index)]
            if df_target.shape[0]>0:
                accord_groups.append(group_name)
            else:
                logger.info("not match:{}".format(group_name))
        
        self.df_all = self.df_all[self.df_all[group_column].isin(accord_groups)]
        self.df_val = self.df_val[self.df_val[group_column].isin(accord_groups)]
        self.df_train = self.df_train[self.df_train[group_column].isin(accord_groups)]
       
    def build_group_rank_map(self,df_data):
        """生成股票代码和rank序号的映射关系"""
        
        rank_group_column = self.get_group_rank_column()
        group_column = self.get_group_column()
        self.group_mapping = {}
        df_group = df_data.groupby(rank_group_column).head(1)
        for index, row in df_group.iterrows():
            self.group_mapping[row[rank_group_column]] = row[group_column]
    
    def prepare_inner_data(self,df_data):
        self.build_group_rank_map(df_data)
        # self.target_scalers = self._create_target_scalers(df_data)
    
    def get_group_code_by_rank(self,group_rank):    
        return self.group_mapping[group_rank]
           
    def create_base_data(self,segments_total=None,val_range=None,outer_df=None):
        """创建基础数据"""
        
        if outer_df is not None:
            # 如果从外部传递了数据，则直接使用
            self.df_all = outer_df
        else:
            # 根据参数计算时间范围，动态调用
            slc_total = slice(*segments_total)
            kwargs = {'col_set': ['feature', 'label'], 'data_key': 'learn'}
            # 首先取得pandas原始数据,使用train,valid数据集
            logger.debug("begin _prepare_seg")
            df_all = self._prepare_seg(slc_total, **kwargs)
            logger.debug("end _prepare_seg")
            # 提前生成嵌入向量空间长度，使用原始的股票数量，即使后续清理了部分股票，保留的应该仍然是大多数，不影响整体使用
            self.emb_size = df_all[self.get_group_column()].unique().shape[0] + 1
            # 前处理
            logger.debug("begin _pre_process_df")
            df_all = self._pre_process_df(df_all,val_range=val_range)
            # 为每个序列生成不同的scaler
            self.df_all = df_all
            logger.debug("emb size after p:{}".format(self.get_emb_size()))
        self.target_scalers = self._create_target_scalers(self.df_all)       
        
    def build_series_data(self,data_file=None,no_series_data=False,val_ds_filter=False):
        """从pandas数据生成时间序列类型数据"""
        
        if data_file is not None:
            # 直接从文件中读取数据
            with open(data_file, "rb") as fin:
                df_ref = pickle.load(fin)          
                self.train_range = self.kwargs["segments"]["train_total"] 
                self.valid_range = self.kwargs["segments"]["valid"] 
                self.prepare_inner_data(df_ref) 
                self.df_all = df_ref
                df_train = df_ref[df_ref["datetime"]<pd.to_datetime(str(self.valid_range[0]))]
                df_val = df_ref[(df_ref["datetime"]>=pd.to_datetime(str(self.valid_range[0]))) & (df_ref["datetime"]<=pd.to_datetime(str(self.valid_range[1])))]
                self.target_scalers = self._create_target_scalers(self.df_all)
            if no_series_data:
                return None          
            self.df_train = df_train
            self.df_val = df_val 
            return self.create_series_data(df_ref,df_train,df_val,fill_future=False,no_series_data=no_series_data)
            
        total_range = self.segments["train_total"]
        valid_range = self.segments["valid"]
        return self.build_series_data_step_range(total_range, valid_range,val_ds_filter=val_ds_filter,no_series_data=no_series_data)
        
    def build_series_data_step_range(self,total_range,val_range,fill_future=False,outer_df=None,val_ds_filter=False,no_series_data=False):
        """根据时间点参数，从pandas数据生成时间序列类型数据
           Params:total_range--完整时间序列开始和结束时间
                  val_range--预测时间序列开始和结束时间
        """    
        
        logger.info("begin create_base_data")
        # 生成基础数据
        self.create_base_data(total_range,val_range,outer_df=outer_df)

        # 默认使用valid配置进行数据集分割
        valid_start = val_range[0]
        valid_end = val_range[1]
        # 截取训练集与测试集
        df_all = self.df_all
        df_train = df_all[df_all["datetime"]<pd.to_datetime(str(valid_start))]
        df_val = df_all[(df_all["datetime"]>=pd.to_datetime(str(valid_start))) & (df_all["datetime"]<pd.to_datetime(str(valid_end)))]
        # 在筛选的过程中，有可能产生股票个数不一致的情况，取交集
        df_train = df_train[df_train[self.get_group_column()].isin(df_val[self.get_group_column()])]
        df_val = df_val[df_val[self.get_group_column()].isin(df_train[self.get_group_column()])]
        # 存储df数据，用于后续评估和回测等过程
        self.df_train = df_train
        self.df_val = df_val
        
        # 根据标志，决定是否进行验证集数据检查清理
        if val_ds_filter:
            self.check_clear_valid_df()
        # 如果只需要df数据，则不进行series数据生成
        if no_series_data:
            return
        return self.create_series_data(self.df_all,self.df_train,self.df_val,fill_future=fill_future)
        
    def create_series_data(self,df_all,df_train,df_val,fill_future=False):
        
        group_column = self.get_group_rank_column()
        target_column = self.get_target_column()
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        future_columns = self.get_future_columns()
        static_columns = self.get_static_columns()
        
        # 分别生成训练和测试序列数据
        train_series = TimeSeries.from_group_dataframe(df_train,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 static_cols=static_columns,
                                                 value_cols=target_column)   
        val_series = TimeSeries.from_group_dataframe(df_val,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 static_cols=static_columns,
                                                 value_cols=target_column) 
        total_series = TimeSeries.from_group_dataframe(df_all,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 static_cols=static_columns,
                                                 value_cols=target_column)    
        # 生成归一化的目标序列--取消，改为dataset内部进行           
        train_series_transformed = []
        val_series_transformed = []
        total_series_transformed = []
        
        if not self.transform_inner:
            for index,ts in enumerate(train_series):
                target_scaler = self.target_scalers[int(ts.static_covariates[group_column].values[0])]
                ts_transformed = target_scaler.fit_transform(ts)
                vs_transformed = target_scaler.transform(val_series[index])
                total_transformed = target_scaler.transform(total_series[index])
                train_series_transformed.append(ts_transformed)
                val_series_transformed.append(vs_transformed)
                total_series_transformed.append(total_transformed)
            
        def build_covariates(column_names,transform_columns=None):
            covariates_array = []
            for index,series in enumerate(train_series):
                group_col_val = series.static_covariates[group_column].values[0]
                scaler = Scaler()
                # 遍历并筛选出不同分组字段(股票)的单个dataframe
                df_item = df_all[df_all[group_column]==group_col_val]
                df_item_train = df_train[df_train[group_column]==group_col_val] 
                covariates = TimeSeries.from_dataframe(df_item,time_col=time_column,
                                                         freq='D',
                                                         fill_missing_dates=True,
                                                         value_cols=column_names)  
                train_covariates = TimeSeries.from_dataframe(df_item_train,time_col=time_column,
                                                         freq='D',
                                                         fill_missing_dates=True,
                                                         value_cols=column_names)       
                if self.transform_inner:
                    covariates_array.append(covariates)
                    continue
                
                # 使用训练数据fit，并transform到整个序列    
                scaler.fit(train_covariates)
                covariates_transformed = scaler.transform(covariates)    
                covariates_array.append(covariates_transformed)
            return covariates_array            

        # 生成过去协变量，并归一化
        logger.info("begin build_covariates")
        # 在过去协变量的数据中加入目标值原值，借用此协变量带入后续dataset中，用于原值分类计算--cancel
        # past_columns = [target_column] + past_columns
        past_convariates = build_covariates(past_columns)      
        # 生成未来协变量，并归一化
        future_convariates = build_covariates(future_columns)    
        # 生成静态协变量，并归一化
        # static_convariates = build_covariates(static_columns)   
        
        # 补充未来协变量数据,与验证数据相对应    
        if fill_future:           
            future_convariates = self.fill_future_data(future_convariates,future_columns,self.pred_len)
            
        # 分别返回用于训练预测的序列series_transformed，以及完整序列series
        if not self.transform_inner:
            return train_series_transformed,val_series_transformed,total_series_transformed,past_convariates,future_convariates
        return train_series,val_series,total_series,past_convariates,future_convariates   

    def fill_future_data(self,future_convariates,column_names,fill_length):
        """补充未来协变量数据"""
        
        def get_filling_data(last_datas,loop_values,length):
            data_dim = len(last_datas)
            index = [0] * data_dim
            for idx,last_data in enumerate(last_datas):
                # 取得需要插入的下标
                index[idx] = loop_values[:,idx].tolist().index(last_data) + 1
            datas = []
            for i in range(length):
                inner_data = [0] * data_dim
                for idx in range(data_dim):
                    # 根据起始标号，依次取得需要插入的标号及数据
                    total_index = i + index[idx]
                    real_index = total_index % len(loop_values[:,idx])
                    inner_data[idx] = loop_values[:,idx][real_index]
                datas.append(inner_data)
            return datas
            
        rtn_convariates = []  
        
        for conv in future_convariates:
            conv_values = conv.all_values()
            fill_datas = None
            for index in range(conv_values.shape[1]):
                conv_value = conv_values[:,index,:]
                last_values = [conv_value[-1,0]]
                # 取得需要生成数据的唯一值
                unique_values = np.unique(conv_value)
                unique_values = np.expand_dims(np.sort(unique_values),axis=-1)          
                # 根据固定数据，循环补充实际数组
                fill_data = get_filling_data(last_values,unique_values,fill_length)
                fill_data = np.array(fill_data)
                if fill_datas is None:
                    fill_datas = fill_data
                else:
                    fill_datas = np.concatenate((fill_datas,fill_data),axis=-1)
            conv = conv.append_values(fill_datas)
            rtn_convariates.append(conv)
        return rtn_convariates
    
    def build_static_covariates(self,series):
        """生成静态协变量"""
        
        df = series.pd_dataframe()
        columns = self.col_def["static_covariate_col"] 
        static_covariates = TimeSeries.from_times_and_values(
            times=df.index,
            values=df[columns].values,
            static_covariates=df["instrument"],
            columns=columns,
        )     
        return static_covariates 
            
    def build_future_covariates(self,series):
        """生成未来已知协变量"""
        df = series.pd_dataframe()
        columns = self.col_def["future_covariate_col"] 
        future_covariates = TimeSeries.from_times_and_values(
            times=df.index,
            values=df[columns].values,
            columns=columns,
        )     
        return future_covariates  
    
    def build_past_covariates(self,series,past_columns):
        """生成过去协变量系列"""
        
        past_covariates = []
        # 逐个生成每个列的协变量系列
        for column in past_columns:
            past = series.univariate_component(column)  
            past_covariates.append(past)   
        # 整合为一个系列
        past_covariates = concatenate(past_covariates, axis=1)
        return past_covariates     
    
    def align_pred_and_label(self,pred_series_list,val_series_list):
        """对齐并加工预测结果和实际结果"""
        
        group_column = self.get_group_rank_column()
        time_column = self.get_time_column()
        target_column = self.get_target_column()
        dt_index_column = self.get_datetime_index_column()
        
        pred_label_df_list = []
        for i in range(len(pred_series_list)):
            pred_series = pred_series_list[i]
            val_series = val_series_list[i]
            group_col_val = val_series.static_covariates[group_column].values[0]
            # 根据股票代码，以及时间点，取得原有df各个数据
            pred_df = self.df_all[(self.df_all[group_column]==group_col_val) 
                                  & (self.df_all[time_column]>=pred_series.time_index.start) 
                                  & (self.df_all[time_column]<pred_series.time_index.stop)]
            # 转换概率预测数据为中位数数据
            pred_center_data = get_pred_center_value(pred_series)
            pred_df[target_column] = pred_center_data.data
            # 为了后续进行和标签数据的比较，需要重新设置索引为日期列
            pred_df.set_index(dt_index_column, inplace=True)
            
            # 同样处理生成标签数据,注意在此和预测数据对齐
            label_df = self.df_all[(self.df_all[group_column]==group_col_val) 
                                  & (self.df_all[time_column]>=pred_series.time_index.start) 
                                  & (self.df_all[time_column]<pred_series.time_index.stop)]   
            val_df = val_series.pd_dataframe()
            val_df = val_df[(val_df.index>=pred_series.time_index.start) 
                                  & (val_df.index<pred_series.time_index.stop)]  
            # 使用加工后的label数据   
            label_df[target_column] = val_df[target_column].values
            label_df.set_index(dt_index_column, inplace=True)
            pred_label_df_list.append([pred_df,label_df])
        return pred_label_df_list


    def get_part_time_range(self,date_position,ref_df=None,offset=3):
        """根据给定日期，取得对应部分的数据集时间范围，需要满足预测要求"""
        
        # 首先取得总数据集范围，然后根据这个范围以及当前时间点，动态计算所需要的数据集范围以及验证集范围
        total_df_range = self.kwargs["segments"]["train_total"]
        total_start = total_df_range[0]
        total_end = total_df_range[1]
        total_range = [None,None]
        val_range = [None,None]
        
        # 全集的开始时间就是配置中的开始时间
        total_range[0] = total_start
        # 全集的结束时间为当前分割时间点
        total_range[1] = date_position
        # 验证数据集的开始时间为分割时间点往前的n个长度，其中n为配置中的训练时间序列长度。由于不同股票数据长度不一致，因此需要根据参照数据集进行移动。
        val_range[0],missing_instruments = self.shift_days(date_position, self.step_len,ref_df)
        # 验证数据集的结束时间为当前分割时间点
        val_range[1] = total_range[1]
        # 如果计算后的结束时间超出原有数据集结束时间，则返回空用于后续异常处理
        if total_range[1] is None:
            return None,None
        return total_range,val_range,missing_instruments

    def shift_days(self,date_position,n,ref_df,max_space=45):
        """取得对应日期前面第n天的日期"""
        
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"] 
        ref_date = get_tradedays_dur(date_position,-max_space*1)
        new_df = None
        missing_instruments = []
        min_date = None
        for group_name,group_data in ref_df.groupby(group_column):
            # 根据时间点取得对应序号，并根据序号取得序列间隔,需要考虑到当日没有数据的情况
            time_index = group_data[group_data["datetime"]<=str(date_position)][time_column].max()
            shift_time_index = time_index - n 
            # 如果越界，说明数据集中含有不具备的序列，忽略并记录
            if shift_time_index < 0:
                missing_instruments.append(group_name)
                logger.warning("missing_instruments:{}".format(group_name))
                continue
            group_date = group_data[group_data[time_column]<=shift_time_index]["datetime"].max()
            group_date = pd.to_datetime(group_date).date()
            # 如果前推的距离对应的日期太早（超过指定参数max_space），则忽略
            if group_date<ref_date:
                missing_instruments.append(group_name)
                logger.warning("missing_instruments with max_space:{}".format(group_name))
                continue           
            if min_date is None:
                min_date = group_date
            if group_date<min_date:
                min_date = group_date
                
        # 返回所有序列时间的最小值，以兼容全部数据集要求      
        return min_date,missing_instruments
                    
    def reverse_transform_preds(self,pres_series_list):
        """反向归一化(标准化)"""
        
        group_column = self.get_group_rank_column()
        result_list = []
        for series in pres_series_list:
            target_scaler = self.target_scalers[int(series.static_covariates[group_column].values[0])]
            result_list.append(target_scaler.inverse_transform(series))
        return result_list
    
    def get_series_by_group_code(self,series_list,group_code):
        group_column = self.get_group_rank_column()
        target_series = None
        for series in series_list:
            if group_code == int(series.static_covariates[group_column].values[0]):
                target_series = series
        return target_series
    
    def get_real_data(self,df_data,pred_list,instrument,extend_begin=0):   
        """根据预测数据的日期，取得数据集中的实际数值进行对照"""
        
        time_column = self.get_time_column()
        group_column = self.get_group_rank_column()
        # 取得预测数据,以及对应的开始时间序号
        pred_data = self.get_series_by_group_code(pred_list,instrument)
        pred_begin_time = pred_data.time_index.start
        df_item = df_data[(df_data[group_column]==instrument)]
        # 根据预测范围参数，取得几天以内的数据
        time_end = pred_begin_time + self.pred_len
        # 向前多取得一些原有数据，用于更好的可视化
        time_begin = pred_begin_time - extend_begin
        df_result = df_item[(df_item[time_column]>=time_begin)&(df_item[time_column]<time_end)].reset_index(drop=True)
        pred_center_data = get_pred_center_value(pred_data).data
        # 把预测数据从后面进行对齐插入
        pred_center_data = np.pad(pred_center_data,(extend_begin,0),'constant',constant_values=(0,0))
        df_result["pred"] = pred_center_data
        return df_result,pred_begin_time
    
    def get_data_by_trade_date(self,df,instrument,date):
        """根据股票代码和日期，取得当天交易数据"""

        time_column = self.get_time_column()
        group_column = self.get_group_rank_column()
        df_result = df[(df["datetime"]==pd.to_datetime(date))&(df[group_column]==instrument)]       
        if df_result.shape[0]==0:
            return None
        return df_result
    
    def get_data_by_group_code(self,group_code_list):   
        """根据分组字段数值，取得df数据"""
        
        group_column = self.get_group_column()
        df = self.df_all[self.df_all[group_column].isin(group_code_list)]
        return df
    
    def filter_pred_data_by_instrument(self,pred_list,instrument=[]):
        group_column = self.get_group_rank_column()
        filter_list = []
        for pred_item in pred_list:
            group_rank = int(pred_item.static_covariates[group_column].values[0])
            group_code = self.get_group_code_by_rank(group_rank)
            if group_code in instrument:
                filter_list.append(pred_item)
        
        return filter_list        
        
    def get_scaler_by_group_code(self,code):
        return self.target_scalers[code]
       
    def filter_pred_data_by_mape(self,pred_list,threhold=10,result_id=0):
        """根据得分筛选预测数据"""
        
        group_column = self.get_group_rank_column()
        # 如果没有指定批次号，则找到最近一个批次数据
        if result_id==0:
            result_id = self.dbaccessor.do_query("select id from pred_result order by id desc limit 1")[0][0]
        # 从之前的预测保存数据中取得小于阈值的股票
        results = self.dbaccessor.do_query("select instrument_rank,instrument,mape from pred_result_detail where result_id=%s and mape<=%s",params=(result_id,threhold))     
        results = np.array(results).astype(np.float)
        filter_list = []
        for pred_item in pred_list:
            group_rank = int(pred_item.static_covariates[group_column].values[0])
            group_code = self.get_group_code_by_rank(group_rank)
            if group_code in results[:,1]:
                filter_list.append(pred_item)
        
        return filter_list

    def filter_pred_data_by_corr(self,pred_list,threhold=0.5,result_id=0):
        """根据得分筛选预测数据"""
        
        group_column = self.get_group_rank_column()
        # 如果没有指定批次号，则找到最近一个批次数据
        if result_id==0:
            result_id = self.dbaccessor.do_query("select id from pred_result order by id desc limit 1")[0][0]
        # 从之前的预测保存数据中取得小于阈值的股票
        results = self.dbaccessor.do_query("select instrument_rank,instrument,mape,corr from pred_result_detail where result_id=%s and corr>=%s",params=(result_id,threhold))     
        results = np.array(results).astype(np.float)
        filter_list = []
        for pred_item in pred_list:
            group_rank = int(pred_item.static_covariates[group_column].values[0])
            group_code = self.get_group_code_by_rank(group_rank)
            if group_code in results[:,1]:
                filter_list.append(pred_item)
        
        return filter_list
        
    def ind_column_names(self,ind_len):
        ind_column = ["pred_{}".format(i) for i in range(ind_len)]
        return ind_column
    
    def build_df_data_for_pred_list(self,date_range,ind_len,pred_data_path=None,load_cache=False,type=None):
        """把连续预测结果数据，生成为dataframe格式
            Params:
                date_range 预测结果日期范围
                ind_len 预测结果长度
        """
        
        cache_file = pred_data_path + "/pred_cache_{}.npy".format(type)
        group_column = self.get_group_rank_column()
        # 每个预测数值作为一个字段
        ind_columns = self.ind_column_names(ind_len)
        # 加上日期及股票代码
        all_columns = [self.get_group_column()] + ["datetime"] + ind_columns       
        if not load_cache:
            data_array = []
            for date in date_range:
                # 动态取出之前存储的预测数据
                pred_series_list = self.get_pred_result(pred_data_path,date)
                logger.debug('pred_series_list process,{}'.format(date))
                for series in pred_series_list:
                    # 拼接预测数据到每个股票
                    group_rank = series.static_covariates[group_column].values[0]
                    group_item = self.get_group_code_by_rank(group_rank)
                    pred_center_data = get_pred_center_value(series).data
                    data_line = [float(group_item),float(date)] + pred_center_data.tolist()
                    data_array.append(data_line)
            data_array = np.array(data_array)
            np.save(cache_file,data_array)
        else:
            data_array = np.load(cache_file)
        df = pd.DataFrame(data_array,columns = all_columns) 
        df["datetime"] = df["datetime"].astype(int).astype(str)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[self.get_group_column()] = df[self.get_group_column()].astype(int)
        return df
        
    def get_pred_result(self,pred_data_path,cur_date):
        """取得之前生成的预测数据"""
        
        data_path = pred_data_path + "/" + str(cur_date) + ".pkl"
        with open(data_path, "rb") as fin:
            pred_series_list = pickle.load(fin)            
            return pred_series_list        
        
        
    def fill_miss_data(self,df):
        """填充某些天没哟交易的空数据"""
        
        list_date = list(pd.date_range(df['datetime'].min(),df['datetime'].max()).astype(str))
        list_ticker = df[self.get_group_column()].unique().tolist()
        combination = list(itertools.product(list_date,list_ticker))
        edf = pd.DataFrame(combination,columns=["datetime",self.get_group_column()])
        edf["datetime"] = pd.to_datetime(edf["datetime"])
        processed_full = edf.merge(df,on=["datetime",self.get_group_column()],how="left")
        processed_full = processed_full[processed_full['datetime'].isin(df['datetime'])]
        processed_full = processed_full.sort_values(['datetime',self.get_group_column()])        
        processed_full = processed_full.fillna(0)
        
        return processed_full

    def transfer_pred_data(self,pred_data):
        """预测数据转换为逐个元素差
           Params:
                pred_data 原预测数据
        """        
        
        mean_value = np.expand_dims(np.mean(pred_data,axis=1),axis=-1)
        new_array = []
        for row in range(pred_data.shape[0]): 
            row_array = []
            for col in range(pred_data.shape[1]):
                if col+1>=pred_data.shape[1]:
                    break
                value = pred_data[row,col+1] - pred_data[row,col] 
                row_array.append(value)    
            new_array.append(row_array)
        result = np.array(new_array)
        result = np.concatenate((result,mean_value),axis=-1)
        return result
        
    def combine_pred_class(self,pred_class_total):
        pred_class = F.softmax(pred_class_total,dim=-1)
        pred_class = torch.max(pred_class,dim=-1)
        return pred_class

    def build_pred_class_real(self,pred_class):
        """根据softmax数据，生成实际分类信息"""
        
        pred_class = torch.tensor(pred_class)
        pred_class_max = self.combine_pred_class(pred_class)   
        pred_class_real = pred_class_max[1].item()    
        return pred_class_real
        
    def pred_data_columns(self):
        pred_len = self.pred_len
        data_columns = ["instrument","date"]
        # 预测数据
        pred_columns = ["pred_{}".format(i) for i in range(pred_len)]
        # 标签数据
        label_columns = ["label_{}".format(i) for i in range(pred_len)]
        # 实际价格（滑动窗之前的原始数据）
        price_columns = ["price_{}".format(i) for i in range(pred_len*2)]
        data_columns = data_columns + pred_columns + label_columns + price_columns      
        return data_columns 
        
    def get_datetime_with_index(self,instrument,begin_time_index,end_time_index):
        """取得指定股票对应时间索引的具体日期范围"""
        
        df_target = self.query_data(instrument, [begin_time_index,end_time_index])
        return df_target["datetime"].dt.strftime("%Y%m%d").astype(int).values.tolist()        
    
    def get_series_value_by_range(self,series,range):
        target_series = series.drop_before(range[0]).drop_after(range[1])
        return target_series
        
    def query_data(self,instrument,time_range):
        df_target = self.df_all[(self.df_all["instrument"]==instrument)&(self.df_all["time_idx"]>=time_range[0])&(self.df_all["time_idx"]<time_range[1])]
        return df_target   
    
    
    
    