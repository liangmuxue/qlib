from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES

import pandas as pd
import numpy as np
import pickle
import itertools

from datetime import datetime
from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.data_filter import DataFilter
from numba.core.types import none
from cus_utils.db_accessor import DbAccessor

from cus_utils.log_util import AppLogger
from gym import logger
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
        # df_all = self.prepare("train_total", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
    def _create_target_scalers(self,df):
        scaler_dict = {}
        group_column = self.get_group_rank_column()
        for item in df[group_column].unique():
            scaler_dict[item] = Scaler()
        return scaler_dict
            
    def get_emb_size(self):
        return self.emb_size
            
    def _pre_process_df(self,df,val_range=None):
        data_filter = DataFilter()
        # 清除序列长度不够的股票
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"]       
        df = data_filter.data_clean(df, self.step_len,valid_range=val_range,group_column=group_column,time_column=time_column)        
        # 生成时间字段
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
        # df["label"] = df["label"].astype("float64")
        # 使用前几天的移动平均值作为目标数值
        df["label_ori"]  = df["label"]
        df["label"]  = df.groupby(group_column)[self.get_target_column()].rolling(window=self.pred_len,min_periods=1).mean().reset_index(0,drop=True)
        # 删除空值，并重新编号
        df = df.dropna()    
        new_df = None
        for group_name,group_data in df.groupby(group_column):
            group_data[time_column] -= group_data[time_column].min()
            if new_df is None:
                new_df = group_data
            else:
                new_df = pd.concat([new_df,group_data])
        df = new_df
        # group字段需要转换为数值型
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')   
        # 按照股票代码，新增排序字段，用于后续embedding
        rank_group_column = self.get_group_rank_column()
        df[rank_group_column] = df[group_column].rank(method='dense',ascending=False).astype("int")  
        self.build_group_rank_map(df)
        return df    
    
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
            df_all = self._prepare_seg(slc_total, **kwargs)
            # 提前生成嵌入向量空间长度，使用原始的股票数量，即使后续清理了部分股票，保留的应该仍然是大多数，不影响整体使用
            self.emb_size = df_all[self.get_group_column()].unique().shape[0] + 1
            # 前处理
            df_all = self._pre_process_df(df_all,val_range=val_range)
            # 为每个序列生成不同的scaler
            self.df_all = df_all
            print("emb size after p:",self.get_emb_size())
        self.target_scalers = self._create_target_scalers(self.df_all)       
        
    def build_series_data(self,data_file=None):
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
            return self.create_series_data(df_ref,df_train,df_val,fill_future=False)
            
        total_range = self.segments["train_total"]
        valid_range = self.segments["valid"]
        return self.build_series_data_step_range(total_range, valid_range)
        
    def build_series_data_step_range(self,total_range,val_range,fill_future=False,outer_df=None):
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
        # 存储df数据，用于后续评估和回测等过程
        self.df_train = df_train
        self.df_val = df_val
        
        return self.create_series_data(df_all,df_train,df_val,fill_future=fill_future)
        
    def create_series_data(self,df_all,df_train,df_val,fill_future=False):
        
        group_column = self.get_group_rank_column()
        target_column = self.get_target_column()
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        future_columns = self.get_future_columns()
        
        # 分别生成训练和测试序列数据
        train_series = TimeSeries.from_group_dataframe(df_train,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 value_cols=target_column)    
        val_series = TimeSeries.from_group_dataframe(df_val,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 value_cols=target_column) 
              
        train_series_transformed = []
        val_series_transformed = []
        # 生成归一化的目标序列
        for index,ts in enumerate(train_series):
            target_scaler = self.target_scalers[int(ts.static_covariates[group_column].values[0])]
            ts_transformed = target_scaler.fit_transform(ts)
            vs_transformed = target_scaler.transform(val_series[index])
            train_series_transformed.append(ts_transformed)
            val_series_transformed.append(vs_transformed)
        self.target_scalers
        
        def build_covariates(column_names):
            covariates_array = []
            for series in train_series_transformed:
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
                # 使用训练数据fit，并transform到整个序列    
                scaler.fit(train_covariates)
                covariates_transformed = scaler.transform(covariates)    
                covariates_array.append(covariates_transformed)
            return covariates_array            

        # 生成过去协变量，并归一化
        logger.info("begin build_covariates")
        past_convariates = build_covariates(past_columns)      
        # 生成未来协变量，并归一化
        future_convariates = build_covariates(future_columns)    
        # 补充未来协变量数据,与验证数据相对应    
        if fill_future:           
            future_convariates = self.fill_future_data(future_convariates,future_columns,self.pred_len)
        # 分别返回用于训练预测的序列series_transformed，以及完整序列series
        return train_series_transformed,val_series_transformed,past_convariates,future_convariates
            

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
            last_values = conv[-1].all_values()[:,:,0]
            # 取得需要生成数据的唯一值
            unique_values = np.unique(conv.all_values(),axis=0)
            unique_values = np.sort(unique_values[:,:,0],axis=0)            
            # 根据固定数据，循环补充实际数组
            fill_data = get_filling_data(last_values,unique_values,fill_length)
            conv = conv.append_values(np.expand_dims(np.array(fill_data),axis=-1))
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


    def get_part_time_range(self,date_position,ref_df=None):
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
        val_range[0] = self.shift_days(date_position, (self.step_len-self.pred_len),ref_df)
        # 验证数据集的结束时间为当前分割时间点
        val_range[1] = total_range[1]
        # 如果计算后的结束时间超出原有数据集结束时间，则返回空用于后续异常处理
        if total_range[1] is None:
            return None,None
        return total_range,val_range

    def shift_days(self,date_position,n,ref_df):
        """取得对应日期前面第n天的日期"""
        
        group_column = self.get_group_column()
        time_column = self.col_def["time_column"] 
        new_df = None
        for group_name,group_data in ref_df.groupby(group_column):
            # 根据时间点取得对应序号，并根据序号取得序列间隔,需要考虑到当日没有数据的情况
            time_index = group_data[group_data["datetime"]<=str(date_position)][time_column].max()
            shift_time_index = time_index - n 
            # 如果越界，说明数据集中含有不具备的序列，返回空处理异常
            if shift_time_index < 0:
                return None
            data = group_data[group_data[time_column]==shift_time_index]
            if new_df is None:
                new_df = data
            else:
                new_df = pd.concat([new_df,data])
                
        # 返回所有序列时间的最小值，以兼容全部数据集要求      
        return new_df["datetime"].min()
                    
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

    def filter_pred_data_by_corr(self,pred_list,threhold=0.8,result_id=0):
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
    
    def combine_complex_data(self,df_ref,date_range,pred_data_path=None,load_cache=False,type="train"):
        """合并预测数据，以及实际行情数据
           Params:
                df_ref 数据集
                date_range 合并数据日期范围
        """
        
        cache_file = pred_data_path + "/classify_data_cache_{}.npy".format(type)
        ref_cache_file = pred_data_path + "/classify_data_cache_ref_{}.npy".format(type)
        group_column = self.get_group_rank_column()
        (start_time,end_time) = date_range
        # 取得日期列表
        df_range = df_ref[(df_ref["datetime"]>=pd.to_datetime(str(start_time))) & (df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        date_list = df_range["datetime"].dt.strftime('%Y%m%d').unique()
        if not load_cache:
            data_array = []
            ref_price_array = []
            for date in date_list:
                # 动态取出之前存储的预测数据
                try:
                    pred_series_list = self.get_pred_result(pred_data_path,date)
                    # pred_series_list = self.filter_pred_data_by_mape(pred_series_list,result_id=3)
                    pred_series_list = self.filter_pred_data_by_corr(pred_series_list,result_id=6)
                except Exception as e:
                    print("no data for {}".format(date))
                    continue
                logger.debug('pred_series_list process,{}'.format(date))
                for series in pred_series_list:
                    # 拼接预测数据到每个股票
                    group_rank = series.static_covariates[group_column].values[0]
                    group_item = self.get_group_code_by_rank(group_rank)
                    pred_center_data = get_pred_center_value(series).data
                    # 前边是数据，最后一位是标签
                    data_line = pred_center_data.tolist()
                    # 取得下几个个交易日收盘涨跌幅数据，作为标签参考数据
                    real_label_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref["datetime"]>date)]["label_ori"].values
                    item_value = 0
                    item_price_list = []
                    real_range = min(self.pred_len,len(real_label_data))
                    for i in range(real_range):
                        item_value += real_label_data[i]
                        item_price_list = item_price_list + [real_label_data[i]]
                    # 连续数值进行分类
                    p_taraget = [k for k, v in CLASS_SIMPLE_VALUES.items() if (item_value>=v[0] and item_value<v[1])]
                    data_line = data_line + p_taraget
                    data_array.append(data_line)
                    ref_price_array.append(item_price_list)
            data_array = np.array(data_array)
            ref_data_array = np.array(ref_price_array)
            np.save(cache_file,data_array)
            np.save(ref_cache_file,ref_data_array)
        else:
            data_array = np.load(cache_file)
            ref_data_array = np.load(ref_cache_file,allow_pickle=True)
        return data_array,ref_data_array

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
        
        
    def combine_complex_df_data(self,df_ref,date_range,pred_data_path=None,load_cache=False,type="train"):
        """合并预测数据,实际行情数据,价格数据等
           Params:
                df_ref 数据集
                date_range 合并数据日期范围
        """
        
        cache_file = pred_data_path + "/complex_df_cache_{}.pickel".format(type)
        group_column = self.get_group_rank_column()
        (start_time,end_time) = date_range
        # 取得日期列表
        df_range = df_ref[(df_ref["datetime"]>=pd.to_datetime(str(start_time))) & (df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        date_list = df_range["datetime"].dt.strftime('%Y%m%d').unique()
        if not load_cache:
            data_array = []
            data_columns = ["instrument","date"]
            # 预测数据
            pred_columns = ["pred_{}".format(i) for i in range(self.pred_len)]
            # 标签数据）
            label_columns = ["label_{}".format(i) for i in range(self.pred_len)]
            # 实际价格（滑动窗之前的原始数据）
            price_columns = ["price_{}".format(i) for i in range(self.pred_len*2)]
            data_columns = data_columns + pred_columns + label_columns + price_columns
            for date in date_list:
                # 动态取出之前存储的预测数据
                try:
                    pred_series_list = self.get_pred_result(pred_data_path,date)
                    # pred_series_list = self.filter_pred_data_by_mape(pred_series_list,result_id=6)
                    pred_series_list = self.filter_pred_data_by_corr(pred_series_list,result_id=6)
                except Exception as e:
                    print("no data for {}".format(date))
                    continue
                logger.debug('pred_series_list process,{}'.format(date))
                for series in pred_series_list:
                    # 拼接预测数据到每个股票
                    group_rank = series.static_covariates[group_column].values[0]
                    group_item = self.get_group_code_by_rank(group_rank)
                    pred_center_data = get_pred_center_value(series).data
                    time_index_df = df_ref[(df_ref[group_column]==group_rank)&
                                        (df_ref["datetime"]>=date)]
                    if time_index_df.shape[0]==0:
                        continue
                    time_index = time_index_df.iloc[0][self.get_time_column()]                    
                    # 预测数据
                    data_line = [float(group_item),float(date)] + pred_center_data.tolist()
                    # 实际数据部分
                    label_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index)&(df_ref[self.get_time_column()]<time_index+5)]["label"].values.tolist()
                    # 有可能长度不够，补0
                    if len(label_data)<self.pred_len:
                        label_data = label_data + [0.0 for i in range(self.pred_len-len(label_data))]
                    data_line = data_line + label_data
                    # 实际价格部分，往前取相同范围的数据
                    pre_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index-5)&(df_ref[self.get_time_column()]<time_index)]["label_ori"].values.tolist()
                    next_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index)&(df_ref[self.get_time_column()]<time_index+5)]["label_ori"].values.tolist()
                    # 实际数据部分，有可能长度不够，补0
                    if len(next_price_data)<self.pred_len:
                        next_price_data = next_price_data + [0.0 for i in range(self.pred_len-len(next_price_data))]        
                    data_line = data_line + pre_price_data + next_price_data
                    # types = [isinstance(item,float) for item in data_line]
                    # if not np.array(types).all():
                    #     print("not float")
                    data_array.append(data_line)          
                     
            data_array = np.array(data_array)
            target_df = pd.DataFrame(data_array,columns=data_columns)
            target_df["date"] = pd.to_datetime(target_df["date"].astype("int").astype("str"))
            with open(cache_file, "wb") as fout:
                pickle.dump(target_df, fout)              
        else:
            with open(cache_file, "rb") as fin:
                target_df = pickle.load(fin)
        return target_df

    def combine_pred_df_data(self,df_ref,date_range,pred_data_path=None,load_cache=False,type="train"):
        """合并预测数据,实际行情数据,价格数据等
           Params:
                df_ref 数据集
                date_range 合并数据日期范围
        """
        
        cache_file = pred_data_path + "/pred_df_cache_{}.pickel".format(type)
        group_column = self.get_group_rank_column()
        (start_time,end_time) = date_range
        # 取得日期列表
        df_range = df_ref[(df_ref["datetime"]>=pd.to_datetime(str(start_time))) & (df_ref["datetime"]<pd.to_datetime(str(end_time)))]
        date_list = df_range["datetime"].dt.strftime('%Y%m%d').unique()
        if not load_cache:
            data_array = []
            data_columns = ["instrument","date"]
            # 预测数据
            pred_columns = ["pred_{}".format(i) for i in range(self.pred_len)]
            # 标签数据
            label_columns = ["label_{}".format(i) for i in range(self.pred_len)]
            # 实际价格（滑动窗之前的原始数据）
            price_columns = ["price_{}".format(i) for i in range(self.pred_len*2)]
            data_columns = data_columns + pred_columns + label_columns + price_columns
            for date in date_list:
                # 动态取出之前存储的预测数据
                try:
                    pred_series_list = self.get_pred_result(pred_data_path,date)
                    pred_series_list = self.filter_pred_data_by_mape(pred_series_list,result_id=3)
                except Exception as e:
                    print("no data for {}".format(date))
                    continue
                logger.debug('pred_series_list process,{}'.format(date))
                for series in pred_series_list:
                    # 拼接预测数据到每个股票
                    group_rank = series.static_covariates[group_column].values[0]
                    group_item = self.get_group_code_by_rank(group_rank)
                    pred_center_data = get_pred_center_value(series).data
                    time_index_df = df_ref[(df_ref[group_column]==group_rank)&
                                        (df_ref["datetime"]>=date)]
                    if time_index_df.shape[0]==0:
                        continue
                    time_index = time_index_df.iloc[0][self.get_time_column()]                    
                    # 预测数据
                    data_line = [float(group_item),float(date)] + pred_center_data.tolist()
                    # 实际数据部分
                    label_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index)&(df_ref[self.get_time_column()]<time_index+5)]["label"].values.tolist()
                    # 有可能长度不够，补0
                    if len(label_data)<self.pred_len:
                        label_data = label_data + [0.0 for i in range(self.pred_len-len(label_data))]
                    data_line = data_line + label_data
                    # 实际价格部分，往前取相同范围的数据
                    pre_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index-5)&(df_ref[self.get_time_column()]<time_index)]["label_ori"].values.tolist()
                    next_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[self.get_time_column()]>=time_index)&(df_ref[self.get_time_column()]<time_index+5)]["label_ori"].values.tolist()
                    # 实际数据部分，有可能长度不够，补0
                    if len(next_price_data)<self.pred_len:
                        next_price_data = next_price_data + [0.0 for i in range(self.pred_len-len(next_price_data))]        
                    data_line = data_line + pre_price_data + next_price_data
                    # types = [isinstance(item,float) for item in data_line]
                    # if not np.array(types).all():
                    #     print("not float")
                    data_array.append(data_line)          
                     
            data_array = np.array(data_array)
            target_df = pd.DataFrame(data_array,columns=data_columns)
            target_df["date"] = pd.to_datetime(target_df["date"].astype("int").astype("str"))
            with open(cache_file, "wb") as fout:
                pickle.dump(target_df, fout)              
        else:
            with open(cache_file, "rb") as fin:
                target_df = pickle.load(fin)
        return target_df      

        
        
        