from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

import pandas as pd
import numpy as np
from datetime import datetime
from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.data_filter import DataFilter
from numba.core.types import none
from cus_utils.db_accessor import DbAccessor

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
    
    def get_group_code_by_rank(self,group_rank):    
        return self.group_mapping[group_rank]
           
    def create_base_data(self,segments_total=None,val_range=None):
        """创建基础数据"""
        
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
        self.target_scalers = self._create_target_scalers(df_all)       
        
    def build_series_data(self):
        """从pandas数据生成时间序列类型数据"""
        
        total_range = self.segments["train_total"]
        valid_range = self.segments["valid"]
        return self.build_series_data_step_range(total_range, valid_range)
        
    def build_series_data_step_range(self,total_range,val_range,fill_future=False):
        """根据时间点参数，从pandas数据生成时间序列类型数据
           Params:total_range--完整时间序列开始和结束时间
                  val_range--预测时间序列开始和结束时间
        """    
        
        # 生成基础数据
        self.create_base_data(total_range,val_range)

        group_column = self.get_group_rank_column()
        target_column = self.get_target_column()
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        future_columns = self.get_future_columns()
        
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
        
        dt_index_column = self.get_datetime_index_column()
        group_column = self.get_group_rank_column()
        time_column = self.col_def["time_column"]
        target_column = self.get_target_column()
        redun_step = 3
        
        # 首先取得总数据集范围，然后根据这个范围以及当前时间点，动态计算所需要的数据集范围以及验证集范围
        total_df_range = self.segments["train_total"]
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
    
    def filter_pred_data_by_mape(self,pred_list,threhold=10,batch_no=0):
        """根据得分筛选预测数据"""
        
        group_column = self.get_group_rank_column()
        # 如果没有指定批次号，则找到最近一个批次数据
        if batch_no==0:
            batch_no = self.dbaccessor.do_query("select id from pred_result order by id desc limit 1")[0][0]
        # 从之前的预测保存数据中取得小于阈值的股票
        results = self.dbaccessor.do_query("select instrument_rank,instrument,mape from pred_result_detail where result_id=%s and mape<=%s",params=(batch_no,threhold))     
        resutls = np.array(results).astype(np.float)
        filter_list = []
        for pred_item in pred_list:
            group_rank = int(pred_item.static_covariates[group_column].values[0])
            if group_rank in resutls[:,0]:
                filter_list.append(pred_item)
        
        return filter_list
    
    