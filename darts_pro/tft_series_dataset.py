from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

import pandas as pd

from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
from darts_pro.tft_dataset import TFTDataset
from cus_utils.data_filter import DataFilter

class TFTSeriesDataset(TFTDataset):
    """
    自定义数据集，使用darts封装的TimeSeries类型数据
    """

    def __init__(self, step_len = 30,pred_len = 5,low_threhold=-5,high_threhold=5,over_time=2,aug_type="no",model_type="lstm",col_def={},**kwargs):
        # 基层数据处理器
        self.data_extractor = StockDataExtractor() 
        super().__init__(col_def=col_def,step_len=step_len,pred_len=pred_len,**kwargs)
        
        self.future_covariate_col = col_def['future_covariate_col']
        self.past_covariate_col = col_def['past_covariate_col']
        self.static_covariate_col = col_def['static_covariate_col']    
        self.low_threhold = low_threhold
        self.high_threhold = high_threhold
        self.over_time = over_time        
        self.aug_type = aug_type
        self.model_type = model_type
        self.columns = self.get_seq_columns()
        
        # 首先取得pandas原始数据,使用train,valid数据集
        df_all = self.prepare("train_total", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 前处理
        self.df_all = self._pre_process_df(df_all)

    def get_emb_size(self):
        group_column = self.get_group_rank_column()
        return self.df_all[group_column].unique().shape[0] + 1
            
    def _pre_process_df(self,df):
        data_filter = DataFilter()
        # 清除序列长度不够的股票
        group_column = self.get_group_column()
        valid_range = self.segments["valid"]
        valid_start = int(valid_range[0].strftime('%Y%m%d'))
        time_column = self.col_def["time_column"]       
        df = data_filter.data_clean(df, self.step_len,valid_start=valid_start,group_column=group_column,time_column=time_column)        
        # 生成时间字段
        df['datetime'] = pd.to_datetime(df['datetime_number'].astype(str))
        # 使用后几天的移动平均值作为目标数值
        df["label"]  = df.groupby(self.get_group_column())[self.get_target_column()].shift(self.pred_len).rolling(window=5,min_periods=1).mean()
        df = df.dropna()       
        # group字段需要转换为数值型
        group_column = self.get_group_column()
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')   
        # 新增排序字段，用于后续embedding
        rank_group_column = self.get_group_rank_column()
        df[rank_group_column] = df[group_column].rank(method='dense',ascending=False).astype("int")  
        return df    
        
    def get_series_data(self,type="train"):
        """从pandas数据取得时间序列类型数据"""
        
        group_column = self.get_group_rank_column()
        target_column = self.get_target_column()
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        future_columns = self.get_future_columns()
        
        # train模式表示进行训练，使用全集和验证集，test模式表示进行测试，使用全集和测试集
        if type=="train":
            valid_range = self.segments["valid"]
        else:
            valid_range = self.segments["test"]
        valid_start = int(valid_range[0].strftime('%Y%m%d'))
        
        df_all = self.df_all
        df_train = df_all[df_all["datetime"]<pd.to_datetime(str(valid_start))]
        df_val = df_all[df_all["datetime"]>=pd.to_datetime(str(valid_start))]
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
            target_scaler = Scaler()
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

        # 分别返回用于训练预测的序列series_transformed，以及完整序列series
        return train_series_transformed,val_series_transformed,past_convariates,future_convariates

    def get_pred_series_data(self):
        """从pandas数据取得时间序列类型数据,用于预测过程"""
        
        # 首先取得pandas原始数据,使用test数据集
        df_test = self.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 前处理
        df_test = self._pre_process_df(df_test)        
        group_column = self.get_group_rank_column()
        target_column = self.get_target_column()
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        future_columns = self.get_future_columns()
    
        # 分别生成训练和测试序列数据
        pred_series = TimeSeries.from_group_dataframe(df_test,
                                                time_col=time_column,
                                                 group_cols=group_column,# 会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 value_cols=target_column)    
 
        series_transformed = []
        val_series_transformed = []
        
        # 生成归一化的目标序列
        for index,ts in enumerate(pred_series):
            target_scaler = Scaler()
            s_transformed = target_scaler.fit_transform(ts)
            series_transformed.append(s_transformed)
            # 去掉预测长度部分数据作为预测数据，保证数据对齐
            cut_size = s_transformed.time_index.stop - self.pred_len - 1
            val_transformed,_ = s_transformed.split_after(cut_size)             
            val_series_transformed.append(val_transformed)
        
        def build_covariates(column_names):
            covariates_array = []
            for series in series_transformed:
                group_col_val = series.static_covariates[group_column].values[0]
                scaler = Scaler()
                # 遍历并筛选出不同分组字段(股票)的单个dataframe
                df_item = df_test[df_test[group_column]==group_col_val]
                covariates = TimeSeries.from_dataframe(df_item,time_col=time_column,
                                                         freq='D',
                                                         fill_missing_dates=True,
                                                         value_cols=column_names)     
                covariates_transformed = scaler.fit_transform(covariates)    
                covariates_array.append(covariates_transformed)
            return covariates_array            

        # 生成过去协变量，并归一化
        past_convariates = build_covariates(past_columns)      
        # 生成未来协变量，并归一化
        future_convariates = build_covariates(future_columns)                   

        # 分别返回用于训练预测的序列series_transformed，以及相关协变量
        return series_transformed,val_series_transformed,past_convariates,future_convariates
    
    def normolize(self,data,training_range=None,val_range=None):
        """对数据进行归一化"""
        
        train_start,train_end = training_range
        valid_start,valid_end = val_range
        training_data = data[(data[:,-1,-1]> train_start) & (data[:,-1,-1]<train_end)]
        val_data = data[(data[:,0,-1]> valid_start) & (data[:,0,-1]<valid_end)]
        
        # 目标数据归一化
        target_scaler = self.get_scaler()
        target_index = self.get_target_column_index()
        training_data[:,:,target_index] = target_scaler.fit_transform(training_data[:,:,target_index])   
        val_data[:,:,target_index] = target_scaler.transform(val_data[:,:,target_index]) 
                
        def transfer_data(column_index):
            scaler = self.get_scaler()
            scaler.fit(training_data[:,:,column_index])      
            t_data = scaler.transform(training_data[:,:,column_index])     
            v_data = scaler.transform(val_data[:,:,column_index]) 
            return  t_data, v_data    
        
        # 对过去协变量值进行标准化(归一化)
        for index in self.get_past_column_index():
            training_data[:,:,index],val_data[:,:,index] = transfer_data(index)

        # 对未来协变量值进行标准化(归一化)
        for index in self.get_future_column_index():
            training_data[:,:,index],val_data[:,:,index] = transfer_data(index)
                        
        return training_data,val_data
    
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