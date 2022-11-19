from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from darts import TimeSeries, concatenate

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
        
        # 首先取得pandas原始数据,使用test数据集
        self.data = self.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
      
        # from cus_utils.tensor_viz import TensorViz
        # viz_input = TensorViz(env="data_hist")   
        # v_title = "np_data"
        # viz_input.viz_data_hist(self.data[self.get_target_column()].values,numbins=10,win=v_title,title=v_title)  
        # print("lll")
    
    def get_series_data(self):
        """从pandas数据取得时间序列类型数据"""
        
        group_column = self.col_def["group_column"]
        target_column = self.col_def["target_column"]
        time_column = self.col_def["time_column"]
        past_columns = self.get_past_columns()
        
        data_filter = DataFilter()
        # 清除序列长度不够的股票
        df = data_filter.data_clean(self.data, self.step_len,group_column=group_column,time_column=time_column)
        # 使用后5天的移动平均值作为目标数值
        df[target_column]  = df.groupby(group_column)[target_column].shift(-self.pred_len).rolling(window=self.pred_len,min_periods=1).mean()
        df = df.dropna()      
        # 对目标值进行归一化,标准化
        df[[target_column]] = self.get_scaler().fit_transform(df[[target_column]])        
        # 对协变量值进行归一化,标准化 
        for item in self.get_past_columns():
            if item==self.get_target_column():
                continue
            df[[item]] = self.get_scaler().fit_transform(df[[item]])                 
        # group需要转换为数值型
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')    
        value_cols = self.get_seq_columns()
        value_cols.remove(group_column)  
        # dataframe转timeseries,使用group模式，每个股票生成一个序列,
        series = TimeSeries.from_group_dataframe(df,
                                                time_col=time_column,
                                                 group_cols=group_column,# group_cols会自动成为静态协变量
                                                 freq='D',
                                                 fill_missing_dates=True,
                                                 value_cols=value_cols)          
        series_transformed = []
        future_covariates = []
        past_covariates = []
        static_covariates = []
        series_total = []
        
        for s_transformed in series:
            # 对于测试集再次切分，多出一部分用于协变量查询
            cut_size = s_transformed.pd_dataframe().shape[0] - self.pred_len
            val_ser,_ = s_transformed.split_after(cut_size)        
            # 生成未来协变量       
            future = self.build_future_covariates(s_transformed)     
            future_covariates.append(future)  
            # 生成过去协变量
            past = self.build_past_covariates(s_transformed,past_columns=past_columns) 
            past_covariates.append(past)
            # 删除其他的目标数据，target只保留一个数据 
            ignore_cols = self.get_without_target_columns()
            ignore_cols.remove(group_column)
            val_ser = val_ser.drop_columns(ignore_cols)
            s_transformed = s_transformed.drop_columns(ignore_cols)
            series_total.append(s_transformed)
            series_transformed.append(val_ser)
           
        # 分别返回用于训练预测的序列series_transformed，以及完整序列series
        return series_transformed,past_covariates,future_covariates,static_covariates,series_total

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