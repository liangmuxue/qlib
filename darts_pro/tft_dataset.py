from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.utils import init_instance_by_config
from qlib.log import get_module_logger
from typing import Union, List, Tuple, Dict, Text, Optional
from inspect import getfullargspec

from pytorch_forecasting.data.encoders import TorchNormalizer,GroupNormalizer
from tft.timeseries_cus import TimeSeriesCusDataset
from tft.timeseries_crf import TimeSeriesCrfDataset
from tft.timeseries_numpy import TimeSeriesNumpyDataset

import bisect
from typing import Any, Callable, Dict, List, Tuple, Union
from copy import deepcopy
import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from data_extract.data_baseinfo_extractor import StockDataExtractor

class TFTDataset(DatasetH):
    """
    自定义数据集，负责组装底层原始数据，形成DataFrame数据
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, col_def={},step_len=30,pred_len=5,load_dataset_file=False,**kwargs):
        # 基层数据处理器
        self.data_extractor = StockDataExtractor() 
        
        self.col_def = col_def
        self.step_len = step_len
        self.pred_len = pred_len
        self.scaler_type = kwargs['scaler_type']
        self.load_dataset_file = load_dataset_file
        
        # 如果加载数据文件，则不需要进行数据获取了
        if self.load_dataset_file:
            self.segments = kwargs["segments"].copy()
            self.fetch_kwargs = {}
            self.handler: DataHandler = init_instance_by_config(kwargs["handler"], accept_types=DataHandler)
            # self.cus_setup_data(**kwargs)
            return        
        super().__init__(**kwargs)
        
    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        self.cus_setup_data(**kwargs)
    
    def cus_setup_data(self, **kwargs):
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        ###### 用于静态连续变量的数据字典 ######
        # 商品价格指数
        # self.qyspjg_data = self.data_extractor.load_data("qyspjg")
        self.cal = sorted(cal)        
        
    @staticmethod
    def _extend_slice(slc: slice, cal: list, step_len: int) -> slice:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - step_len)
        pad_start = cal[pad_start_idx]
        return slice(pad_start, end)

    def _reindex_inner(self,df):
        """给每个股票顺序编号"""
        
        # 取得时间唯一值并映射为连续编号,生成字典
        time_uni = np.sort(df['time_idx'].unique())
        time_uni_dict = dict(enumerate(time_uni.flatten(), 1))
        time_uni_dict = {v: k for k, v in time_uni_dict.items()}    
        # 使用字典批量替换为连续编号
        df["time_idx"]  = df["time_idx"].map(time_uni_dict)  
        return df   
 
    def prepare(
        self,
        segments: Union[List[Text], Tuple[Text], Text, slice],
        col_set=DataHandler.CS_ALL,
        data_key=DataHandlerLP.DK_I,
        **kwargs,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """重载父类方法"""
    
        # 存储segments类别用于后续判断
        self.segments_mode = segments
        logger = get_module_logger("DatasetH")
        fetch_kwargs = {"col_set": col_set}
        fetch_kwargs.update(kwargs)
        if "data_key" in getfullargspec(self.handler.fetch).args:
            fetch_kwargs["data_key"] = data_key
        else:
            logger.info(f"data_key[{data_key}] is ignored.")

        # Handle all kinds of segments format
        if isinstance(segments, (list, tuple)):
            return [self._prepare_seg(slice(*self.segments[seg]), **fetch_kwargs) for seg in segments]
        elif isinstance(segments, str):
            return self._prepare_seg(slice(*self.segments[segments]), **fetch_kwargs)
        elif isinstance(segments, slice):
            return self._prepare_seg(segments, **fetch_kwargs)
        else:
            raise NotImplementedError(f"This type of input is not supported")   
         
    def _prepare_seg(self, slc: slice, **kwargs) -> pd.DataFrame:
        """
        组装数据,内容加工
        """
        
        # 使用成交量作为标签时的处理
        # return self.prepare_seg_volume(slc)
        
        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)
        # 恢复成一维列索引
        data = data.reset_index() 
        # 重新定义动态数据字段,忽略前面几个无效字段,并手工添加label字段
        self.reals_cols = data.columns.get_level_values(1).values[2:-1]
        self.reals_cols  = np.concatenate((self.reals_cols,['label']))
        # 重建字段名，添加日期和股票字段
        new_cols_idx = np.concatenate((np.array(["datetime","instrument"]),self.reals_cols))
        data.columns = pd.Index(new_cols_idx)    
        # 清除NAN,INF数据(不处理辅助类别数据)
        data["datetime_number"] = data.datetime.dt.strftime('%Y%m%d').astype(int) 
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 价格不能为负数
        data = data[data["label"]>0]
        # 清除极值数据
        # data = self.filter_extremum_data(data, columns=self.col_def["past_covariate_col"])
        # 补充辅助数据,添加日期编号
        data["month"] = data.datetime.astype("str").str.slice(0,7)
        data["time_idx"] = data.datetime.dt.year * 365 + data.datetime.dt.dayofyear
        data["time_idx"] -= data["time_idx"].min() 
        # 重新编号,解决节假日以及相关日期不连续问题
        data = data.groupby("instrument").apply(lambda df: self._reindex_inner(df))        
        # 补充商品指数数据,按照月份合并
        # data = data.merge(self.qyspjg_data,on="month",how="left",indicator=True)
        # month重新编号为1到12
        data["month"] = data["month"].str.slice(5,7).astype(int) - 1
        data["dayofweek"] = data.datetime.dt.dayofweek    
        data["dayofmonth"] = data.datetime.dt.day  
        # 保留时间戳
        data["datetime_number"] = data.datetime.dt.strftime('%Y%m%d').astype(int)     
        data["label"] = data["label"].astype("float64")
        # 使用前几天的移动平均值作为目标数值
        data["label_ori"] = data["label"]
        data["price_norm"] = data[["label_ori","instrument"]].groupby("instrument").transform(lambda x: ((x-x.min())/(x.max()-x.min())+1e-5)) 
        # Mock
        data["diff_range"] = data["price_norm"] 
        data["rsv_diff"] = data["price_norm"] 
        data["qtlu_diff"] = data["price_norm"] 
        data["cci_diff"] = data["price_norm"] 
        
        group_column = self.get_group_column()
        data["label"] = data.groupby(group_column)["label"].rolling(window=self.pred_len,min_periods=1).mean().reset_index(0,drop=True)
        # 生成KDJ指标
        self.compute_kdj(data)    
        self.compute_atr(data)     
        # 使用指定字段
        columns = self.get_seq_columns() + ["label","datetime_number"]
        columns = list(set(columns))
        data = data[columns]
        # 目标字段值转换为64位
        for col in self.get_target_column_exc_ext():
            data[col] = data[col].astype(np.float64)
        return data

    def compute_kdj(self,df):
        """KDJ指标计算"""
        
        window_size = 9
        low_list=df['LOW'].rolling(window=window_size).min()
        low_list.fillna(value=df['LOW'].expanding().min(), inplace=True)
        high_list = df['HIGH'].rolling(window=window_size).max()
        high_list.fillna(value=df['HIGH'].expanding().max(), inplace=True)
        rsv = (df['CLOSE'] - low_list) / (high_list - low_list) * 100
        df['KDJ_K'] = rsv.ewm(com=2,adjust=False).mean()  
        df['KDJ_D'] = df['KDJ_K'].ewm(com=2,adjust=False).mean()
        df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']
    
    def compute_atr(self,df):
        """平均真实范围指标计算"""
        
        def wwma(values, n):
            """
             J. Welles Wilder's EMA 
            """
            return values.ewm(alpha=1/n, adjust=False).mean()
        
        def atr(df, n=5):
            data = df.copy()
            high = data['HIGH']
            low = data['LOW']
            close = data['CLOSE']
            data['tr0'] = abs(high - low)
            data['tr1'] = abs(high - close.shift())
            data['tr2'] = abs(low - close.shift())
            tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
            atr = wwma(tr, n)
            return atr
        
        df['ATR5'] = atr(df)
           
    def filter_extremum_data(self,data,columns=[]):
        """清除极值数据"""
        
        for col in columns:
            data = data[data[col].abs()>=0.1**6]  
        return data

    def get_seq_columns(self):
        """取得所有数据字段名称"""
        
        time_column = self.col_def["time_column"]
        col_list = self.col_def["col_list"]
        group_column = self.col_def["group_column"]
        target_column = self.get_target_column_exc_ext()
        future_covariate_col = self.col_def["future_covariate_col"]
        columns = [time_column] + [group_column] + target_column + col_list + future_covariate_col
        # 去重
        columns = list(set(columns))
        return columns  

    def get_without_target_columns(self):
        """取得不包括目标字段的其他数据字段名称"""
        
        time_column = self.col_def["time_column"]
        col_list = self.col_def["col_list"]
        group_column = self.col_def["group_column"]
        future_covariate_col = self.col_def["future_covariate_col"]
        columns = [time_column] + [group_column] + col_list + future_covariate_col
        return columns  

    def get_time_column(self):
        """取得日期字段名称"""
        time_column = self.col_def["time_column"]    
        return time_column
 
    def get_datetime_index_column(self):
        """取得日期索引名称"""
        time_column = self.col_def["datetime_index_column"]    
        return time_column
           
    def get_past_columns(self):
        """取得过去协变量字段名称"""
        past_covariate_col = self.col_def["past_covariate_col"] 
        return past_covariate_col  

    def get_group_column(self):
        """取得分组字段"""
        return self.col_def["group_column"]   

    def get_group_rank_column(self):
        """取得分组rank字段"""
        return self.col_def["group_rank_column"]   
        
    def get_target_column(self):
        """取得目标字段"""
        return self.col_def["target_column"]       

    def get_target_column_exc_ext(self):
        """取得目标字段,排除扩展字段"""
        
        target_column = self.col_def["target_column"]  
        ext_column = self.col_def["ext_column"]     
        # 排除扩展字段
        target_column = list(set(target_column) - set(ext_column))        
        return target_column 
    
    def get_future_columns(self):
        """取得未来协变量字段名"""
        feature_columns = self.col_def["future_covariate_col"]    
        return feature_columns
    
    def get_future_scale_columns(self):
        feature_columns = self.col_def["future_covariate_col"]
        scale_columns = [col+"_scale" for col in feature_columns]
        return scale_columns
        
    def get_static_columns(self):
        """取得静态协变量字段名"""
        static_columns = self.col_def["static_covariate_col"] 
        return static_columns
    
    def get_static_scale_columns(self):
        static_columns = self.col_def["static_covariate_col"]    
        scale_columns = [col+"_scale" for col in static_columns]
        # 还需要添加索引对应的归一化静态协变量
        group_col = self.get_group_rank_column()
        group_col_scale = group_col + "_scale"  
        scale_columns += [group_col_scale]        
        return scale_columns
              
    def get_group_column_index(self):
        """取得分组变量段对应下标"""
        columns = self.get_seq_columns()
        group_column = self.col_def["group_column"]
        return columns.index(group_column)
            
    def get_past_column_index(self):
        """取得过去协变量段对应下标"""
        columns = self.get_seq_columns()
        past_columns = self.get_past_columns()
        return [columns.index(column) for column in past_columns]
    
    def get_past_standard_column_index(self):
        """取得过去协变量段对应下标"""
        columns = self.get_seq_columns()
        past_columns = self.col_def["past_covariate_standard_col"]
        return [columns.index(column) for column in past_columns]    
               
    def get_target_column_index(self):
        """取得目标字段对应下标"""
        columns = self.get_seq_columns()
        target_column = self.col_def["target_column"]    
        return columns.index(target_column)

    def get_time_column_index(self):
        """取得日期字段对应下标"""
        columns = self.get_seq_columns()
        target_column = self.col_def["time_column"]    
        return columns.index(target_column)
    
    def get_future_column_index(self):
        """取得特征字段对应下标"""
        columns = self.get_seq_columns()
        feature_columns = self.col_def["future_covariate_col"]    
        return [columns.index(column) for column in feature_columns]
 
    def get_future_and_target_column_index(self):
        """取得目标字段及特征字段对应下标"""
        columns_index = self.get_future_column_index() + [self.get_target_column_index()]
        return columns_index       
    
    def get_scaler(self):
        if self.scaler_type == "norm":
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()    
        return scaler      

    def view_datatime(self,training_data,val_data):
        t_time = training_data[:,:,-1]
        v_time = val_data[:,:,-1]
        print("training time max:{},val time min:{}".format(t_time.max(), v_time.min()))    
    
    