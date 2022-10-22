from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.log import get_module_logger
from typing import Union, List, Tuple, Dict, Text, Optional
from inspect import getfullargspec

from pytorch_forecasting.data.encoders import TorchNormalizer,GroupNormalizer
from tft.timeseries_cus import TimeSeriesCusDataset
from tft.timeseries_crf import TimeSeriesCrfDataset
from tft.timeseries_numpy import TimeSeriesNumpyDataset

import bisect
import pandas as pd
import numpy as np

from data_extract.data_baseinfo_extractor import StockDataExtractor
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset

class TFTDataset(DatasetH):
    """
    自定义数据集，负责组装底层原始数据，形成DataFrame数据
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len = DEFAULT_STEP_LEN,pred_len = 5,data_path=None,
                 columns=[],future_covariate_col=[],past_covariate_col=[],static_covariate_col=[],**kwargs):
        # 基层数据处理器
        self.data_extractor = StockDataExtractor() 
        super().__init__(**kwargs)
        self.step_len = step_len
        self.pred_len = pred_len
        self.columns = columns
        self.future_covariate_col = future_covariate_col
        self.past_covariate_col = past_covariate_col
        self.static_covariate_col = static_covariate_col     
        self.viz = kwargs["viz"]
        
        self.data = np.load(data_path)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        
        ###### 用于静态连续变量的数据字典 ######
        
        # 商品价格指数
        self.qyspjg_data = self.data_extractor.load_data("qyspjg")
                
        self.cal = sorted(cal)
   
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
        
        dtype = kwargs.pop("dtype", None)
        start, end = slc.start, slc.stop

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)
        # 如果是测试阶段，则需要删除日期不在训练集中的股票
        if self.segments_mode=="test":
            ext_slice_train = self._extend_slice(slice(*self.segments["train"]), self.cal, self.step_len)
            data_train = super()._prepare_seg(ext_slice_train, **kwargs)
            data = data[data.index.get_level_values(1).isin(data_train.index.get_level_values(1).values)]
        # 恢复成一维列索引
        data = data.reset_index() 
        # 重新定义动态数据字段,忽略前面几个无效字段,并手工添加label字段
        self.reals_cols = data.columns.get_level_values(1).values[2:-1]
        self.reals_cols  = np.concatenate((self.reals_cols,['label']))
        # 重建字段名，添加日期和股票字段
        new_cols_idx = np.concatenate((np.array(["datetime","instrument"]),self.reals_cols))
        data.columns = pd.Index(new_cols_idx)    
        # 清除NAN数据
        data = data.dropna() 
        # 删除涨跌幅度过大的数据 
        data = data[data.label.abs()<=0.2]  
        # 增大取值范围
        data['label'] = data['label'] * 100
        # 补充辅助数据,添加日期编号
        data["month"] = data.datetime.astype("str").str.slice(0,7)
        data["time_idx"] = data.datetime.dt.year * 365 + data.datetime.dt.dayofyear
        data["time_idx"] -= data["time_idx"].min() 
        # 重新编号,解决节假日以及相关日期不连续问题
        data = data.groupby("instrument").apply(lambda df: self._reindex_inner(df))        
        # 补充商品指数数据,按照月份合并
        data = data.merge(self.qyspjg_data,on="month",how="left",indicator=True)
        # month重新编号为1到12
        data["month"] = data["month"].str.slice(5,7)
        data["dayofweek"] = data.datetime.dt.dayofweek.astype("str").astype("category")    
        # 添加数字类型的时间戳
        data["datetime_number"] = data.datetime.astype("int")
        # data['instrument'].value_counts().to_pickle("/home/qdata/qlib_data/test/instrument_{}.pkl".format(self.segments_mode))
        return data_
    
    def get_custom_numpy_dataset(self,numpy_data):
        """
        直接使用numpy数据取得DataSet对象

        Parameters
        ----------
        numpy_data : numpy数据
        """     
        
        future_covariate_index = [self.columns.index(item) for item in self.future_covariate_col]
        past_covariate_index = [self.columns.index(item) for item in self.past_covariate_col]
        static_covariate_index = [self.columns.index(item) for item in self.static_covariate_col]
        
        return CustomNumpyDataset(
            numpy_data,
            self.input_chunk_length,
            self.output_chunk_length,
            future_covariate_index,
            past_covariate_index,
            static_covariate_index,            
        ) 

        
    