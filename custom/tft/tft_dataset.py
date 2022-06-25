from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.log import get_module_logger
from typing import Union, List, Tuple, Dict, Text, Optional
from inspect import getfullargspec

from pytorch_forecasting.data.encoders import TorchNormalizer,GroupNormalizer
from tft.timeseries_cus import TimeSeriesCusDataset

import bisect
from typing import Any, Callable, Dict, List, Tuple, Union
from copy import deepcopy
import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset

from data_extract.data_baseinfo_extractor import StockDataExtractor

class TFTDataset(DatasetH):
    """
    自定义数据集，负责组装底层原始数据，形成DataFrame数据
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len = DEFAULT_STEP_LEN,pred_len = 5,**kwargs):
        # 基层数据处理器
        self.data_extractor = StockDataExtractor() 
        super().__init__(**kwargs)
        self.step_len = step_len
        self.pred_len = pred_len
        self.viz = kwargs["viz"]
         
    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        
        ###### 用于静态连续变量的数据字典 ######
        
        # 商品价格指数
        self.qyspjg_data = self.data_extractor.load_data("qyspjg")
                
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
        # 重新定义动态数据字段,忽略前面几个无效字段,注意不需要手工添加label字段了
        self.reals_cols = data.columns.get_level_values(1).values[3:-1]
        # 重建字段名，添加日期和股票字段,以及label字段
        new_cols_idx = np.concatenate((np.array(["datetime","instrument","value_validate"]),self.reals_cols,['label']))
        data.columns = pd.Index(new_cols_idx)    
        # 清除NAN数据
        data = data.dropna() 
        # 删除价格小于0的数据
        # data = data[data.label>0]        
        # 删除涨跌幅度大于20%的数据 
        data = data[data.label.abs()<0.2]  
        # 正数转换
        data['label'] = data['label'] + 0.2 
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
        # datetime转为字符串
        data["dayofweek"] = data.datetime.dt.dayofweek.astype("str").astype("category")        
        # 删除校验列
        data.drop(columns=['value_validate'],inplace=True)
        # data['instrument'].value_counts().to_pickle("/home/qdata/qlib_data/test/instrument_{}.pkl".format(self.segments_mode))
        return data
 
    def prepare_seg_volume(self, slc: slice, **kwargs):
        """处理成交量模式的数据"""
        
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
        # 重新定义动态数据字段,忽略前面几个无效字段,注意不需要手工添加label字段了
        self.reals_cols = data.columns.values[3:-1]
        # 重建字段名，添加日期和股票字段,以及label字段
        new_cols_idx = np.concatenate((np.array(["datetime","instrument","value_validate"]),self.reals_cols,['label']))
        data.columns = pd.Index(new_cols_idx)   
        # 缩小取值空间 
        # data["label"] = data["label"] / 100
        # 清除NAN数据
        data = data.dropna() 
        data["month"] = data.datetime.astype("str").str.slice(0,7)
        data["time_idx"] = data.datetime.dt.year * 365 + data.datetime.dt.dayofyear
        data["time_idx"] -= data["time_idx"].min() 
        # 重新编号,解决节假日以及相关日期不连续问题
        data = data.groupby("instrument").apply(lambda df: self._reindex_inner(df))        
        # 补充商品指数数据,按照月份合并
        data = data.merge(self.qyspjg_data,on="month",how="left",indicator=True)
        # month重新编号为1到12
        data["month"] = data["month"].str.slice(5,7)
        # datetime转为字符串
        data["dayofweek"] = data.datetime.dt.dayofweek.astype("str").astype("category")   
        # 删除校验列
        data.drop(columns=['value_validate'],inplace=True)
        # data['instrument'].value_counts().to_pickle("/home/qdata/qlib_data/test/instrument_{}.pkl".format(self.segments_mode))
        return data
            
    def get_ts_dataset(self,data,mode="train",train_ts=None):
        """
        取得TimeSeriesDataSet对象

        Parameters
        ----------
        data : DataFrame 已经生成的panda数据
        """     
        
        # return self.get_ts_dataset_test(data,mode=mode)
        # 每批次的训练长度
        max_encoder_length = self.step_len
        # 每批次的预测长度
        max_prediction_length = self.pred_len
        # 取得各个因子名称，组装为动态连续变量
        time_varying_unknown_reals = ["label"] # self.reals_cols.tolist()
        # 商品价格指数，用于静态连续变量
        qyspjg = ["qyspjg_total","qyspjg_yoy","qyspjg_mom"]
        qyspjg = []
        # 动态离散变量，可以使用财务数据
        time_varying_unknown_categoricals = []
        # 获取配置文件参数，生成TimeSeriesDataSet类型的对象
        special_days = []
        tsdata = TimeSeriesCusDataset(
            data,
            time_idx="time_idx",
            target="label",
            # 分组字段: 股票代码
            group_ids=["instrument"],
            min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            # 静态固定变量: 股票代码
            static_categoricals=["instrument"], # ["instrument"],
            # 静态连续变量:每年的股市整体和外部经济环境数据
            static_reals=qyspjg,
            # 动态离散变量:日期
            time_varying_known_categoricals=["dayofweek"],
            # 动态已知离散变量: 节假日
            # variable_groups={"special_days": special_days},
            time_varying_known_reals=[], #["time_idx"],
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=["instrument"],transformation="softplus", center=False),
            add_relative_time_idx=False,
            add_target_scales=False,
            add_encoder_length=True,
            viz=self.viz,
        )
        if mode=="valid":
            tsdata = TimeSeriesCusDataset.from_dataset(tsdata, data, predict=True, stop_randomization=True)
        return tsdata     
    
    
    def get_ts_dataset_test(self,data_ori,mode="train",train_ts=None):
        from pytorch_forecasting.data.examples import get_stallion_data
        data = get_stallion_data()
        
        data["month"] = data.date.dt.month.astype("str").astype("category")
        data["log_volume"] = np.log(data.volume + 1e-8)
        
        data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
        data["time_idx"] -= data["time_idx"].min()
        data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
        data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")       
         
        training_cutoff = data["time_idx"].max() - 6
        max_encoder_length = 18
        max_prediction_length = 6        
        tsdata = TimeSeriesCusDataset(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals= ["agency", "sku"],
            static_reals=[],#["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["month"],
            # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
            time_varying_known_reals= [], #["time_idx", "price_regular", "discount_in_percent"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
            ],
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            ),  # use softplus with beta=1.0 and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )       
        if mode=="valid":
            tsdata = TimeSeriesCusDataset.from_dataset(tsdata, data, predict=True, stop_randomization=True)
        return tsdata     
    
    
    