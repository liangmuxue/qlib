from qlib.data.dataset import DatasetH
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet

import bisect
from typing import Any, Callable, Dict, List, Tuple, Union
from copy import deepcopy
import pandas as pd
import numpy as np

from .data_extractor import StockDataExtractor

class TFTDataset(DatasetH):
    """
    自定义数据集，负责组装底层原始数据，形成DataFrame数据
    """

    DEFAULT_STEP_LEN = 30

    def __init__(self, step_len = DEFAULT_STEP_LEN,**kwargs):
        super().__init__(**kwargs)
        self.step_len = step_len
        self.data_extractor = StockDataExtractor()  

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

    def _prepare_seg(self, slc: slice, **kwargs) -> TimeSeriesDataSet:
        """
        组装TimeSeriesDataSet类型的数据集
        """
        dtype = kwargs.pop("dtype", None)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = self._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None
            
        # 补充辅助数据
        flt_data["month"] = flt_data.date.dt.month.astype("str").astype("category")
        flt_data["log_volume"] = np.log(data.volume + 1e-8)
        
        flt_data["time_idx"] = flt_data["date"].dt.year * 12 + flt_data["date"].dt.month
        flt_data["time_idx"] -= flt_data["time_idx"].min() 
               
        # 每批次的训练长度
        max_encoder_length = self.step_len
        # 每批次的预测长度
        max_prediction_length = self.step_len
        # 训练数据截取
        cut_length= kwargs.pop("cut_length", None)
        training_cutoff = flt_data["time_idx"].max() - cut_length
        # 取得各个因子名称，组装为动态连续变量
        time_varying_unknown_reals = self.handler.infer_processors[0].col_list
        # 商品价格指数，用于静态连续变量
        qyspjg = ["qyspjg_total","qyspjg_yoy","qyspjg_mom"]
        # 动态离散变量，用财务数据
        time_varying_unknown_categoricals = []
        # 获取配置文件参数，生成TimeSeriesDataSet类型的对象
        special_days = []
        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            # 分组字段: 股票代码
            group_ids=["instrument"],
            min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            # 静态固定变量: 股票代码
            static_categoricals=["instrument"],
            # 静态连续变量:每年的股市整体和外部经济环境数据
            static_reals=qyspjg,
            # 动态离散变量:月份
            time_varying_known_categoricals=["month"],
            # 动态已知离散变量: 节假日
            # variable_groups={"special_days": special_days},
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["instrument"], transformation="softplus", center=False
            ),  # use softplus with beta=1.0 and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return training
