from darts.utils.data.sequential_dataset import (
    SplitCovariatesTrainingDataset,
)

from typing import Optional, List, Tuple, Union
import numpy as np
from numba.core.types import none
import torch
from typing import Optional, Sequence, Tuple, Union
from sklearn.preprocessing import MinMaxScaler

from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset,DualCovariatesSequentialDataset
from darts.utils.data.inference_dataset import InferenceDataset,PastCovariatesInferenceDataset,DualCovariatesInferenceDataset
from darts.utils.data.shifted_dataset import GenericShiftedDataset,MixedCovariatesTrainingDataset
from darts.utils.data.utils import CovariateType
from darts.logging import raise_if_not
from darts import TimeSeries
from cus_utils.common_compute import normalization,slope_last_classify_compute
from tft.class_define import CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUES,SLOPE_SHAPE_FALL,SLOPE_SHAPE_RAISE,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH,get_simple_class

import cus_utils.global_var as global_var

class CusGenericShiftedDataset(GenericShiftedDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        shift_covariates: bool = False,
        max_samples_per_ts: Optional[int] = None,
        covariate_type: CovariateType = CovariateType.NONE,
        use_static_covariates: bool = True,
    ):
        """
        自定义数据集，用于重载父类getitem方法
        """
        super().__init__(target_series,covariates,input_chunk_length,output_chunk_length,shift,shift_covariates,
                         max_samples_per_ts,covariate_type,use_static_covariates)
        
        df_all = global_var.get_value("dataset").df_all
        self.ass_data = {}
        for series in self.target_series:
            code = int(series.static_covariates["instrument_rank"].values[0])
            price_array = df_all[(df_all["time_idx"]>=series.time_index.start)&(df_all["time_idx"]<series.time_index.stop)
                                &(df_all["instrument_rank"]==code)]["label_ori"].values
            self.ass_data[code] = price_array
            
    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """重载父类方法，用于植入更多数据"""
        
        # determine the index of the time series.
        target_idx = idx // self.max_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(target_vals) - self.size_of_both_chunks + 1
        
        raise_if_not(
            n_samples_in_ts >= 1,
            "The dataset contains some time series that are too short to contain "
            "`max(self.input_chunk_length, self.shift + self.output_chunk_length)` "
            "({}-th series)".format(target_idx),
        )

        # determine the index at the end of the output chunk
        # it is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (
            len(target_series)
            - (idx - (target_idx * self.max_samples_per_ts)) % n_samples_in_ts
        )

        # optionally, load covariates
        covariate_series = (
            self.covariates[target_idx] if self.covariates is not None else None
        )

        main_covariate_type = CovariateType.NONE
        if self.covariates is not None:
            main_covariate_type = (
                CovariateType.FUTURE if self.shift_covariates else CovariateType.PAST
            )

        # get all indices for the current sample
        (
            past_start,
            past_end,
            future_start,
            future_end,
            covariate_start,
            covariate_end,
        ) = self._memory_indexer(
            target_idx=target_idx,
            target_series=target_series,
            shift=self.shift,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            covariate_series=covariate_series,
            covariate_type=main_covariate_type,
        )

        # extract sample target
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]
        # 返回目标信息，用于后续调试,包括目标值，当前索引，总条目等
        code = int(target_series.static_covariates["instrument_rank"].values[0])
        price_array = self.ass_data[code][future_start-1:future_end]
        total_price_array = self.ass_data[code][past_start:future_end]
        target_info = {"item_rank_code":code,"start":target_series.time_index[past_start],
                       "end":target_series.time_index[future_end-1]+1,"past_start":past_start,"past_end":past_end,
                       "future_start":future_start,"future_end":future_end,"price_array":price_array,"total_price_array":total_price_array,
                       "total_start":target_series.time_index.start,"total_end":target_series.time_index.stop}

        # optionally, extract sample covariates
        covariate = None
        if self.covariates is not None:
            raise_if_not(
                covariate_end <= len(covariate_series),
                f"The dataset contains {main_covariate_type.value} covariates "
                f"that don't extend far enough into the future. ({idx}-th sample)",
            )

            covariate_total = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end + self.output_chunk_length
            ]
            covariate = covariate_total[:self.input_chunk_length]
            future_covariate = covariate_total[self.input_chunk_length:]
            raise_if_not(
                len(covariate)
                == (
                    self.output_chunk_length
                    if self.shift_covariates
                    else self.input_chunk_length
                ),
                f"The dataset contains {main_covariate_type.value} covariates "
                f"whose time axis doesn't allow to obtain the input (or output) chunk relative to the "
                f"target series.",
            )

        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return past_target, covariate, static_covariate, future_target,target_info,future_covariate
        
class CustomSequentialDataset(MixedCovariatesTrainingDataset):
    """重载MixedCovariatesSequentialDataset，用于定制加工数据"""
    
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        mode="train",
    ):
        """初始化，分为过去数据集和未来数据集"""

        super().__init__()
        self.mode = mode
        # This dataset is in charge of serving past covariates
        self.ds_past = CusGenericShiftedDataset(
            target_series=target_series,
            covariates=past_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=input_chunk_length,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving historical and future future covariates
        self.ds_dual = DualCovariatesSequentialDataset(
            target_series=target_series,
            covariates=future_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
        )
        
        self.output_chunk_length = output_chunk_length
        self.class1_len = int((output_chunk_length-1)/2)
        self.class2_len = output_chunk_length -1 - self.class1_len
        
    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:

        past_target, past_covariate, static_covariate, future_target,target_info,future_past_covariate = self.ds_past[idx]
        
        _, historic_future_covariate, future_covariate, _, _ = self.ds_dual[idx]
        
        # 使用原价格作为涨跌幅分类参照
        # df_all = global_var.get_value("dataset").df_all
        # price_items = df_all[(df_all["time_idx"]>=target_info["future_start"])&(df_all["time_idx"]<target_info["future_end"])&
        #                         (df_all["instrument_rank"]==target_info["item_rank_code"])]["label_ori"].values
        # price_items = target_info["price_array"]
        # price_items = np.expand_dims(price_items,axis=-1)
        # 添加总体走势分类输出,使用原值比较最大上涨幅度与最大下跌幅度，从而决定幅度范围正还是负
        price_tar = future_target
        # price_tar = price_items
        max_value = np.max(price_tar)
        min_value = np.min(price_tar)
        if max_value - price_tar[0,0] > price_tar[0,0] - min_value:
            raise_range = (max_value - price_tar[0,0])/price_tar[0,0]*100
        else:
            raise_range = (min_value - price_tar[0,0])/price_tar[0,0]*100
            
        # 先计算涨跌幅度分类，再进行归一化
        p_taraget_class = get_simple_class(raise_range)
          
        # 针对目标数据，进行单独归一化，扩展数据波动范围
        scaler = MinMaxScaler(feature_range=(0.01,1))
        scaler.fit(past_target)             
        past_target = scaler.transform(past_target)   
        future_target = scaler.transform(future_target)   
                
        # 对于协变量，也进行单独的归一化    
        past_covariate = normalization(past_covariate)
        future_covariate = normalization(future_covariate)
               
        # 添加末段走势分类目标输出
        target_class = slope_last_classify_compute(future_target)
        target_class = np.expand_dims([target_class],axis=-1)
        p_taraget_class = np.expand_dims(np.array([p_taraget_class]),axis=-1)  
        target_class = np.concatenate((target_class,p_taraget_class),axis=0)
        
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            (scaler,future_past_covariate),
            target_class,
            future_target,
            target_info
        )
        
class CustomInferenceDataset(InferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        use_static_covariates: bool = True,
    ):
        """
       
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(
            target_series=target_series,
            covariates=past_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving historic and future future covariates
        self.ds_future = DualCovariatesInferenceDataset(
            target_series=target_series,
            covariates=future_covariates,
            n=n,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            use_static_covariates=use_static_covariates,
        )   
        
    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        MinMaxScaler,
        TimeSeries,
    ]:

        (
            past_target,
            past_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
        ) = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _, _ = self.ds_future[idx]
        
        # 针对价格数据，进行单独归一化，扩展数据波动范围
        scaler = MinMaxScaler(feature_range=(0.01,1))
        past_target = scaler.fit_transform(past_target)   
        
        # 对于协变量，也进行单独的归一化    
        past_covariate = normalization(past_covariate)
        future_covariate = normalization(future_covariate)  
            
        # 需要返回scaler，用于后续恢复原数据
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            scaler,
            ts_target
        )         