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
from tft.class_define import CLASS_VALUES,get_simple_class,get_complex_class

import cus_utils.global_var as global_var
from cus_utils.encoder_cus import transform_slope_value
from .custom_dataset import CustomSequentialDataset
from tushare.stock.indictor import kdj

class TogeSequentialDataset(CustomSequentialDataset):
    """重载CustomSequentialDataset，用于二阶段数据融合"""
    
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

        super().__init__(target_series,past_covariates,future_covariates,input_chunk_length,
                         output_chunk_length,max_samples_per_ts,use_static_covariates,mode=mode)
        
    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ):
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            (scaler,future_past_covariate),
            target_class,
            future_target,
            target_info,
            price_target
        )   = super().__getitem__(idx)
        
        
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            (scaler,future_past_covariate),
            target_class,
            future_target,
            target_info,
            price_target
        )
        


