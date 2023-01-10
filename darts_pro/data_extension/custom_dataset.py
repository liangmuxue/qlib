from darts.utils.data.sequential_dataset import (
    SplitCovariatesTrainingDataset,
)

from typing import Optional, List, Tuple, Union
import numpy as np
from numba.core.types import none

from typing import Optional, Sequence, Tuple, Union
from sklearn.preprocessing import MinMaxScaler

from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset,DualCovariatesSequentialDataset
from darts.utils.data.inference_dataset import InferenceDataset,PastCovariatesInferenceDataset,DualCovariatesInferenceDataset
from darts.utils.data.shifted_dataset import GenericShiftedDataset,MixedCovariatesTrainingDataset
from darts.utils.data.utils import CovariateType
from darts import TimeSeries

class CustomNumpyDataset(SplitCovariatesTrainingDataset):
    def __init__(
        self,
        numpy_data=None,
        input_chunk_length: int = 25,
        output_chunk_length: int = 5,
        future_covariate_index: List = [],
        past_covariate_index: List = [],
        static_covariate_index: List = [],
        target_index: str = None,
        model_type = "tft",
    ):
        """
        自定义数据集，直接使用numpy数据进行取值
        """
        super().__init__()
        
        self.model_type = model_type
        self.numpy_data = numpy_data
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.future_covariate_index = future_covariate_index
        self.past_covariate_index = past_covariate_index
        self.static_covariate_index = static_covariate_index
        self.target_index = target_index

    def __len__(self):
        return self.numpy_data.shape[0]

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        row = self.numpy_data[idx]
        past_target = row[:self.input_chunk_length,self.target_index:self.target_index+1]
        past_covariate = row[:self.input_chunk_length,self.past_covariate_index]
        future_covariate = row[self.input_chunk_length:,self.future_covariate_index]
        historic_future_covariate = row[:self.input_chunk_length,self.future_covariate_index]
        future_target = row[self.input_chunk_length:,self.target_index:self.target_index+1]
        static_covariate = np.expand_dims(row[0,self.static_covariate_index],axis=0)
        
        if self.model_type=="tft":
            return (
                past_target,
                past_covariate,
                historic_future_covariate,
                future_covariate,
                static_covariate,
                future_target,
            )
        
        if self.model_type=="lstm":
            past_target = row[:self.input_chunk_length,self.target_index:self.target_index+1]
            past_covariate = row[:self.input_chunk_length,self.past_covariate_index]
            if len(self.future_covariate_index)==0:
                future_covariate = None
                historic_future_covariate = None
            else:
                future_covariate = row[self.output_chunk_length:,self.future_covariate_index]
                historic_future_covariate = row[self.output_chunk_length:,self.future_covariate_index]
            # lstm模式下，shift一直为1，因此使用滚动数据
            future_target = row[self.output_chunk_length:,self.target_index:self.target_index+1]
            static_covariate = np.expand_dims(row[0,self.static_covariate_index],axis=0)           
            return (
                past_target,
                past_covariate,
                future_covariate,
                None,
                future_target
            )            
            
        if self.model_type=="tcn":
            return (
                past_target,
                past_covariate,
                None,
                future_target.transpose(1,0)
            )             
            
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
        mode="train"
    ):
        """"""

        super().__init__()
        self.mode = mode
        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(
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
    ]:

        past_target, past_covariate, static_covariate, future_target = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _, _ = self.ds_dual[idx]
        
        # 针对价格数据，进行单独归一化，扩展数据波动范围
        scaler = MinMaxScaler()
        if self.mode=="train":
            # scaler.fit(np.concatenate((past_target,future_target),axis=0))   
            scaler.fit(past_target)
        else:
            scaler.fit(past_target)
        past_target = scaler.transform(past_target)   
        future_target = scaler.transform(future_target)    
        
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            future_target,
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
        scaler = MinMaxScaler()
        past_target = scaler.fit_transform(past_target)   
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