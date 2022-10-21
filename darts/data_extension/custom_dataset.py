from darts.utils.data.sequential_dataset import (
    SplitCovariatesTrainingDataset,
)

from typing import Optional, List, Tuple, Union
import numpy as np

class CustomNumpyDataset(SplitCovariatesTrainingDataset):
    def __init__(
        self,
        numpy_data=None,
        input_chunk_length: int = 25,
        output_chunk_length: int = 5,
        future_covariate_index: List = [],
        past_covariate_index: List = [],
        static_covariate_index: List = [],
    ):
        """
        自定义数据集，直接使用numpy数据进行取值
        """
        super().__init__()

        self.numpy_data = numpy_data
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.future_covariate_index = future_covariate_index
        self.past_covariate_index = past_covariate_index
        self.static_covariate_index = static_covariate_index

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
        past_target = row[:self.input_chunk_length,-1:]
        past_covariate = row[:self.input_chunk_length,self.past_covariate_index]
        future_covariate = row[self.input_chunk_length:,self.future_covariate_index]
        historic_future_covariate = row[:self.input_chunk_length,self.future_covariate_index]
        future_target = row[self.input_chunk_length:,-1:]
        static_covariate = row[:self.input_chunk_length,self.static_covariate_index]
        
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            future_target,
        )