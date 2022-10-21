from darts.models import TFTModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import pytorch_lightning as pl

from demo.cus_utils.data.custom_dataset import CustomNumpyDataset

class TFTCusModel(TFTModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: Union[int, List[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            Dict[str, Union[int, Tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        **kwargs,
    ):
        """
        自定义model，直接使用numpy数据
        """

        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,
                         num_attention_heads,full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,
                         add_relative_index,loss_fn,likelihood,norm_type,**kwargs)

        self.columns = ['time_idx', 'instrument', 'dayofweek', 'STD5', 'VSTD5', 'label','ori_label']
        self.future_covariate_col = ["dayofweek"]
        self.past_covariate_col = ['STD5', 'VSTD5', 'label']
        self.static_covariate_col = ['instrument']
        
    def _build_train_dataset(
        self,
        numpy_data,
    ) -> CustomNumpyDataset:
        """使用自定义数据集"""
        
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

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(
            isinstance(train_dataset, CustomNumpyDataset),
            "TFTModel requires a training dataset of type MixedCovariatesTrainingDataset.",
        )        
    @random_method
    def fit(
        self,
        train_numpy_data,
        val_numpy_data,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):
        """重载父类方法，使用自定义数据集

        Returns
        -------
        self
            Fitted model.
        """
        
        train_dataset = self._build_train_dataset(train_numpy_data)
        val_dataset = self._build_train_dataset(val_numpy_data)

        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )          
        