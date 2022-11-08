from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.data.training_dataset import (
    MixedCovariatesTrainingDataset
)
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_dataset import CustomNumpyDataset
logger = get_logger(__name__)


class CusTcnModel(TCNModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        kernel_size=5,
        num_filters: int = 3,
        num_layers: Optional[int] = None,
        dilation_base: int = 2,
        weight_norm: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        自定义model，直接使用numpy数据
        """

        super().__init__(input_chunk_length,output_chunk_length,dropout=dropout,num_filters=num_filters,num_layers=num_layers,kernel_size=kernel_size,
                         dilation_base=dilation_base,weight_norm=weight_norm,**kwargs)
        self.super_fit_mode = False
        
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if self.super_fit_mode:
            super()._verify_train_dataset_type(train_dataset)          
        else:          
            raise_if_not(
                isinstance(train_dataset, CustomNumpyDataset),
                "TFTCusModel requires a training dataset of type CustomNumpyDataset.",
            )      
          
    @random_method
    def fit(
        self,
        train_dataset,
        val_dataset,
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
        
        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )          

    def numpy_predict(self,input_dataset,trainer=None,batch_size=1024,mc_dropout=False,verbose=True):
        pred_loader = DataLoader(
            input_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_fn,
        )

        # Set mc_dropout rate
        self.model.set_mc_dropout(mc_dropout)

        # setup trainer. will only be re-instantiated if both `trainer` and `self.trainer` are `None`
        trainer = trainer if trainer is not None else self.trainer
        self._setup_trainer(trainer=trainer, verbose=verbose, epochs=self.n_epochs)

        # prediction output comes as nested list: list of predicted `TimeSeries` for each batch.
        predictions = self.trainer.predict(self.model, pred_loader)
        return predictions
                        
    def super_fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):        
        self.super_fit_mode = True
        return super().fit(series,past_covariates,None,val_series,
                               val_past_covariates,None,trainer,verbose,epochs,max_samples_per_ts,num_loader_workers)
        
    
        