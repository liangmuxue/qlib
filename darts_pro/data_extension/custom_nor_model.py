from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.models.forecasting.block_rnn_model import _BlockRNNModule
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


class CusNorModel(BlockRNNModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        model: Union[str, nn.Module] = "RNN",
        hidden_dim: int = 25,
        n_rnn_layers: int = 1,
        hidden_fc_sizes: Optional[List] = None,
        dropout: float = 0.0,
        model_type="lstm",
        **kwargs,
    ):
        """
        自定义model，直接使用numpy数据
        """

        super().__init__(input_chunk_length,output_chunk_length,model,hidden_dim,n_rnn_layers,dropout=dropout,hidden_fc_sizes=hidden_fc_sizes,**kwargs)
        self.super_fit_mode = False
        self.model_type = model_type
        
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if self.super_fit_mode:
            if self.model_type=="lstm":
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
        if self.model_type=="lstm":
            return super().fit(series,past_covariates,None,val_series,
                               val_past_covariates,None,trainer,verbose,epochs,max_samples_per_ts,num_loader_workers)
        
    # def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
    #     """重载父类方法"""
    #
    #     nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
    #
    #     model = _BlockRNNModule(
    #       name=self.rnn_type_or_module,
    #       input_size=self.input_dim,
    #       target_size=self.output_dim,
    #       nr_params=nr_params,
    #       hidden_dim=self.hidden_dim,
    #       num_layers=self.n_rnn_layers,
    #       num_layers_out_fc=self.hidden_fc_sizes,
    #       dropout=self.dropout,
    #       **self.pl_module_params,
    #     )
    
        