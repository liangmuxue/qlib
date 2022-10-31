from darts.models import TFTModel
from darts.models.forecasting.tft_model import _TFTModule
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

class _TFTCusModule(_TFTModule):
    def __init__(
        self,
        output_dim: Tuple[int, int],
        variables_meta: Dict[str, Dict[str, List[str]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        **kwargs,
    ):
        return super().__init__(output_dim,variables_meta,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                full_attention,feed_forward,hidden_continuous_size,categorical_embedding_sizes,dropout,add_relative_index,norm_type,**kwargs)
        
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]  # By convention target is always the last element returned by datasets
        loss = self._compute_loss(output, target)
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self._calculate_metrics(output, target, self.train_metrics)
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self._calculate_metrics(output, target, self.val_metrics)
        return loss
        
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
        self.super_fit_mode = False
        
        self.columns = ['time_idx', 'instrument', 'dayofweek', 'STD5', 'VSTD5', 'label','ori_label']
        self.future_covariate_col = ["dayofweek"]
        self.past_covariate_col = ['STD5', 'VSTD5', 'label']
        self.static_covariate_col = ['instrument']
        
    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        if self.super_fit_mode:
            raise_if_not(
                isinstance(train_dataset, MixedCovariatesTrainingDataset),
                "TFTModel requires a training dataset of type MixedCovariatesTrainingDataset.",
            )    
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
        return super().fit(series,past_covariates,future_covariates,val_series,
                           val_past_covariates,val_future_covariates,trainer,verbose,epochs,max_samples_per_ts,num_loader_workers)
        
    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载父类方法"""
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            future_target,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                        self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index(
                            [i for i in range(static_covariates.shape[1])]
                        )
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                        zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the TFTModel documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes

        return _TFTCusModule(
            output_dim=self.output_dim,
            variables_meta=variables_meta,
            num_static_components=n_static_components,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            feed_forward=self.feed_forward,
            hidden_continuous_size=self.hidden_continuous_size,
            categorical_embedding_sizes=self.categorical_embedding_sizes,
            add_relative_index=self.add_relative_index,
            norm_type=self.norm_type,
            **self.pl_module_params,
        )
    
        