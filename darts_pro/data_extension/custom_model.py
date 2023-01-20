from darts.models import TFTModel
from darts.models.forecasting.tft_model import _TFTModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import _build_forecast_series
from darts_pro.data_extension.series_data_utils import get_pred_center_value,build_serices_with_ndarray,get_np_center_value
from darts.models.forecasting.torch_forecasting_model import _raise_if_wrong_type
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
from darts.utils.utils import seq2series, series2seq

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from cus_utils.tensor_viz import TensorViz
import torchmetrics
from torchmetrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

from losses.mtl_loss import CorrLoss,UncertaintyLoss

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
from darts.utils.data import (
    MixedCovariatesInferenceDataset,
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    TrainingDataset,
    InferenceDataset
)
from darts_pro.data_extension.custom_dataset import CustomNumpyDataset,CustomSequentialDataset,CustomInferenceDataset

logger = get_logger(__name__)

viz_input = TensorViz(env="data_train")

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
        use_weighted_loss_func=False,
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                full_attention,feed_forward,hidden_continuous_size,categorical_embedding_sizes,dropout,add_relative_index,norm_type,**kwargs)
        self.mean_squared_error = MeanSquaredError().to("cuda:0")
        # 提前初始化loss计算对象，避免在加载权重的时候出现空指针
        if use_weighted_loss_func and not isinstance(self.criterion,UncertaintyLoss):
            params = torch.ones(3, requires_grad=True)
            loss_sigma = torch.nn.Parameter(params)    
            self.criterion = UncertaintyLoss(loss_sigma=loss_sigma,device="cuda:0") 
        
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        
        output = self._produce_train_output(train_batch[:-1])
        target = train_batch[
            -1
        ]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained
        loss = self._compute_loss(output,target)
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.custom_histogram_adder(batch_idx)
        self._calculate_metrics(output, target, self.train_metrics)
        corr_loss = self.compute_corr_metrics(output, target)
        # mse_loss = self.compute_mse_metrics(output, target)   
        self.log("train_corr_loss", corr_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("train_mse_loss", mse_loss, batch_size=train_batch[0].shape[0], prog_bar=True)        
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        output = self._produce_train_output(val_batch[:-1])
        target = val_batch[-1]
        loss = self._compute_loss(output, target)
        corr_loss = self.compute_corr_metrics(output, target)
        # value_diff_loss = self.compute_value_diff_metrics(output, target)
        mse_loss = self.compute_mse_metrics(output, target)
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_corrr_loss", corr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_mse_loss", mse_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.val_metric_show(output,target,val_batch=val_batch)
        self._calculate_metrics(output, target, self.val_metrics)
        return loss
    
    def compute_corr_metrics(self,output,target):
        num_outputs = output.shape[0]
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        corr_loss = self.pearson(output_sample.transpose(1,0),target.squeeze(-1).transpose(1,0))  
        corr_loss = torch.mean(1 - corr_loss)
        return corr_loss      

    def compute_mse_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        mse_loss = self.mean_squared_error(output_sample, target[:,:,0])
        return mse_loss 

    def compute_value_diff_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        output_be = output_sample[:,[0,-1]]
        target_be = target[:,[0,-1],0]
        mse_loss = self.mean_squared_error(output_be, target_be)
        return mse_loss 
    
    def _compute_loss(self, output, target):
        """重载父类方法"""
        
        if self.likelihood:
            # 把似然估计损失叠加到自定义多任务损失里
            loss_like = self.likelihood.compute_loss(output, target)
            if self.criterion is None:
                return loss_like
            mtl_loss = self.criterion(output.squeeze(dim=-1), target,outer_loss=loss_like)
            return mtl_loss
        else:
            # If there's no likelihood, nr_params=1, and we need to squeeze out the
            # last dimension of model output, for properly computing the loss.
            return self.criterion(output.squeeze(dim=-1), target)

    def custom_histogram_adder(self,batch_idx):
        # iterating through all parameters
        for name,params in self.named_parameters():
            global_step = self.current_epoch*1000 + batch_idx
            global_step = batch_idx
            if params is not None:
                self.logger.experiment.add_histogram(name + "_weights",params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram(name + "_grad",params.grad,global_step)
        
    def configure_optimizers(self):
        """configures optimizers and learning rate schedulers for model optimization."""

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(
                    ValueError(
                        "Error when building the optimizer or learning rate scheduler;"
                        "please check the provided class and arguments"
                        "\nclass: {}"
                        "\narguments (kwargs): {}"
                        "\nerror:\n{}".format(cls, kws, e)
                    ),
                    logger,
                )

        # Create the optimizer and (optionally) the learning rate scheduler
        # we have to create copies because we cannot save model.parameters into object state (not serializable)
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizer_kws["params"] = self.parameters()

        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)

        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizer

            # ReduceLROnPlateau requires a metric to "monitor" which must be set separately, most others do not
            lr_monitor = lr_sched_kws.pop("monitor", None)

            lr_scheduler = _create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )
            return [optimizer], {
                "scheduler": lr_scheduler,
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }
        else:
            return optimizer
             
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Sequence[TimeSeries]:
        """重载原方法，服务于动态scaler模式"""
        
        input_data_tuple,scalers, batch_input_series = batch[:-2], batch[-2], batch[-1]
        scaler_map = self.build_scaler_map(scalers, batch_input_series)
        # number of individual series to be predicted in current batch
        num_series = input_data_tuple[0].shape[0]

        # number of times the input tensor should be tiled to produce predictions for multiple samples
        # this variable is larger than 1 only if the batch_size is at least twice as large as the number
        # of individual time series being predicted in current batch (`num_series`)
        batch_sample_size = min(
            max(self.pred_batch_size // num_series, 1), self.pred_num_samples
        )

        # counts number of produced prediction samples for every series to be predicted in current batch
        sample_count = 0

        # repeat prediction procedure for every needed sample
        batch_predictions = []
        while sample_count < self.pred_num_samples:

            # make sure we don't produce too many samples
            if sample_count + batch_sample_size > self.pred_num_samples:
                batch_sample_size = self.pred_num_samples - sample_count

            # stack multiple copies of the tensors to produce probabilistic forecasts
            input_data_tuple_samples = self._sample_tiling(
                input_data_tuple, batch_sample_size
            )

            # get predictions for 1 whole batch (can include predictions of multiple series
            # and for multiple samples if a probabilistic forecast is produced)
            batch_prediction = self._get_batch_prediction(
                self.pred_n, input_data_tuple_samples, self.pred_roll_size
            )

            # reshape from 3d tensor (num_series x batch_sample_size, ...)
            # into 4d tensor (batch_sample_size, num_series, ...), where dim 0 represents the samples
            out_shape = batch_prediction.shape
            batch_prediction = batch_prediction.reshape(
                (
                    batch_sample_size,
                    num_series,
                )
                + out_shape[1:]
            )

            # save all predictions and update the `sample_count` variable
            batch_predictions.append(batch_prediction)
            sample_count += batch_sample_size

        # concatenate the batch of samples, to form self.pred_num_samples samples
        batch_predictions = torch.cat(batch_predictions, dim=0)
        batch_predictions = batch_predictions.cpu().detach().numpy()
        # self.predict_data_eval(batch_predictions,batch_input_series)
        
        batch_predictions_unscale = np.zeros([batch_predictions.shape[0],
                    batch_predictions.shape[1],batch_predictions.shape[2],batch_predictions.shape[3]])
        # 分别遍历每个系列，进行反向归一化
        for i in range(batch_predictions.shape[1]):
            scaler = scalers[i]
            for j in range(batch_predictions.shape[0]):
                batch_prediction = batch_predictions[j,i,:,:]
                batch_prediction_unscale = scaler.inverse_transform(batch_prediction)
                batch_predictions_unscale[j,i,:,:] = batch_prediction_unscale
            
        ts_forecasts = Parallel(n_jobs=self.pred_n_jobs)(
            delayed(_build_forecast_series)(
                [batch_prediction[batch_idx] for batch_prediction in batch_predictions_unscale],
                input_series,
            )
            for batch_idx, input_series in enumerate(batch_input_series)
        )
        return ts_forecasts
    
    def predict_data_eval(self,batch_predictions,batch_input_series):
        """保持scaler方式评估预测数据"""
        
        for batch_idx, input_series in enumerate(batch_input_series):
            for batch_prediction in batch_predictions:
                pred = batch_prediction[batch_idx]
    
    def val_metric_show(self,output,target,val_batch=None):
        size = 9 if target.shape[0]>9 else target.shape[0]
        view_range = np.random.randint(target.shape[0],size=size)
        for i,s_index in enumerate(view_range):
            target_sample = target[s_index,:,0].cpu().numpy()
            output_sample = output[s_index,:,0,:].cpu().numpy()
            
            pred_center_data = get_np_center_value(output_sample)
    
            target_title = "val_metric_{}".format(i)
            names = ["output","target"]
            view_data = np.stack((pred_center_data,target_sample),axis=0).transpose(1,0)
            viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,names=names)              
    
    def build_scaler_map(self,scalers, batch_input_series,group_column="instrument_rank"):
        scaler_map = {}
        for i in range(len(scalers)):
            series = batch_input_series[i]
            group_col_val = series.static_covariates[group_column].values[0]
            scaler_map[group_col_val] = scalers[i]
        return scaler_map
    
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


class TFTExtModel(MixedCovariatesTorchModel):
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
        use_weighted_loss_func:bool = False,
        **kwargs,
    ):
        """重载darts相关类"""
        
        model_kwargs = {key: val for key, val in self.model_params.items()}
        self.device = "cuda:" + str(model_kwargs["pl_trainer_kwargs"]["devices"][0])
        self.use_weighted_loss_func = use_weighted_loss_func
            
        if likelihood is None and loss_fn is None:
            # This is the default if no loss information is provided
            model_kwargs["loss_fn"] = None
            model_kwargs["likelihood"] = QuantileRegression()
            
        # 单独定制不确定损失
        if self.use_weighted_loss_func:
            params = torch.ones(3, requires_grad=True)
            loss_sigma = torch.nn.Parameter(params)        
            loss_fn = UncertaintyLoss(device=self.device,loss_sigma=loss_sigma)    
            model_kwargs["loss_fn"] = loss_fn 
            model_kwargs["likelihood"] = QuantileRegression()
            
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
            
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = (
            categorical_embedding_sizes
            if categorical_embedding_sizes is not None
            else {}
        )
        self.add_relative_index = add_relative_index
        self.output_dim: Optional[Tuple[int, int]] = None
        self.norm_type = norm_type
    
    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
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
            use_weighted_loss_func=self.use_weighted_loss_func,
            **self.pl_module_params,
        )      
    
    def fit(
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
        """重载原方法"""
        # guarantee that all inputs are either list of `TimeSeries` or `None`
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        val_series = series2seq(val_series)
        val_past_covariates = series2seq(val_past_covariates)
        val_future_covariates = series2seq(val_future_covariates)

        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_fit_encodings(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        self._verify_past_future_covariates(
            past_covariates=past_covariates, future_covariates=future_covariates
        )
        self._verify_static_covariates(series[0].static_covariates)

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            if self.encoders.encoding_available:
                (
                    val_past_covariates,
                    val_future_covariates,
                ) = self.generate_fit_encodings(
                    series=val_series,
                    past_covariates=val_past_covariates,
                    future_covariates=val_future_covariates,
                )
            self._verify_past_future_covariates(
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )
            self._verify_static_covariates(val_series[0].static_covariates)

            match = (
                series[0].width == val_series[0].width
                and (past_covariates[0].width if past_covariates is not None else None)
                == (
                    val_past_covariates[0].width
                    if val_past_covariates is not None
                    else None
                )
                and (
                    future_covariates[0].width
                    if future_covariates is not None
                    else None
                )
                == (
                    val_future_covariates[0].width
                    if val_future_covariates is not None
                    else None
                )
            )
            raise_if_not(
                match,
                "The dimensions of the series in the training set "
                "and the validation set do not match.",
            )

        train_dataset = self._build_train_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            mode="train"
        )

        if val_series is not None:
            val_dataset = self._build_train_dataset(
                target=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts,
                mode="val"
            )
        else:
            val_dataset = None

        # Pro-actively catch length exceptions to display nicer messages
        length_ok = True
        try:
            len(train_dataset)
        except ValueError:
            length_ok = False
        raise_if(
            not length_ok or len(train_dataset) == 0,  # mind the order
            "The train dataset does not contain even one training sample. "
            + "This is likely due to the provided training series being too short. "
            + "This model expect series of length at least {}.".format(
                self.min_train_series_length
            ),
        )
        logger.info(f"Train dataset contains {len(train_dataset)} samples.")

        if isinstance(series, TimeSeries):
            # if only one series is provided, save it for prediction time (including covariates, if available)
            self.training_series = series
            self.static_covariates = series.static_covariates
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
        else:
            self.static_covariates = series[0].static_covariates
            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True
        self._fit_called = True

        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )
        
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="train"
    ) -> CustomSequentialDataset:
        
        return CustomSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
            mode=mode
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> CustomInferenceDataset:

        return CustomInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )
        
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        """重载方法以规避类型检查"""
        _raise_if_wrong_type(inference_dataset, CustomInferenceDataset)

    @staticmethod
    def _batch_collate_fn(ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        
        batch = []
        # 过滤不符合条件的记录
        for b in ori_batch:
            # 如果每天的target数值都相同，则会出现loss的NAN，需要过滤掉
            future_target = b[-1]
            if not np.all(future_target == future_target[0]):
                batch.append(b)
            
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                aggregated.append(
                    torch.from_numpy(np.stack([sample[i] for sample in batch], axis=0))
                )
            elif isinstance(elem, MinMaxScaler):
                aggregated.append([sample[i] for sample in batch])
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)
                                        
    @staticmethod
    def _supports_static_covariates() -> bool:
        return True         