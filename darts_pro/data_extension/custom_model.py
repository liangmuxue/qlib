from darts.models import TFTModel
from darts.models.forecasting.tft_model import _TFTModule
from darts.models.forecasting.tide_model import _TideModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts_pro.data_extension.series_data_utils import get_pred_center_value,build_serices_with_ndarray,get_np_center_value
from darts.models.forecasting.torch_forecasting_model import _raise_if_wrong_type,_get_runs_folder,INIT_MODEL_NAME
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
from darts.utils.utils import seq2series, series2seq

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import joblib

from .series_data_utils import StatDataAssis
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer,unverse_transform_slope_value
from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import compute_price_class,normalization,pairwise_compare,comp_max_and_rate
from cus_utils.metrics import compute_cross_metrics,compute_vr_metrics
import cus_utils.global_var as global_var
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC,SLOPE_SHAPE_SMOOTH,CLASS_LAST_VALUE_MAX
from darts_pro.data_extension.custom_tcn_model import LSTMReg,TargetDataReg
from darts_pro.data_extension.custom_module import _CusModule
from losses.mtl_loss import TripletLoss,UncertaintyLoss

import torchmetrics
from sklearn.preprocessing import MinMaxScaler
import tsaug
import re
import os

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
from darts_pro.data_extension.custom_dataset import CustomSequentialDataset,CustomInferenceDataset


hide_target = True

logger = get_logger(__name__)
from cus_utils.log_util import AppLogger
app_logger = AppLogger()

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
        loss_number=3,
        monitor=None,
        mode="train",
        past_split=None,
        filter_conv_index=0,
        no_dynamic_data=False,
        model_type="tft",
        **kwargs,
    ):
        """重载darts相关类"""
        
        self.mode = mode
        model_kwargs = {key: val for key, val in self.model_params.items()}
        del model_kwargs["model_type"]
        
        if "devices" in model_kwargs["pl_trainer_kwargs"]:
            self.device = "cuda:" + str(model_kwargs["pl_trainer_kwargs"]["devices"][0])
        else:
            self.device = "cpu"
        self.use_weighted_loss_func = use_weighted_loss_func
            
        if likelihood is None and loss_fn is None:
            # This is the default if no loss information is provided
            model_kwargs["loss_fn"] = None
            model_kwargs["likelihood"] = QuantileRegression()
            
        # 单独定制不确定损失
        # if self.use_weighted_loss_func:
        #     # 定义损失函数种类数量
        #     params = torch.ones(loss_number, requires_grad=True)
        #     loss_fn = UncertaintyLoss(device=self.device)    
        #     model_kwargs["loss_fn"] = loss_fn 
        #     model_kwargs["likelihood"] = likelihood
        #     self.loss_number = loss_number
            
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
        self.pl_module_params["batch_size"] = self.batch_size
            
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
        self.monitor = monitor
        self.past_split = past_split
        self.filter_conv_index = filter_conv_index
        self.no_dynamic_data = no_dynamic_data
        self.model_type = model_type
        self.loss_number = loss_number
        
    def _build_vriable_metas(self,tensors,static_covariates,seq=0):   
        
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
        conv_defs = [
                "past_target",
                "past_covariate",
                "historic_future_covariate",
            ]          
        input_meta = {}
        for i in range(len(type_names)):
            type_name = type_names[i]
            var_name = variable_names[i]
            tensor = tensors[i]
            # 根据相关设置，决定是否使用单独目标参数
            if hide_target:
                if type_name=="past_target":
                    type_values = ["target_0"]
                elif type_name=="future_target":
                    type_values = ["target_0"]
                else:
                    type_values = [f"{var_name}_{i}" for i in range(tensor.shape[-1])]
            else:
                type_values = [f"{var_name}_{i}" for i in range(tensor.shape[-1])]
            input_meta[type_name] = type_values
            
        variables_meta = {
            "input": input_meta,
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
                if input_var in conv_defs:
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
        
        return variables_meta,categorical_embedding_sizes  
        
    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            target_scaler,
            future_target_class,
            future_target,
            target_info,
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
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = self.define_output_dim()
        
        # 根据拆分的过去协变量，生成多个配置
        variables_meta_array = []
        for i in range(len(self.past_split)):
            past_index = self.past_split[i]
            past_covariate_item = past_covariate[:,past_index[0]:past_index[1]]
            tensors = [
                past_target,
                past_covariate_item,
                historic_future_covariate,  # for time varying encoders
                future_covariate,
                future_target,  # for time varying decoders
                static_covariates,  # for static encoder
            ]            
            variables_meta,categorical_embedding_sizes = self._build_vriable_metas(tensors, static_covariates,seq=i)
            variables_meta_array.append(variables_meta)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        output_dim = [len(CLASS_SIMPLE_VALUES.keys()),1]
        return _CusModule(
            output_dim=output_dim,
            variables_meta_array=variables_meta_array,
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
            loss_number=self.loss_number,
            past_split=self.past_split,
            filter_conv_index=self.filter_conv_index,
            device=self.device,
            train_sample=train_sample,
            model_type=self.model_type,
            **self.pl_module_params,
        )      

    def _init_model(self, trainer: Optional[pl.Trainer] = None):
        """Override Super"""
        
        tokens = pl.__version__.split(".")
        pl_200_or_above = int(tokens[0]) >= 2
        
        raise_if(
            self.pl_module_params is None,
            "`pl_module_params` must be extracted in __init__ method of `TorchForecastingModel` subclass after "
            "calling `super.__init__(...)`. Do this with `self._extract_pl_module_params(**self.model_params).`",
        )
        
        self.pl_module_params["train_sample_shape"] = []
        for variate in self.train_sample:
            if variate is not None and (isinstance(variate, torch.Tensor) or isinstance(variate, np.ndarray)):
                self.pl_module_params["train_sample_shape"].append(variate.shape)
            else:
                self.pl_module_params["train_sample_shape"].append(None)  
        # the tensors have shape (chunk_length, nr_dimensions)
        model = self._create_model(self.train_sample)
        self._module_name = model.__class__.__name__

        precision = None
        dtype = self.train_sample[0].dtype
        if np.issubdtype(dtype, np.float32):
            logger.info("Time series values are 32-bits; casting model to float32.")
            precision = "32" if not pl_200_or_above else "32-true"
        elif np.issubdtype(dtype, np.float64):
            logger.info("Time series values are 64-bits; casting model to float64.")
            precision = "64" if not pl_200_or_above else "64-true"
        else:
            raise_log(
                ValueError(
                    f"Invalid time series data type `{dtype}`. Cast your data to `np.float32` "
                    f"or `np.float64`, e.g. with `TimeSeries.astype(np.float32)`."
                ),
                logger,
            )
        precision_int = int(re.findall(r"\d+", str(precision))[0])

        precision_user = (
            self.trainer_params.get("precision", None)
            if trainer is None
            else trainer.precision
        )
        if precision_user is not None:
            # currently, we only support float 64 and 32
            valid_precisions = (
                ["64", "32"] if not pl_200_or_above else ["64-true", "32-true"]
            )
            if str(precision_user) not in valid_precisions:
                raise_log(
                    ValueError(
                        f"Invalid user-defined trainer_kwarg `precision={precision_user}`. "
                        f"Use one of ({valid_precisions})"
                    ),
                    logger,
                )
            precision_user_int = int(re.findall(r"\d+", str(precision_user))[0])
        else:
            precision_user_int = None

        raise_if(
            precision_user is not None and precision_user_int != precision_int,
            f"User-defined trainer_kwarg `precision='{precision_user}'` does not match dtype: `{dtype}` of the "
            f"underlying TimeSeries. Set `precision` to `{precision}` or cast your data to `{precision_user}"
            f"` with `TimeSeries.astype(np.float{precision_user_int})`.",
            logger,
        )
        self.trainer_params["precision"] = int(precision)

        # we need to save the initialized TorchForecastingModel as PyTorch-Lightning only saves module checkpoints
        if self.save_checkpoints:
            self.save(
                os.path.join(
                    _get_runs_folder(self.work_dir, self.model_name), INIT_MODEL_NAME
                )
            )
        return model
       
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
                mode="valid"
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

    def _setup_for_train(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        num_loader_workers: int = 0,
    ) -> Tuple[pl.Trainer, PLForecastingModule, DataLoader, Optional[DataLoader]]:

        """重载父类方法，规避相关数据检查"""
        self._fit_called = True
        # self._verify_train_dataset_type(train_dataset)

        # Pro-actively catch length exceptions to display nicer messages
        train_length_ok, val_length_ok = True, True
        try:
            len(train_dataset)
        except ValueError:
            train_length_ok = False
        if val_dataset is not None:
            try:
                len(val_dataset)
            except ValueError:
                val_length_ok = False

        raise_if(
            not train_length_ok or len(train_dataset) == 0,  # mind the order
            "The provided training time series dataset is too short for obtaining even one training point.",
            logger,
        )
        flag = val_dataset is not None and (not val_length_ok or len(val_dataset) == 0)
        raise_if(
            flag,
            "The provided validation time series dataset is too short for obtaining even one training point.",
            logger,
        )

        train_sample = train_dataset[0]
        if self.model is None:
            # 使用future_target部分(倒数第二列)，进行输出维度判断
            self.train_sample, self.output_dim = train_sample, train_sample[7].shape[1]
            model = self._init_model(trainer)
            model.monitor = self.monitor
        else:
            model = self.model
            # Check existing model has input/output dims matching what's provided in the training set.
            raise_if_not(
                len(train_sample) == len(self.train_sample),
                "The size of the training set samples (tuples) does not match what the model has been "
                "previously trained on. Trained on tuples of length {}, received tuples of length {}.".format(
                    len(self.train_sample), len(train_sample)
                ),
            )          
            # same_dims = tuple(
            #     s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in train_sample
            # ) == tuple(s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in self.train_sample)
            # raise_if_not(
            #     same_dims,
            #     "The dimensionality of the series in the training set do not match the dimensionality"
            #     " of the series the model has previously been trained on. "
            #     "Model input/output dimensions = {}, provided input/ouptput dimensions = {}".format(
            #         tuple(
            #             s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in self.train_sample
            #         ),
            #         tuple(s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in train_sample),
            #     ),
            # )

        # Setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_filter,
        )

        # Prepare validation data
        val_loader = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_loader_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_filter,
            )
        )

        # if user wants to train the model for more epochs, ignore the n_epochs parameter
        train_num_epochs = epochs if epochs > 0 else self.n_epochs

        # setup trainer
        trainer = self._setup_trainer(trainer,model, verbose, train_num_epochs)

        # TODO: multiple training without loading from checkpoint is not trivial (I believe PyTorch-Lightning is still
        #  working on that, see https://github.com/PyTorchLightning/pytorch-lightning/issues/9636)
        if self.epochs_trained > 0 and not self.load_ckpt_path:
            logger.warning(
                "Attempting to retrain the model without resuming from a checkpoint. This is currently "
                "discouraged. Consider setting `save_checkpoints` to `True` and specifying `model_name` at model "
                f"creation. Then call `model = {self.__class__.__name__}.load_from_checkpoint(model_name, "
                "best=False)`. Finally, train the model with `model.fit(..., epochs=new_epochs)` where "
                "`new_epochs` is the sum of (epochs already trained + some additional epochs)."
            )
        
        # Train model
        if epochs>0 and self.n_epochs>0:
            self._train(trainer,model,train_loader, val_loader)
            
        return trainer,model,train_loader,val_loader
    
           
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
            use_static_covariates=True,
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
            use_static_covariates=True,
        )
        
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        """重载方法以规避类型检查"""
        return
        # _raise_if_wrong_type(inference_dataset, CustomInferenceDataset)
    
    def _verify_predict_sample(self, predict_sample: Tuple):
        """重载方法以规避类型检查"""
        return
                        
    def _batch_collate_filter(self,ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        
        batch = ori_batch
        aggregated = []
        first_sample = ori_batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.from_numpy(np.stack(sample_list, axis=0))
                )
            elif isinstance(elem, MinMaxScaler):
                aggregated.append([sample[i] for sample in batch])
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, StockNormalizer):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)
    
    def define_output_dim(self):
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        output_dim = (1,1)    
        return output_dim    
                                                          
    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_multivariate(self) -> bool:
        return True    
    
    