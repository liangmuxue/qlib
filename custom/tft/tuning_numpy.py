"""
Hyperparameters can be efficiently tuned with `optuna <https://optuna.readthedocs.io/>`_.
"""
import copy
import logging
import os
from typing import Any, Dict, Tuple, Union

import pickle
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import optuna.logging
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import statsmodels.api as sm
import torch
from torch.utils.data import DataLoader

from tft.tft_model_nor import TftModelNor
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

optuna_logger = logging.getLogger("optuna")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class OptimizeHyperparameters(object):
    
    def __init__(self,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        model_path: str = None,
        max_epochs: int = 20,
        n_trials: int = 100,
        timeout: float = 3600 * 8.0,  # 8 hours
        gradient_clip_val_range: Tuple[float, float] = (0.01, 100.0),
        pred_size: int = 5,
        hidden_size_range: Tuple[int, int] = (16, 265),
        hidden_continuous_size_range: Tuple[int, int] = (8, 64),
        attention_head_size_range: Tuple[int, int] = (1, 4),
        dropout_range: Tuple[float, float] = (0.1, 0.3),
        learning_rate_range: Tuple[float, float] = (1e-5, 1.0),
        use_learning_rate_finder: bool = True,
        trainer_kwargs: Dict[str, Any] = {},
        log_dir: str = "lightning_logs",
        study: optuna.Study = None,
        verbose: Union[int, bool] = None,
        pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner(),
        clean_mode=False,
        load_weights=None,
        trial_no=None,
        epoch_no=None,
        viz=False,
        gpus=[1],
        **kwargs
    ) -> optuna.Study:
        """重构超参数trial"""
        
        self.model_path = model_path
        self.pred_size = pred_size
        self.load_weights = load_weights
        self.trial_no = trial_no
        self.epoch_no = epoch_no
        if clean_mode is True:
            return None
                
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_epochs = max_epochs
        self.n_trials = n_trials
        self.timeout = timeout
        self.gradient_clip_val_range = gradient_clip_val_range
        self.hidden_size_range = hidden_size_range
        self.hidden_continuous_size_range = hidden_continuous_size_range
        self.attention_head_size_range = attention_head_size_range
        self.dropout_range = dropout_range
        self.learning_rate_range = learning_rate_range
        self.use_learning_rate_finder = use_learning_rate_finder
        self.trainer_kwargs = trainer_kwargs
        self.log_dir = log_dir
        self.study = study
        self.verbose = verbose        
        self.pruner = pruner
        self.viz = viz
        self.gpus = gpus
        self.kwargs = kwargs  
        
        assert isinstance(train_dataloader.dataset, TimeSeriesDataSet) and isinstance(
            val_dataloader.dataset, TimeSeriesDataSet
        ), "dataloaders must be built from timeseriesdataset"
    
        logging_level = {
            None: optuna.logging.get_verbosity(),
            0: optuna.logging.WARNING,
            1: optuna.logging.INFO,
            2: optuna.logging.DEBUG,
        }
        verbose = 2
        self.optuna_verbose = logging_level[verbose]
        optuna.logging.set_verbosity(self.optuna_verbose)
    
        self.loss = kwargs.get(
            "loss", QuantileLoss()
        )  # need a deepcopy of loss as it will otherwise propagate from one trial to the next

    
    # create objective function
    
    def objective(self,trial: optuna.Trial) -> float:
        # Filenames for each trial must be made unique in order to access each checkpoint.
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(self.model_path, "trial_{}".format(trial.number)), filename="{epoch}", monitor="val_loss"
        )

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()
        learning_rate_callback = LearningRateMonitor()
        logger = TensorBoardLogger(self.log_dir, name="optuna", version=trial.number)
        gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *self.gradient_clip_val_range)
        default_trainer_kwargs = dict(
            accelerator='gpu', 
            devices=[0],
            max_epochs=self.max_epochs,
            gradient_clip_val=gradient_clip_val,
            callbacks=[
                metrics_callback,
                learning_rate_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ],
            logger=logger,
            enable_progress_bar=self.optuna_verbose < optuna.logging.INFO,
            enable_model_summary=False,
        )
        default_trainer_kwargs.update(self.trainer_kwargs)
        trainer = pl.Trainer(
            **default_trainer_kwargs,
        )

        # create model
        hidden_size = trial.suggest_int("hidden_size", *self.hidden_size_range, log=True)
        self.kwargs["loss"] = copy.deepcopy(self.loss)
        device = torch.device("cuda:{}".format(self.gpus))
        model = TftModelNor.from_dataset(
            self.train_dataloader.dataset,
            dropout=trial.suggest_uniform("dropout", *self.dropout_range),
            hidden_size=hidden_size,
            hidden_continuous_size=trial.suggest_int(
                "hidden_continuous_size",
                self.hidden_continuous_size_range[0],
                min(self.hidden_continuous_size_range[1], hidden_size),
                log=True,
            ),
            attention_head_size=trial.suggest_int("attention_head_size", *self.attention_head_size_range),
            log_interval=-1,
            device=device,
            **self.kwargs,
        )
        model.ext_properties(fig_save_path=None,viz=self.viz)
        # find good learning rate
        if self.use_learning_rate_finder:
            lr_trainer = pl.Trainer(
                gradient_clip_val=gradient_clip_val,
                gpus=[0] if torch.cuda.is_available() else None,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            res = lr_trainer.tuner.lr_find(
                model,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader,
                early_stop_threshold=10000,
                min_lr=self.learning_rate_range[0],
                num_training=100,
                max_lr=self.learning_rate_range[1],
            )

            loss_finite = np.isfinite(res.results["loss"])
            if loss_finite.sum() > 3:  # at least 3 valid values required for learning rate finder
                lr_smoothed, loss_smoothed = sm.nonparametric.lowess(
                    np.asarray(res.results["loss"])[loss_finite],
                    np.asarray(res.results["lr"])[loss_finite],
                    frac=1.0 / 10.0,
                )[min(loss_finite.sum() - 3, 10) : -1].T
                optimal_idx = np.gradient(loss_smoothed).argmin()
                optimal_lr = lr_smoothed[optimal_idx]
            else:
                optimal_idx = np.asarray(res.results["loss"]).argmin()
                optimal_lr = res.results["lr"][optimal_idx]
            optuna_logger.info(f"Using learning rate of {optimal_lr:.3g}")
            # add learning rate artificially
            model.hparams.learning_rate = trial.suggest_uniform("learning_rate", optimal_lr, optimal_lr)
        else:
            model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *self.learning_rate_range)

        # fit
        trainer.fit(model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        # report result
        return metrics_callback.metrics[-1]["val_loss"].item()

    # setup optuna and run
    def objective_best(self,trial: optuna.Trial) -> float:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(self.model_path, "trial_{}".format(trial.number)), filename="{epoch}", monitor="val_loss"
        )
        metrics_callback = MetricsCallback()
        logger = TensorBoardLogger(self.log_dir, name="optuna_best", version=trial.number)
        gradient_clip_val = trial.suggest_loguniform("gradient_clip_val", *self.gradient_clip_val_range)
        default_trainer_kwargs = dict(
            gpus=[0] if torch.cuda.is_available() else None,
            max_epochs=self.max_epochs,
            gradient_clip_val=gradient_clip_val,
            logger=logger,
            callbacks=[
                metrics_callback,
                checkpoint_callback,
            ],            
            enable_progress_bar=self.optuna_verbose < optuna.logging.INFO,
            enable_model_summary=False,
        )
        default_trainer_kwargs.update(self.trainer_kwargs)     
        self.trainer = pl.Trainer(
            **default_trainer_kwargs,
        )               
        if self.load_weights:
            model = self.get_tft()     
        else:        
            # create model
            hidden_size = trial.suggest_int("hidden_size", *self.hidden_size_range, log=True)
            self.kwargs["loss"] = copy.deepcopy(self.loss)
            model = TftModelNor.from_dataset(
                self.train_dataloader.dataset,
                dropout=trial.suggest_uniform("dropout", *self.dropout_range),
                hidden_size=hidden_size,
                hidden_continuous_size=trial.suggest_int(
                    "hidden_continuous_size",
                    self.hidden_continuous_size_range[0],
                    min(self.hidden_continuous_size_range[1], hidden_size),
                    log=True,
                ),
                attention_head_size=trial.suggest_int("attention_head_size", *self.attention_head_size_range),
                log_interval=-1,
                **self.kwargs,
            )
            model.hparams.learning_rate = trial.suggest_loguniform("learning_rate", *self.learning_rate_range)
        model.ext_properties(fig_save_path=None,viz=self.viz)   
        # fit
        self.trainer.fit(model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        # report result
        return metrics_callback.metrics[-1]["val_loss"].item()

    def do_study(self):
        study = optuna.create_study(direction="minimize", pruner=self.pruner) 
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        return study

    def do_apply(self,study_path):
        df = open(study_path,'rb')
        study = pickle.load(df)
        df.close()
        self.objective_best(study.best_trial)
    
    def get_tft(self,fig_save_path=None,viz=False):
        best_model_path = os.path.join(self.model_path, "trial_{}/epoch={}.ckpt".format(self.trial_no,self.epoch_no))
        best_tft = TftModelNor.load_from_checkpoint(best_model_path)
        best_tft.ext_properties(fig_save_path=fig_save_path,viz=viz)
        return best_tft
    