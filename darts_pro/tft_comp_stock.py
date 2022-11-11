#!/usr/bin/env python
# coding: utf-8

# # Temporal Fusion Transformer
# In this notebook, we show two examples of how two use Darts' `TFTModel`.
# If you are new to darts, we recommend you first follow the `darts-intro.ipynb` notebook.

# In[1]:


# fix python path if working locally
from examples.utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings


def process(df):
    
    warnings.filterwarnings("ignore")
    import logging
    
    logging.disable(logging.CRITICAL)
    
    
    num_samples = 200
    
    figsize = (9, 6)
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
    
    
    
    # Read data
    series = TimeSeries.from_dataframe(df,time_col="time_idx",
                                             freq='D',
                                             fill_missing_dates=True,
                                             value_cols="label")      
    
    series = series.astype(np.float32)
    
    # Create training and validation sets:
    training_cutoff = 1560
    train, val = series.split_after(training_cutoff)
    
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(series)
    
    covariates = TimeSeries.from_dataframe(df,time_col="time_idx",
                                             freq='D',
                                             fill_missing_dates=True,
                                             value_cols="dayofweek")   
    covariates = covariates.astype(np.float32)
    
    # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
    scaler_covs = Scaler()
    cov_train, cov_val = covariates.split_after(training_cutoff)
    scaler_covs.fit(cov_train)
    covariates_transformed = scaler_covs.transform(covariates)

    covariates_past = TimeSeries.from_dataframe(df,time_col="time_idx",
                                             freq='D',
                                             fill_missing_dates=True,
                                             value_cols=["STD5","PRICE_SCOPE"])   
    scaler_covs_past = Scaler()
    cov_train_past, cov_val_past = covariates_past.split_after(training_cutoff)
    scaler_covs_past.fit(cov_train_past)
    covariates_transformed_past = scaler_covs_past.transform(covariates_past)
    
    # default quantiles for QuantileRegression
    quantiles = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]
    input_chunk_length = 25
    forecast_horizon = 5
    n_epochs = 100
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=128,
        n_epochs=n_epochs,
        model_name="Stock_TFT",
        add_relative_index=True,
        add_encoders=None,
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
        log_tensorboard=True,
        force_reset=True,
        save_checkpoints=True,
        work_dir="custom/data/darts",
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}     
    )
    
    # my_model = TFTModel.load_from_checkpoint("Stock_TFT",work_dir="custom/data/darts",best=True)
    
    
    my_model.fit(train_transformed, future_covariates=covariates_transformed, val_series=val_transformed[30:],
                 val_future_covariates=covariates_transformed,past_covariates=covariates_transformed_past,val_past_covariates=covariates_transformed_past,
                 verbose=True,epochs=n_epochs)
    
    
    def eval_model(model, n, actual_series, val_series):
        pred_series = model.predict(n=n, num_samples=num_samples)
    
        # plot actual series
        plt.figure(figsize=figsize)
        actual_series[1500: pred_series.end_time()].plot(label="actual")
    
        # plot prediction with quantile ranges
        pred_series.plot(
            low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        )
        pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    
        plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
        plt.legend()
        plt.savefig('custom/data/darts/result_view/eval_exp.jpg')
    
    eval_model(my_model, 15, series_transformed, val_transformed)
    

    
    # backtest_series = my_model.historical_forecasts(
    #     series_transformed,
    #     future_covariates=covariates_transformed,
    #     start=train.end_time() + train.freq,
    #     num_samples=num_samples,
    #     forecast_horizon=forecast_horizon,
    #     stride=forecast_horizon,
    #     last_points_only=False,
    #     retrain=False,
    #     verbose=True,
    # )
    #
    #
    #
    # def eval_backtest(backtest_series, actual_series, horizon, start, transformer):
    #     plt.figure(figsize=figsize)
    #     actual_series.plot(label="actual")
    #     backtest_series.plot(
    #         low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    #     )
    #     backtest_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    #     plt.legend()
    #     plt.title(f"Backtest, starting {start}, {horizon}-months horizon")
    #     print(
    #         "MAPE: {:.2f}%".format(
    #             mape(
    #                 transformer.inverse_transform(actual_series),
    #                 transformer.inverse_transform(backtest_series),
    #             )
    #         )
    #     )
    #
    #
    # eval_backtest(
    #     backtest_series=concatenate(backtest_series),
    #     actual_series=series_transformed,
    #     horizon=forecast_horizon,
    #     start=training_cutoff,
    #     transformer=transformer,
    # )



