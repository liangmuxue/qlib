import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, SunspotsDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

# Read data:
series = AirPassengersDataset().load()

# Create training and validation sets:
train, val = series.split_after(pd.Timestamp("19590101"))

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

# create month and year covariate series
year_series = datetime_attribute_timeseries(
    pd.date_range(start=series.start_time(), freq=series.freq_str, periods=1000),
    attribute="year",
    one_hot=False,
)
year_series = Scaler().fit_transform(year_series)
month_series = datetime_attribute_timeseries(
    year_series, attribute="month", one_hot=False
)
month_series = Scaler().fit_transform(month_series)
covariates = year_series.stack(month_series)
cov_train, cov_val = covariates.split_after(pd.Timestamp("19590101"))

my_model = RNNModel(
    model="LSTM",
    hidden_dim=20,
    dropout=0,
    batch_size=16,
    n_epochs=300,
    optimizer_kwargs={"lr": 1e-3},
    model_name="Air_RNN",
    log_tensorboard=True,
    random_state=42,
    training_length=15,
    input_chunk_length=14,
    force_reset=True,
    save_checkpoints=True,
    work_dir="custom/data/darts",
    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]}
)

forecast_model,train_loader,val_loader = my_model.fit(
    train_transformed,
    future_covariates=None,
    val_series=val_transformed,
    val_future_covariates=None,
    epochs=1,
    verbose=True,
)
my_model.model.cpu()
del my_model
print("train ds len:",train_loader.dataset.__len__())
print("val ds len:",val_loader.dataset.__len__())

def data_view(dataset,type="train"):
    target = None
    for item in dataset:
        if target is None:
            target = item[0]
        else:
            target = np.concatenate((target,item[0]),axis=0)
            
    from cus_utils.tensor_viz import TensorViz        
    viz_input = TensorViz(env="data_hist")        
    title = "lstm_{}_data".format(type)
    viz_input.viz_data_hist(target.reshape(-1),numbins=10,win=title,title=title) 

data_view(train_loader.dataset)   
data_view(val_loader.dataset,type="val") 
