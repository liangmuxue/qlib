# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP

from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from .tuning_cus import OptimizeHyperparameters
from tft.tft_dataset import TFTDataset

class TftModel(Model):
    def __init__(
        self,
        d_model: int = 64,
        batch_size: int = 8192,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="",
        early_stop=5,
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        optargs=None,
        **kwargs
    ):
        
        # 超参数部分
        self.optargs = optargs
        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

    def get_study(self,dataset):
        study_opt = OptimizeHyperparameters(
            train_loader,
            val_loader,
            model_path="examples/weights",
            n_trials=self.n_trials,
            max_epochs=self.max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=0.1,log_every_n_steps=16),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )    
        return study_opt    
        
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def train_epoch(self, data_loader):

        self.model.train()

        for data in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            feature = transforms_def(feature)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device)
            feature = transforms_def(feature)
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float())  # .float()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def get_training(self):
        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
            time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
                "avg_volume_by_agency",
                "avg_volume_by_sku",
            ],
            target_normalizer=GroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            ),  # use softplus with beta=1.0 and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return training
    
    def fit(
        self,
        dataset: TFTDataset
    ):
        if self.type=="train":
            self.train_dataloader = dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)
        else:
            validation = TFTDataset.from_dataset(dataset, dataset.data, predict=True, stop_randomization=True)
            self.val_dataloader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)            

        self.study_opt = self.get_study(dataset)       

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()
