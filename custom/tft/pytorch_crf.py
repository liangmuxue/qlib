# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import Counter
import pandas as pd
import pickle
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
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from .tuning_cus import OptimizeHyperparameters
from tft.tft_dataset import TFTDataset
from custom_model.seq2seq_crf import SeqCrf
from custom_model.parameters import *
from cus_utils.utils_crf import save_checkpoint
from time import time

class CrfModel(Model):
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
        # 模式 opt_train:寻找最优化参数训练 "best_train":使用最优化参数训练
        type="opt_train",
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
        self.gpus = GPU
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        
        self.type = type
        self.fig_save_path = kwargs['fig_save_path']
    
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def fit(
        self,
        dataset: TFTDataset
    ):
        if self.type.startswith("pred"):
            # 直接进行预测,只需要加载模型参数
            print("do nothing for pred")
            return      
        # 取得训练数据(DataFrame)
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 生成tft时间序列训练数据集
        ts_data_train = dataset.get_crf_dataset(df_train)
        train_loader = ts_data_train.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)
        # 生成tft验证集,并删除不在训练集中的股票数据
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = df_valid[df_valid['instrument'].isin(df_train['instrument'].unique())]
        validation = dataset.get_crf_dataset(df_valid,mode="valid")
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)       
        self.model = SeqCrf(
            hidden_size=self.optargs['hidden_size'],
            input_size=self.optargs["input_size"],
            step_len=self.optargs["step_len"],
            viz=self.optargs['viz'],
            gpus=self.gpus
            
        )      
        self.enc_optim = torch.optim.Adam(self.model.enc.parameters(), lr = LEARNING_RATE)
        self.dec_optim = torch.optim.Adam(self.model.dec.parameters(), lr = LEARNING_RATE)    
        epoch = 0  
        num_epochs = self.optargs["max_epochs"]
        
        
        for step in range(self.n_epochs):
            timer = time()
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss = self.train_epoch(train_loader)
            scores = self.test_epoch(val_loader)
            timer = time() - timer
            self.logger.info("train %.6f, valid %.6f" % (train_loss, scores))
            save_checkpoint("", None, step, train_loss, timer)

    def predict(self, dataset: TFTDataset):

        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        test_ds = dataset.get_ts_dataset(df_test,mode="valid")
        test_loader = test_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)        

                
    def train_epoch(self, data_loader):
        self.model.train()
        loss_sum = 0              
        for (x,y) in data_loader:
            loss = self.model(x,y) # forward pass and compute loss
            loss.backward() # compute gradients
            self.enc_optim.step() # update encoder parameters
            self.dec_optim.step() # update decoder parameters
            loss_sum += loss.item()   
        return loss_sum
        
    def test_epoch(self, data_loader):

        self.model.eval()

        scores = torch.tensor([]).cuda()
        losses = []

        for (x,y)  in data_loader:
            score = self.model.val(x, y)
            scores = torch.cat((scores,score))

        return torch.mean(scores)
