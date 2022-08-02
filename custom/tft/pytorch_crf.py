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
from cus_utils.utils_crf import save_checkpoint,load_checkpoint
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
        device = torch.device("cuda:{}".format(self.gpus)) 
        self.device = device
        self.model = SeqCrf(
            hidden_size=self.optargs['hidden_size'],
            input_size=self.optargs["input_size"],
            step_len=self.optargs["step_len"],
            viz=self.optargs['viz'],
            device=device
            
        )      
        num_epochs = self.optargs["max_epochs"]
        weight_path = self.optargs['weight_path']        
        if self.optargs['load_weights']:
            step = self.optargs["best_ckpt_no"]
            filepath = "{}/crf_{}.pth".format(weight_path,step)
            epoch = load_checkpoint(filepath, self.model)
        self.enc_optim = torch.optim.Adam(self.model.enc.parameters(), lr = LEARNING_RATE)
        self.dec_optim = torch.optim.Adam(self.model.dec.parameters(), lr = LEARNING_RATE)   
        self.crf_optim = torch.optim.Adam(self.model.crf.parameters(), lr = LEARNING_RATE * 2)  
        epoch = 0  

        
        for step in range(num_epochs):
            weight_file = "{}/crf_{}.pth".format(weight_path,step)
            timer = time()
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            train_loss = self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            scores = self.test_epoch(val_loader)
            timer = time() - timer
            self.logger.info("train %.6f, valid %.6f" % (train_loss, scores))
            save_checkpoint(weight_file, self.model, step, train_loss, timer)

    def predict(self, dataset: TFTDataset):
        device = torch.device("cuda:{}".format(self.gpus)) 
        self.device = device
        self.model = SeqCrf(
            hidden_size=self.optargs['hidden_size'],
            input_size=self.optargs["input_size"],
            step_len=self.optargs["step_len"],
            viz=self.optargs['viz'],
            device=device
            
        )     
        weight_path = self.optargs['weight_path']        
        if self.optargs['load_weights']:
            step = self.optargs["best_ckpt_no"]
            filepath = "{}/crf_{}.pth".format(weight_path,step)
            epoch = load_checkpoint(filepath, self.model)        
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 生成tft时间序列训练数据集
        ts_data_train = dataset.get_crf_dataset(df_train)
        train_loader = ts_data_train.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)  
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = df_valid[df_valid['instrument'].isin(df_train['instrument'].unique())]
        validation = dataset.get_crf_dataset(df_valid,mode="valid")
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)           
        acc = self.test_epoch(val_loader)
        print("test acc:",acc)
                
    def train_epoch(self, data_loader):
        self.model.train()
        loss_sum = 0       
        step_number = len(data_loader.dataset) // self.batch_size    
        # print("data length:{},step_number:{}".format(len(data_loader.dataset),step_number))
        step_idx = 0
        for (x,y) in data_loader:
            loss = self.model(x,y) # forward pass and compute loss
            # self.logger.info("step:{}/{},training step,loss:{}".format(step_idx,step_number,loss))
            loss.backward() # compute gradients
            self.enc_optim.step() # update encoder parameters
            self.dec_optim.step() # update decoder parameters
            self.crf_optim.step()
            loss_sum += loss.item()   
            step_idx = step_idx + 1
        return loss_sum
        
    def test_epoch(self, data_loader):

        self.model.eval()

        acc_total = None
        losses = []
        
        for (x,y)  in data_loader:
            score,tag_seq = self.model.val(x, y)
            pred,target = self.clean_tag_data(tag_seq,y[0])
            acc = np.sum(pred == target)/(target.shape[0]*target.shape[1])
            acc = np.array([acc])
            if acc_total is None:
                acc_total = acc
            else:
                acc_total = np.concatenate((acc_total,acc))
        return np.mean(acc_total)
    
    def clean_tag_data(self,pred,target):
        pred_new = []
        target_new = []
        for index,item in enumerate(pred):
            if len(item)==5:
                pred_new.append(item)
                target_new.append(target[index].numpy())
        return np.array(pred_new),np.array(target_new)
