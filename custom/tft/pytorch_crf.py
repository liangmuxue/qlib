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
from tft.tft_dataset import TFTDataset
from custom_model.seq2seq_crf import SeqCrf
from custom_model.seq2seq import Seq2Seq
from custom_model.parameters import *
from tft.class_define import CLASS_VALUES
from cus_utils.utils_crf import save_checkpoint,load_checkpoint
from cus_utils.visualization import VisUtil
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
        self.viz = self.optargs['viz']
        self.qcut_len = self.optargs['qcut_len']
    
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
        opt = {"qcut_len":self.qcut_len}
        ts_data_train = dataset.get_crf_dataset(df_train,opt=opt)
        train_loader = ts_data_train.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)
        # 生成tft验证集,并删除不在训练集中的股票数据
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = df_valid[df_valid['instrument'].isin(df_train['instrument'].unique())]
        validation = dataset.get_crf_dataset(df_valid,mode="valid",opt=opt)
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)      
        device = torch.device("cuda:{}".format(self.gpus)) 
        self.device = device
        if self.viz:
            self.viz_util = VisUtil()
        
        # 注意需要添加一个结束符号长度, 并且需要考虑跳过0，因此加2
        num_classes = len(CLASS_VALUES) + 2      
        self.model = Seq2Seq(
            hidden_size=self.optargs['hidden_size'],
            num_classes=num_classes,
            qcut_len=self.qcut_len,
            viz=self.viz,
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
        epoch = 0  

        
        for step in range(num_epochs):
            weight_file = "{}/crf_{}.pth".format(weight_path,step)
            timer = time()
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            train_loss = self.train_epoch(train_loader,epoch=step)
            self.logger.info("evaluating...")
            test_lose = self.test_epoch(val_loader,epoch=step)
            # acc = self.eval_epoch(val_loader,epoch=step)
            timer = time() - timer
            self.logger.info("train loss:%.6f, test loss:%.6f" % (train_loss, test_lose))
            save_checkpoint(weight_file, self.model, step, train_loss, timer)

    def predict(self, dataset: TFTDataset):
        device = torch.device("cuda:{}".format(self.gpus)) 
        self.device = device
        if self.viz:
            self.viz_util = VisUtil()
                    
        num_classes = len(CLASS_VALUES) + 2   
        self.model = Seq2Seq(
            hidden_size=self.optargs['hidden_size'],
            num_classes=num_classes,
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
        opt = {"qcut_len":self.qcut_len}
        ts_data_train = dataset.get_crf_dataset(df_train,opt=opt)
        train_loader = ts_data_train.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)  
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = df_valid[df_valid['instrument'].isin(df_train['instrument'].unique())]
        validation = dataset.get_crf_dataset(df_valid,mode="valid",opt=opt)
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)           
        acc = self.test_epoch(val_loader)
        print("test acc:",acc)
                
    def train_epoch(self, data_loader,epoch=0):
        self.model.train()
        loss_sum = 0       
        step_number = len(data_loader.dataset) // self.batch_size    
        # print("data length:{},step_number:{}".format(len(data_loader.dataset),step_number))
        step_idx = 0
        for (x,y) in data_loader:
            loss,data = self.model(x,y) # forward pass and compute loss
            # self.logger.info("step:{}/{},training step,loss:{}".format(step_idx,step_number,loss))
            loss.backward() # compute gradients
            self.enc_optim.step() # update encoder parameters
            self.dec_optim.step() # update decoder parameters
            loss_sum += loss.item()   
            step_idx = step_idx + 1
            if step_idx%10==0:
                print("training step in:{}/{}:".format(step_idx,step_number))
            if self.viz:
                self.viz_util.viz_input_data(data,epoch=epoch,index=step_idx,type="training") 
                self.viz_util.viz_target_data(data,epoch=epoch,index=step_idx,loss=loss,loss_value=loss,type="training")                
        return loss_sum
        
    def test_epoch(self, data_loader,epoch=0):
        self.model.eval()
        loss_sum = 0       
        # print("data length:{},step_number:{}".format(len(data_loader.dataset),step_number))
        step_idx = 0
        for (x,y) in data_loader:
            loss,data = self.model(x,y) # forward pass and compute loss
            loss_sum += loss.item()   
            step_idx = step_idx + 1
            if self.viz:
                # self.viz_util.viz_input_data(data,epoch=epoch,index=step_idx,type="test") 
                self.viz_util.viz_target_data(data,epoch=epoch,index=step_idx,loss=loss,loss_value=loss,type="test")                        
        return loss_sum
        
    def eval_epoch(self, data_loader,epoch=0):

        self.model.eval()

        acc_total = None
        losses = []
        
        for (x,y)  in data_loader:
            results  = self.model.val(x, y)
            for item in results:
                (x0, y0, y1) = item
                pred,target = self.clean_tag_data(x0,y0)
                acc = np.sum(pred == target)/(target.shape[0]*target.shape[1])
                acc = np.array([acc])
                if acc_total is None:
                    acc_total = acc
                else:
                    acc_total = np.concatenate((acc_total,acc))
        return np.mean(acc_total)

    def test_epoch_crf(self, data_loader):

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
        if target.shape[0]<5:
            print("555")
        pred_new = []
        target_new = []
        for index,item in enumerate(pred):
            if len(item)==5:
                pred_new.append(item.cpu().numpy())
                target_new.append(target.cpu().numpy())
        return np.array(pred_new),np.array(target_new)
