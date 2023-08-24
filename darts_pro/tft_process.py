# 使用darts架构的TFT模型，定制化numpy数据集模式

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
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from darts.metrics import mape

from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model

from darts_pro.data_extension.batch_dataset import BatchDataset
from darts_pro.tft_series_dataset import TFTSeriesDataset
import cus_utils.global_var as global_var
from darts_pro.data_extension.custom_tcn_model import ClassifierTrainer

class TftDatafAnalysis():
    
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
        # 业务参数部分
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
        self.kwargs = kwargs
        
        global_var._init()

    def fit(
        self,
        dataset: TFTSeriesDataset,
    ):
        dataset.build_series_data(no_series_data=True)
        global_var.set_value("dataset", dataset)  
        if self.type.startswith("data_pca"):
            self.data_pca(dataset)
        if self.type.startswith("data_lstm"):
            self.data_lstm(dataset)
                                        
    def data_pca(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行主成分分析"""
         
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_batch.pickel".format(batch_file_path)
        col_list = dataset.col_def["col_list"]
        col_list.remove("label_ori")
        col_list.remove("REV5_ORI")
        col_list = ["CCI5"]
        ds = BatchDataset(batch_file,fit_names=col_list)
        ret_file = "{}/pca_ret_cci.npy".format(batch_file_path)
        ds.analysis_df_pca(fit_names=col_list,range_num=3000,ret_file=ret_file)     
        
    def data_lstm(
        self,
        dataset: TFTSeriesDataset,
    ):
        """对数据进行主成分分析"""
         
        batch_file_path = self.kwargs["batch_file_path"]
        batch_file = "{}/train_part_batch.pickel".format(batch_file_path)
        col_list = dataset.col_def["col_list"]
        col_list.remove("label_ori")
        col_list.remove("REV5_ORI")
        col_list = ["CCI5"]
        train_ds = BatchDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[0,10000])
        valid_ds = BatchDataset(batch_file,fit_names=col_list,mode="analysis",range_num=[10000,12000])
        trainer = ClassifierTrainer(train_ds,valid_ds,input_dim=len(col_list))
        trainer.training()
                