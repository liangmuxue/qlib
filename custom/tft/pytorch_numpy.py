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
from .tuning_numpy import OptimizeHyperparameters
from tft.tft_dataset import TFTDataset

class NumpyModel(Model):
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
        # 生成tft时间序列训练数据集
        file_path = "/home/qdata/project/qlib/custom/data/aug/test100_more.npy"
        ts_data_train = dataset.get_numpy_dataset(file_path,mode="train")
        train_loader = ts_data_train.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)
        validation = dataset.get_numpy_dataset(file_path,mode="valid")
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)       
        self.study_opt = OptimizeHyperparameters(
            train_loader,
            val_loader,
            model_path=self.optargs['weight_path'],
            n_trials=self.optargs["n_trials"],
            max_epochs=self.optargs["max_epochs"],
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            hidden_continuous_size_range=(8, 128),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=0.1,log_every_n_steps=16,gpus=self.gpus),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            load_weights=self.optargs['load_weights'],
            trial_no = self.optargs['best_trial_no'],
            epoch_no = self.optargs['best_ckpt_no'],            
            log_dir=self.optargs['log_path'],
            viz=self.optargs['viz'],
            gpus=self.gpus
            
        )                       
        # 根据标志决定优化学习,还是使用优化好的参数直接训练
        if self.type=="opt_train":
            study = self.study_opt.do_study()        
            with open("custom/data/test_study.pkl", "wb") as fout:
                pickle.dump(study, fout)      
        # 使用已经生成的最优化参数进行训练                    
        if self.type=="best_train":
            # 根据参数决定是否加载之前训练的权重
            if self.optargs['load_weights']:
                self.study_opt.trial_no = self.optargs['best_trial_no']
                self.study_opt.epoch_no = self.optargs['best_ckpt_no']            
            study = self.study_opt.do_apply("custom/data/test_study.pkl")      
        
    def predict(self, dataset: TFTDataset):
        # 只预测部分数据
        if self.type=="pred_sub":
            return self.predict_subset(dataset) 
       
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        test_ds = dataset.get_ts_dataset(df_test,mode="valid")
        test_loader = test_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)        
        best_tft = OptimizeHyperparameters(clean_mode=True,model_path=self.optargs['weight_path'],load_weights=True,
                                           trial_no=self.optargs['best_trial_no'],
                                           epoch_no=self.optargs['best_ckpt_no']).get_tft(fig_save_path=self.fig_save_path,viz=self.optargs['viz'])
        actuals = torch.cat([y[0] for x, y in iter(test_loader)])
        predictions, x, pred_ori = best_tft.predict(test_loader,return_x=True,return_ori_outupt=True)   
        loss = (actuals - predictions).abs().mean()
        print("loss is:",loss)        
        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);  
        return pd.Series(np.concatenate(predictions), index=df_test.get_index())
    
    def predict_subset(self, dataset: TFTDataset):
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = df_valid[df_valid['instrument'].isin(df_train['instrument'].unique())]
        validation = dataset.get_ts_dataset(df_valid,mode="valid")
        val_loader = validation.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1)  
        trial_no = self.optargs['best_trial_no']
        epoch_no = self.optargs['best_ckpt_no']
        ohp = OptimizeHyperparameters(clean_mode=True,model_path=self.optargs['weight_path'],load_weights=True,
                                           trial_no=trial_no,
                                           epoch_no=epoch_no)
        best_tft = ohp.get_tft(fig_save_path=self.fig_save_path,viz=self.optargs['viz'])
        # 筛选出部分数据进行预测
        # subset = test_ds.filter(lambda x: (x.instrument == "600010"))
        # test_loader_syb = test_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1) 
        # actuals_sub = torch.cat([y[0] for x, y in iter(test_loader_syb)])
        # test_loader = subset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=1) 
        actuals = torch.cat([y[0] for x, y in iter(val_loader)])
        predictions, x, pred_ori = best_tft.predict(val_loader,return_x=True,return_ori_outupt=True)   
        loss = (actuals - predictions).abs().mean()
        print("loss is:",loss)
        self.cal_range(predictions,actuals)
        # self.show_pred_act(predictions,pred_ori,actuals)
        raw_predictions, x = best_tft.predict(val_loader, mode="raw", return_x=True)
        # self.show_worst(best_tft,val_loader,x,raw_predictions,actuals)
        # for idx in range(0,100,5):  # plot 10 examples
        #     best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);        
        # subset.calculate_prediction_oridata(subset,predictions=predictions)
        predictions, x = best_tft.predict(validation, mode="quantiles",return_x=True)  
        predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
        best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);  
        return pd.Series(np.concatenate(predictions), index=df_valid.get_index())        
    
    def cal_range(self,predictions,actuals):
        """计算区间差值的准确度"""
        pred_se_range = (predictions[:,4] - predictions[:,0]) / predictions[:,0]
        pred_se_range_sort,idx = torch.sort(pred_se_range,descending=True, dim=-1)
        act_range = actuals[idx]
        act_se_range = (actuals[:,4] - actuals[:,0]) / actuals[:,0]
        act_se_range_sort,act_idx = torch.sort(act_se_range,descending=True, dim=-1)
        print("pred idx:{} and act idx:{}".format(idx,act_idx))
    
    def show_worst(self,best_tft,val_dataloader,x,raw_predictions,actuals):
        predictions = best_tft.predict(val_dataloader)
        mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
        indices = mean_losses.argsort(descending=True)  # sort losses
        for idx in range(10):  # plot 10 examples
            best_tft.plot_prediction(
                x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
            );
            print("ctn")   
             
    def show_pred_act(self,predictions,pred_ori,actuals):
        from cus_utils.tensor_viz import TensorViz
        viz_compare = TensorViz(env="dataview_compare")
        
        length = actuals.shape[0]
        for i in range(length):
            actual = actuals[i]
            prediction = predictions[i]
            compare_matrix = torch.cat([actual.unsqueeze(-1),prediction.unsqueeze(-1)],1)   
            viz_compare.viz_matrix_var(compare_matrix,win="comp_{}".format(i),names=["actual","prediction"])
            names=["pred_{}".format(i) for i in range(7)]
            win_target = "comp_quant_" + str(i)            
            viz_compare.viz_matrix_var(pred_ori[i],win=win_target,names=names)
            continue
