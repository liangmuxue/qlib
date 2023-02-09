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
from qlib.data.dataset import DatasetH, TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

from tft.tft_dataset import TFTDataset
from darts.utils.likelihood_models import QuantileRegression

from cus_utils.data_filter import DataFilter
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import random_int_list
from darts_pro.tft_series_dataset import TFTSeriesDataset

class BaseNumpyModel(Model):
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")
        
    def build_aug_data(self,dataset):
        save_file_path = self.optargs["save_path"]
        group_column = dataset.col_def["group_column"]
        target_column = dataset.col_def["target_column"]
        forecast_horizon = self.optargs["forecast_horizon"]
        wave_threhold = self.optargs["wave_threhold"]
        wave_window = self.optargs["wave_window"]
        wave_period = self.optargs["wave_period"]
        over_time = self.optargs["over_time"]
        wave_threhold_type = self.optargs["wave_threhold_type"]
        
        df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        # 股票代码(即分组字段)转换为数值型
        df[group_column] = df[group_column].apply(pd.to_numeric,errors='coerce')
        data_filter = DataFilter()
        # 使用前几天（根据预测长度而定）的移动平均值作为目标数值
        df[target_column]  = df.groupby(group_column)[target_column].shift(wave_window).rolling(window=wave_window,min_periods=1).mean()
        df = df.dropna().reset_index(drop=True)
        # 按照规则进行数据筛选
        wave_data = data_filter.filter_wave_data(df, target_column=target_column, group_column=group_column,forecast_horizon=forecast_horizon,wave_period=wave_period,
                                                 wave_threhold_type=wave_threhold_type,wave_threhold=wave_threhold,over_time=over_time)
        print("wave_data",wave_data.shape)
        np.save(save_file_path,wave_data)  
            
    def numpy_data_view(self,dataset,numpy_data,title="train_data",no_convi=False): 
        viz_input = TensorViz(env="data_hist") 
        wave_period = self.optargs["wave_period"]
        forecast_horizon = self.optargs["forecast_horizon"]     
        target_index = dataset.get_target_column_index()   
        start = wave_period - forecast_horizon
        value = numpy_data[:,start:,target_index]
        # 标签数据分布图
        viz_input.viz_data_hist(value.reshape(-1),numbins=10,win=title,title=title) 
        if no_convi:
            return
        columns = dataset.get_past_columns() 
        columns_index = dataset.get_past_column_index()
        # 随机取得某些数据，并显示折线图
        for i in random_int_list(1,numpy_data.shape[0]-1,3):
            # 标签数据折线图
            target_title= title + "_target_{}".format(i)
            view_data = np.expand_dims(numpy_data[i,:,target_index],axis=-1)
            viz_input.viz_matrix_var(view_data,win=target_title,title=target_title)              
            sub_title = title + "_line_{}".format(i)
            view_data = numpy_data[i,:,columns_index].transpose(1,0)
            viz_input.viz_matrix_var(view_data,win=sub_title,title=sub_title,names=columns)       
            
    def predict_show(self,val_series_list,pred_series_list,series_total):       
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99 
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"       
        forecast_horizon = self.optargs["forecast_horizon"]    
        # 整个序列比较多，只比较某几个序列
        figsize = (9, 6)
        # 创建比较序列，后面保持所有，或者自己指定长度
        actual_series_list = []
        for index,train_ser in enumerate(val_series_list):
            ser_total = series_total[index]
            # 从数据集后面截取一定长度的数据，作为比较序列
            actual_series = ser_total[
                ser_total.end_time() - (2 * forecast_horizon - 1) * ser_total.freq : 
            ]
            actual_series_list.append(actual_series)   
        mape_all = 0
        r = 3
        for i in range(len(val_series_list)):
            pred_series = pred_series_list[i]
            actual_series = actual_series_list[i]
            # 实际数据集的结尾与预测序列对齐
            actual_series[: pred_series.end_time()].plot(label="actual")
            # 分位数范围显示
            pred_series.plot(
                low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
            )
            pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
            # 与实际的数据集进行比较，比较的是两个数据集的交集
            mape_item = mape(actual_series, pred_series)
            mape_all = mape_all + mape_item
            if i<r:
                plt.figure(figsize=figsize)
                # 实际数据集的结尾与预测序列对齐
                pred_series.plot(label="forecast")            
                actual_series[: pred_series.end_time()].plot(label="actual")           
                plt.title("ser_{},MAPE: {:.2f}%".format(i,mape_item))
                plt.legend()
                plt.savefig('{}/result_view/eval_{}.jpg'.format(self.optargs["work_dir"],i))
                plt.clf()   
        mape_mean = mape_all/len(val_series_list)
        print("mape_mean:",mape_mean)        
            