from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet

from copy import copy as _copy, deepcopy
from functools import lru_cache
import inspect
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
import torch
from torch.distributions import Beta
from torch.nn.utils import rnn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)

from cus_utils.common_compute import np_qcut,enhance_data_complex
from cus_utils.tensor_viz import TensorViz
from cus_utils.data_aug import outliers_proc
from tft.class_define import CLASS_VALUES, VALUE_BINS, EOS_IDX, PAD_IDX
from sqlalchemy.dialects.mysql import enumerated

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]

def target_normalize(data):
    mu = np.mean(data[:,-1,-1])
    std = np.std(data[:,-1,-1])
    data[:,-1,-1] = (data[:,-1,-1]-mu)/std
    return data

class TimeSeriesNumpyDataset(TimeSeriesDataSet):
    """实现numpy数据直接读取的数据集功能"""
    
    def __init__(self, 
                data,
                time_idx: str,
                target: Union[str, List[str]],
                group_ids: List[str],
                weight: Union[str, None] = None,
                max_encoder_length: int = 30,
                min_encoder_length: int = None,
                min_prediction_idx: int = None,
                min_prediction_length: int = None,
                max_prediction_length: int = 1,
                data_columns=[],
                static_categoricals: List[str] = [],
                static_reals: List[str] = [],
                time_varying_known_categoricals: List[str] = [],
                time_varying_known_reals: List[str] = [],
                time_varying_unknown_categoricals: List[str] = [],
                time_varying_unknown_reals: List[str] = [],
                variable_groups: Dict[str, List[int]] = {},
                constant_fill_strategy: Dict[str, Union[str, float, int, bool]] = {},
                allow_missing_timesteps: bool = False,
                lags: Dict[str, List[int]] = {},
                add_relative_time_idx: bool = False,
                add_target_scales: bool = False,
                add_encoder_length: Union[bool, str] = "auto",
                target_normalizer: Union[NORMALIZER, str, List[NORMALIZER], Tuple[NORMALIZER]] = "auto",
                categorical_encoders: Dict[str, NaNLabelEncoder] = {},
                scalers: Dict[str, Union[StandardScaler, RobustScaler, TorchNormalizer, EncoderNormalizer]] = {},
                randomize_length: Union[None, Tuple[float, float], bool] = False,
                predict_mode: bool = False,
                qcut_len: int = 15,  
                viz: bool = False               
        ):
        # 训练集测试集的比例为7比3
        SP_RATIO = 0.7
        sp = int(data.shape[0] * SP_RATIO)
        self.predict_mode = predict_mode
        # 根据类型切分数据
        if not predict_mode:
            self.data = data[:sp,:,:]
        else:
            self.data = data[sp:,:,:]  
        # 需要对label数值进行scale
        self.data = target_normalize(self.data)  
        # 直接使用上级父类进行初始化
        Dataset.__init__(self)
        
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length
        self.add_encoder_length = add_encoder_length
        self.lags = lags

        self.data_columns = data_columns + ["ori_label"]
        self.target = target
        self.weight = weight
        self.time_idx = time_idx
        self.group_ids = [] + group_ids
        self.static_categoricals = [] + static_categoricals
        self.static_reals = [] + static_reals
        self.time_varying_known_categoricals = [] + time_varying_known_categoricals
        self.time_varying_known_reals = [] + time_varying_known_reals
        self.time_varying_unknown_categoricals = [] + time_varying_unknown_categoricals
        self.time_varying_unknown_reals = [] + time_varying_unknown_reals
        self.add_relative_time_idx = add_relative_time_idx
        self.viz = viz
        self.qcut_len = qcut_len
        
        # set automatic defaults
        if isinstance(randomize_length, bool):
            if not randomize_length:
                randomize_length = None
            else:
                randomize_length = (0.2, 0.05)
        self.randomize_length = randomize_length

        self.min_prediction_idx = min_prediction_idx
        self.constant_fill_strategy = {} if len(constant_fill_strategy) == 0 else constant_fill_strategy
        self.predict_mode = predict_mode
        self.allow_missing_timesteps = allow_missing_timesteps
        self.target_normalizer = target_normalizer
        self.categorical_encoders = {} if len(categorical_encoders) == 0 else categorical_encoders
        self.scalers = {} if len(scalers) == 0 else scalers
        self.add_target_scales = add_target_scales
        self.variable_groups = {} if len(variable_groups) == 0 else variable_groups
  
        # visdom可视化环境
        if self.viz:
            self.viz_cat = TensorViz(env="dataview_cat")
            self.viz_cont = TensorViz(env="dataview_cont") 
 
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """定制单个数据获取方式"""
        
        row = self.data[idx]
        pd_data = pd.DataFrame(row,columns=self.data_columns)
        p_targets = torch.from_numpy(row[self.max_encoder_length:])[:,-1]
        encoder_target = torch.from_numpy(row[:self.max_encoder_length])
        cat_columns = self.static_categoricals + self.time_varying_known_categoricals
        cont_columns = self.time_varying_known_reals + self.time_varying_unknown_reals
        # 通过列明分别生成离散数据和连续数据
        data_cat = torch.from_numpy(pd_data[cat_columns].values).int()
        data_cont = torch.from_numpy(pd_data[cont_columns].values).float()
        groups = torch.from_numpy(pd_data[self.group_ids].values[0].astype(np.int64))
        return (
            dict(
                x_cat=data_cat,
                x_cont=data_cont,
                encoder_length=self.max_encoder_length,
                decoder_length=1,
                encoder_target=encoder_target,
                encoder_time_idx_start=0,
                groups=groups,
                target_scale=np.array([0,10.6]),                
            ),
            (p_targets, None),
        )      
    
 
        
    def __len__(self) -> int:
        return self.data.shape[0]
            
    def classify_target(self,target):
        """把目标数值进行分类"""
        p_taragets = []
        for item in target:
            p_taraget = [k for k, v in CLASS_VALUES.items() if (item>=v[0] and item<v[1])]
            p_taragets.append(p_taraget)
        # 添加結束標誌
        # p_taragets.append([EOS_IDX])
        p_taragets = torch.Tensor(p_taragets).squeeze(-1).long()        
        return p_taragets

        
        