import numpy as np

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torchmetrics
from torchmetrics import MeanSquaredError

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not, raise_log

from typing import Callable, Optional, Sequence, Tuple, Union
from darts_pro.data_extension.series_data_utils import get_pred_center_value

from cus_utils.common_compute import target_scale,slope_classify_compute
from cus_utils.log_util import AppLogger
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES,get_simple_class

logger = AppLogger()

def vr_acc_compute(actual_series, pred_series, vr_class,intersect: bool = True):
    """涨跌幅度类别的准确率计算"""
    
    # 通过预测系列，取得实际目标数据
    y_true, y_hat = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    raise_if_not(
        (y_true != 0).all(),
        "The actual series must be strictly positive to compute the vr_acc_compute.",
        logger,
    )
    
    # 根据实际数据取得涨跌幅类别
    vr_target = value_range_class_compute(y_true)    
    # 返回结果中包含目标分类值，用于后续进行重点类统计
    rtn_flag = [0,vr_target]
    if vr_target==vr_class.item():
        rtn_flag[0] = 1
    else:
        rtn_flag[0] = 0
    return rtn_flag

def value_range_class_compute(value_arr):
    """根据走势数据计算涨跌幅类别"""
    
    max_value = np.max(value_arr)
    min_value = np.min(value_arr)
    
    begin_value = value_arr[0]
    # 比较最大上涨幅度与最大下跌幅度，从而决定幅度范围正还是负
    if max_value - begin_value > begin_value - min_value:
        raise_range = (max_value -begin_value)/begin_value * 100
    else:
        raise_range = (min_value -begin_value)/begin_value * 100    
        
    p_taraget_class = get_simple_class(raise_range)
    return p_taraget_class
    
def cel_compute(actual_series, pred_series, pred_class,intersect: bool = True):
    """走势分类交叉熵距离计算"""
    
    # 通过预测系列，取得实际目标数据
    y_true, y_hat = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    raise_if_not(
        (y_true != 0).all(),
        "The actual series must be strictly positive to compute the cel_compute.",
        logger,
    )
    # 目标数据生成实际分类数据
    class1_len = int((y_true.shape[0]-1)/2)
    target_class, target_scaler = slope_classify_compute(np.expand_dims(y_true,axis=-1),class1_len)
    # 使用交叉熵衡量分类差距
    cross_metric = compute_cross_metrics(torch.unsqueeze(pred_class,0), torch.unsqueeze(torch.tensor(target_class),0))
    return cross_metric.item()

def cel_acc_compute(actual_series, pred_series, pred_class,intersect: bool = True):
    """走势分类准确度计算"""
    
    # 通过预测系列，取得实际目标数据
    y_true, y_hat = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    raise_if_not(
        (y_true != 0).all(),
        "The actual series must be strictly positive to compute the cel_compute.",
        logger,
    )
    # 目标数据生成实际分类数据
    class1_len = int((y_true.shape[0]-1)/2)
    target_class, target_scaler = slope_classify_compute(np.expand_dims(y_true,axis=-1),class1_len)
    # 与预测分类数据进行比较
    acc_score = 0
    if target_class[0]==pred_class[0].item():
        acc_score += 0.5
    if target_class[1]==pred_class[1].item():
        acc_score += 0.5        
    return acc_score    
    
def compute_cross_metrics(output_class,target_class):
    cross_loss = nn.CrossEntropyLoss()(output_class,target_class)
    return cross_loss 

def compute_vr_metrics(output_class,target_class):
    value_range_loss = nn.CrossEntropyLoss()(output_class,target_class)  
    return value_range_loss
    
def corr_dis(actual_series, pred_series, intersect: bool = True,pred_len=5):
    
    y_true, y_hat = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    raise_if_not(
        (y_true != 0).all(),
        "The actual series must be strictly positive to compute the MAPE.",
        logger,
    )
    corr_obj = torchmetrics.PearsonCorrCoef()
    corr = corr_obj(torch.tensor(y_hat),torch.tensor(y_true))
    return corr.item()

def diff_dis(actual_series, pred_series, intersect: bool = True,pred_len=5):
    
    y_true, y_hat = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )
    raise_if_not(
        (y_true != 0).all(),
        "The actual series must be strictly positive to compute the MAPE.",
        logger,
    )
    mse_metric = MeanSquaredError()
    output_be = y_hat[[0,-1]]
    target_be = y_true[[0,-1]]    
    diff = mse_metric(torch.tensor(output_be), torch.tensor(target_be))
    return diff.item()

def series_target_scale(target_series,scaler=None):
    """使用训练中生成的scaler，对目标系列数据进行缩放"""
    
    target_df = target_series.pd_dataframe()
    # pred_center_data = get_pred_center_value(pred_series)
    target_value = target_df["label"].values
    target_value = scaler.transform(np.expand_dims(target_value,axis=-1))
    target_df["label"] = target_value.squeeze(-1)
    rtn_seriees = TimeSeries.from_dataframe(target_df,
                                             fill_missing_dates=True,
                                             value_cols="label")      
    return rtn_seriees
    
def _get_values(
    series: TimeSeries, stochastic_quantile: Optional[float] = 0.5
) -> np.ndarray:
    """
    Returns the numpy values of a time series.
    For stochastic series, return either all sample values with (stochastic_quantile=None) or the quantile sample value
    with (stochastic_quantile {>=0,<=1})
    """
    if series.is_deterministic:
        series_values = series.univariate_values()
    else:  # stochastic
        if stochastic_quantile is None:
            series_values = series.all_values(copy=False)
        else:
            series_values = series.quantile_timeseries(
                quantile=stochastic_quantile
            ).univariate_values()
    return series_values
    
def _get_values_or_raise(
    series_a: TimeSeries,
    series_b: TimeSeries,
    intersect: bool,
    stochastic_quantile: Optional[float] = 0.5,
    remove_nan_union: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.

    Raises a ValueError if the two time series (or their intersection) do not have the same time index.

    Parameters
    ----------
    series_a
        A univariate deterministic ``TimeSeries`` instance (the actual series).
    series_b
        A univariate (deterministic or stochastic) ``TimeSeries`` instance (the predicted series).
    intersect
        A boolean for whether or not to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, remove all indices from `series_a` and `series_b` which have a NaN value
        in either of the two input series.
    """

    raise_if_not(
        series_a.width == series_b.width,
        "The two time series must have the same number of components",
        logger,
    )

    raise_if_not(isinstance(intersect, bool), "The intersect parameter must be a bool")

    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b

    raise_if_not(
        series_a_common.has_same_time_as(series_b_common),
        "The two time series (or their intersection) "
        "must have the same time index."
        "\nFirst series: {}\nSecond series: {}".format(
            series_a.time_index, series_b.time_index
        ),
        logger,
    )

    series_a_det = _get_values(series_a_common, stochastic_quantile=stochastic_quantile)
    series_b_det = _get_values(series_b_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union:
        return series_a_det, series_b_det

    b_is_deterministic = bool(len(series_b_det.shape) == 1)
    if b_is_deterministic:
        isnan_mask = np.logical_or(np.isnan(series_a_det), np.isnan(series_b_det))
    else:
        isnan_mask = np.logical_or(
            np.isnan(series_a_det), np.isnan(series_b_det).any(axis=2).flatten()
        )
    return np.delete(series_a_det, isnan_mask), np.delete(
        series_b_det, isnan_mask, axis=0
    )
    
    
    
    