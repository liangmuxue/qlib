import numpy as np

import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
import torchmetrics
from torchmetrics import MeanSquaredError

from darts import TimeSeries
from darts.logging import get_logger, raise_if_not, raise_log

from typing import Callable, Optional, Sequence, Tuple, Union
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.log_util import AppLogger
logger = AppLogger()

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
    
    
    
    