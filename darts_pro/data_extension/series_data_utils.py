import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
import xarray as xr

def get_pred_center_value(series):
    """取得区间范围的中位数数值"""
    
    comp = series.data_array().sel(component="label")
    central_series = comp.quantile(q=0.5, dim="sample")
    return central_series

def get_np_center_value(np_data):
    """取得区间范围的中位数数值"""
    
    m = np.mean(np_data,axis=1)
    return m

def build_serices_with_ndarray(np_array):
    """根据numpy数组数据，生成时间序列对象"""
    
    times = [i for i in range(np_array.shape[0])]
    sample = [i for i in range(np_array.shape[1])]
    arr = xr.DataArray(np_array, coords=[times, sample], dims=['time_index', 'component', 'sample'],name="label")

    series = TimeSeries.from_xarray(arr)   
    return series 
