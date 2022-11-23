def get_pred_center_value(series):
    """取得区间范围的中位数数值"""
    
    comp = series.data_array().sel(component="label")
    central_series = comp.quantile(q=0.5, dim="sample")
    return central_series