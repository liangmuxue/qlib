import numpy as np
    
def slope_status(slope_arr,mea_sum=3,threhold=1):
    """均线形态判断，0：上升 1：下降 2：平缓 3：波动 
        mea_sum 标志至少有几条线段需要满足同一要求
        threhold 幅度阈值，单位为百分点
    """
    
    if isinstance(slope_arr,list):
        slope_arr = np.array(slope_arr)
    if np.sum(np.absolute(slope_arr)<threhold)>=mea_sum:
        return 2
    if np.sum(slope_arr>threhold)>=mea_sum:
        return 0        
    if  np.sum(slope_arr<-threhold)>=mea_sum:
        return 1        
    return 3    