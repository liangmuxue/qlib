import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.animation as animation

from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.tensor_viz import TensorViz

class DataViewer():
    """数据的图像化工具"""
    
    def __init__(self):
        pass
        
    def show_single_complex_pred_data(self,single_data,dataset=None,save_path=None):
        """预测数据单条显示"""
        
        target_data = single_data
        target_data = target_data.set_index("datetime")
        target_data.index.name = "Time Index"
        # 修改为mpf的标准字段名
        target_data = target_data.rename(columns={'OPEN':'open','CLOSE':'close','HIGH':'high','LOW':'low','VOLUME_CLOSE':'volume'})
        # 附加数据
        apds = [
            # 标签数据
            mpf.make_addplot(target_data["label"],panel=0,color='lime',secondary_y=False),
            # 预测数据
            mpf.make_addplot(target_data["pred_data"],panel=0,color='c',secondary_y=True),                  
            # 附加指标数据
            mpf.make_addplot(target_data["MACD"],panel=2,color='dimgray',secondary_y=False),
            mpf.make_addplot(target_data["RSI20"],panel=3,color='fuchsia',secondary_y=False),
        ]
        title = "date:{},instrument:{}".format(target_data["pred_date"].values[0],target_data["instrument"].values[0])
        # 绘制图形，包括均线和成交量
        file_path = save_path + "/" + str(target_data["instrument"].values[0]) + ".png"
        mav = (5, 10, 20)
        mav = ()
        mpf.plot(target_data, title=title,type='candle',addplot=apds, mav=mav,volume=True,savefig=file_path)
        figsize = (9, 6)
        plt.figure(figsize=figsize)
        plt.legend()
        mpf.show()
        plt.clf()  
        
    def show_single_complex_pred_data_visdom(self,single_data,dataset=None,cut_target=False):
        viz_input = TensorViz(env="stat_pred_classify") 
        
        instrument = int(single_data["instrument"].values[0])
        date = single_data["pred_date"].values[0]
        class1 = single_data["class1"].values[0]
        class2 = single_data["class2"].values[0]
        vr_class = single_data["vr_class"].values[0]
        names = ["pred_data","label","label_ori"]        
        target_title = "{}_{}_{}-{}/{}".format(date,instrument, class1,class2,vr_class)
        pred_line = single_data["pred_data"].values
        label_line = single_data["label"].values
        price_line = single_data["label_ori"].values
        if cut_target:
            label_line[dataset.pred_len+1:] = 0
            price_line[dataset.pred_len+1:] = 0
        view_data = np.stack((pred_line,label_line,price_line),axis=0).transpose(1,0)
        x_range = np.arange(single_data["time_idx"].values[0],single_data["time_idx"].values[-1]+1)
        viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,x_range=x_range,names=names)     
        
                      