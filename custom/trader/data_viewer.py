import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.animation as animation

from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.tensor_viz import TensorViz

matplotlib.use('TkAgg')

class DataViewer():
    """数据的图像化工具"""
    
    def __init__(self,env_name=None):
        
        if env_name is None:
            env_name = "stat_pred_classify"
        self.viz_input = TensorViz(env=env_name) 
        style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 8})
        fig = mpf.figure(figsize=(18, 12), style=style)
        
    def show_single_complex_pred_data(self,single_data,correct=0,dataset=None,save_path=None):
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
            mpf.make_addplot(target_data["MACD"],ylabel='MACD',panel=2,color='dimgray',secondary_y=False,y_on_right=True),
            mpf.make_addplot(target_data["RSI20"],ylabel='RSI20',panel=3,color='fuchsia',secondary_y=False,y_on_right=True),
        ]
        title = "date:{},instrument:{},correct:{}".format(target_data["pred_date"].values[0],target_data["instrument"].values[0],correct)
        # 绘制图形，包括均线和成交量
        correct_path = "correct" if correct==2 else "incorrect"
        file_path = save_path + "/" + correct_path + "/" + str(target_data["pred_date"].values[0]) + "_" + str(target_data["instrument"].values[0]) +".png"
        mav = (5, 10, 20)
        mav = ()
        mpf.plot(target_data, title=title,type='candle',addplot=apds, mav=mav,volume=True,savefig=file_path,figsize=(12, 8))
        # figsize = (18, 12)
        # plt.figure(figsize=figsize)
        # plt.legend()
        # mpf.show()
        
    def show_single_complex_pred_data_visdom(self,single_data,correct=0,dataset=None,ytick_range=0.05,cut_target=False):
        
        instrument = int(single_data["instrument"].values[0])
        date = single_data["pred_date"].values[0]
        class1 = single_data["class1"].values[0]
        class2 = single_data["class2"].values[0]
        vr_class = single_data["vr_class"].values[0]
        names = ["pred_data","label","label_ori"]        
        target_title = "{}_{}_{}-{}/{},correct:{}".format(date,instrument, class1,class2,vr_class,correct)
        pred_line = single_data["pred_data"].values
        label_line = single_data["label"].values
        price_line = single_data["label_ori"].values
        if cut_target:
            label_line[dataset.pred_len+1:] = 0
            price_line[dataset.pred_len+1:] = 0
        view_data = np.stack((pred_line,label_line,price_line),axis=0).transpose(1,0)
        x_range = np.arange(single_data["time_idx"].values[0],single_data["time_idx"].values[-1]+1)
        ymax = view_data[view_data>0].max()*(1+ytick_range)
        ymin = view_data[view_data>0].min()*(1-ytick_range)
        self.viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,x_range=x_range,names=names,ytickmax=ymax,ytickmin=ymin)     
        
    def market_price(self,df_ref,date_range,instrument,dataset=None):
        """显示某个股票在指定日期范围内的行情走势"""
        
        time_column = dataset.get_time_column()
        group_column = dataset.get_group_column()
        
        time_begin = date_range[0]
        time_end = date_range[1]
        df_item = df_ref[(df_ref[time_column]>=time_begin)&(df_ref[time_column]<time_end)&(df_ref[group_column]==instrument)]
        x_range = df_item["time_idx"].values
        target_title = "{}".format(instrument)
        view_data = df_item["label_ori"].values
        self.viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,x_range=x_range)  
        
        
                      