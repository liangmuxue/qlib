import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.animation as animation

from darts_pro.data_extension.series_data_utils import get_pred_center_value
from cus_utils.tensor_viz import TensorViz
import cus_utils.global_var as global_var

matplotlib.use('TkAgg')

class DataViewer():
    """数据的图像化工具"""
    
    def __init__(self,env_name=None):
        
        if env_name is None:
            env_name = "stat_pred_classify"

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
        if correct<0:
            file_path = save_path + "/" + str(target_data["pred_date"].values[0]) + "_" + str(target_data["instrument"].values[0]) +".png"
        else:
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
        """visdom模式，显示预测数据与实际数据对比"""
        
        instrument = int(single_data["instrument"].values[0])
        date = single_data["pred_date"].values[0]
        vr_class = single_data["vr_class"].values[0]
        names = ["pred_data","label","label_ori"]        
        target_title = "{}_{}/{},correct:{}".format(date,instrument,vr_class,correct)
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

    def show_trades_data_visdom(self,stat_df,pred_df,ytick_range=0.05):
        """visdom模式，显示回测数据与实际数据对比"""
        
        instrument = int(pred_df["instrument"].values[0])
        date = pred_df["pred_date"].values[0]
        class1 = pred_df["class1"].values[0]
        class2 = pred_df["class2"].values[0]
        vr_class = pred_df["vr_class"].values[0]
        names = ["pred_data","label","label_ori"]        
        target_title = "{}_{}/differ:{},duration:{}".format(date,instrument,round(stat_df["differ_range"],2),stat_df["duration"])
        pred_line = pred_df["pred_data"].values
        label_line = pred_df["label"].values
        price_line = pred_df["label_ori"].values
        view_data = np.stack((pred_line,label_line,price_line),axis=0).transpose(1,0)
        x_range = np.arange(pred_df["time_idx"].values[0],pred_df["time_idx"].values[-1]+1)
        ymax = view_data[view_data>0].max()*(1+ytick_range)
        ymin = view_data[view_data>0].min()*(1-ytick_range)
        self.viz_input.viz_matrix_var(view_data,win=target_title,title=target_title,x_range=x_range,names=names,ytickmax=ymax,ytickmin=ymin)
                
    def market_price(self,df_ref,date_range,instrument,dataset=None,viz_target=None,att_cols=[]):
        """显示某个股票在指定日期范围内的行情走势"""
        
        time_column = dataset.get_time_column()
        time_column = "datetime_number"
        group_column = dataset.get_group_column()
        
        time_begin = date_range[0]
        time_end = date_range[1]
        df_item = df_ref[(df_ref[time_column]>=time_begin)&(df_ref[time_column]<time_end)&(df_ref[group_column]==instrument)]
        if df_item.shape[0]==0:
            print("no data!")
            return
        x_range = df_item[time_column].values
        target_title = "{}".format(instrument)
        names = ["price"]
        view_data = np.expand_dims(df_item["label_ori"].values,-1)
        for col in att_cols:
            att_view_data = np.expand_dims(df_item[col].values,-1)
            view_data = np.concatenate([view_data,att_view_data],-1)
            names = names + [col]
        viz_target.viz_matrix_var(view_data,win=target_title,title=target_title,names=names,x_range=x_range)  
        
    def market_price_mpl(self,df_ref,date_range,instrument,dataset=None,save_path=None):

        time_column = dataset.get_time_column()
        time_column = "datetime_number"
        group_column = dataset.get_group_column()
        
        time_begin = date_range[0]
        time_end = date_range[1]
               
        df_item = df_ref[(df_ref[time_column]>=time_begin)&(df_ref[time_column]<time_end)&(df_ref[group_column]==instrument)]
        if df_item.shape[0]==0:
            print("no data!")
            return        
        target_data = df_item
        target_data = target_data.set_index("datetime")
        target_data.index.name = "Time Index"
        # 修改为mpf的标准字段名
        target_data = target_data.rename(columns={'OPEN':'open','CLOSE':'close','HIGH':'high','LOW':'low','VOLUME_CLOSE':'volume'})
        title = "date:{},instrument:{}".format(time_begin,instrument)
        mav = (5, 10, 20)
        mav = ()
        apds = []
        file_path = os.path.join(save_path,"{}.png".format(instrument))
        mpf.plot(target_data, title=title,type='candle',addplot=apds, mav=mav,volume=True,savefig=file_path,figsize=(12, 8))        
                      