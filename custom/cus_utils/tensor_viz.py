from visdom import Visdom
import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf

def test_normal_vis():
    viz = Visdom(env="debug",port=7098)
    t = torch.tensor([[1.85640e+02, 1.15424e+02, 7.67866e-01],
                      [1.78960e+02, 6.99677e+01, 8.03120e-01],
                      [1.89179e+02, 6.52888e+01, 4.34248e-01],
                      [1.70759e+02, 6.56513e+01, 6.35204e-01],
                      [2.02092e+02, 7.07454e+01, 7.88445e-01],
                      [1.61624e+02, 7.03727e+01, 3.44357e-01],
                      [2.19908e+02, 1.11385e+02, 6.91779e-01],
                      [1.44841e+02, 1.17850e+02, 7.47350e-01],
                      [2.29477e+02, 1.57608e+02, 4.85386e-01],
                      [1.26864e+02, 1.57121e+02, 2.61525e-01],
                      [1.97478e+02, 1.86200e+02, 7.40117e-02],
                      [1.55604e+02, 1.89440e+02, 3.51216e-01],
                      [2.09568e+02, 2.03195e+02, 6.91535e-01],
                      [1.67933e+02, 2.04309e+02, 4.71567e-01],
                      [2.12652e+02, 2.68038e+02, 6.51705e-01],
                      [1.53427e+02, 2.77876e+02, 6.35284e-01],
                      [2.20104e+02, 3.36257e+02, 6.50923e-01],
                      [1.55992e+02, 3.38123e+02, 4.83625e-01]], device='cpu', dtype=torch.float64)
    # viz.image(
    #     np.random.rand(3, 512, 256),
    #     opts=dict(title='Random!', caption='how random'),
    # )
    # viz.image(
    #     t.numpy(),
    #     opts=dict(title='t', caption='show numpy'),
    # )    
    # x = np.outer(np.arange(1, 6), np.arange(1, 11))
    # print(x)
    # # heatmap
    # viz.heatmap(
    #     X=x,
    #     opts=dict(
    #         columnnames=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    #         rownames=['y1', 'y2', 'y3', 'y4', 'y5'],
    #         colormap='Electric',
    #     )
    # )
    win = viz.line(X=np.arange(0, 10),Y=(np.linspace(5, 10, 10)))

class TensorViz(object):
    def __init__(self,env="debug",type="cont_data"):
        self.viz = Visdom(env=env,port=8098)  
        self.env = env
    
    def remove_env(self):
        self.viz.delete_env(self.env)
        
    def viz_data_bar(self,data,win="matrix",names=None,desc=None,title=None):     
        """绘制柱状比较图"""
        
        length = data.shape[0]
        rownames = ["t{}".format(i) for i in range(length)]
        self.viz.bar(X=data,
                     win=win,
                     opts=dict(
                        stacked=False,
                        legend=names,
                        rownames=rownames,
                        title=title,
                        ylabel='values',
                        xtickmin=0.4, 
                        xtickstep=0.4
                    )
            )               
        
    def viz_matrix_var(self,data,win="matrix",names=None,desc=None,title=None,x_range=None,ytickmin=None,ytickmax=None):
        length = data.shape[1]
        ts = data.shape[0]
        if x_range is None:
            x_range = np.arange(ts)    
        if desc is not None:    
            self.viz.text(desc)
        for i in range(length):
            if names is not None:
                line_name = "{}".format(names[i])
            else:
                line_name = "line{}".format(i+1)

            viz_line_data = np.copy(data[:,i])
            index_arr = np.where(viz_line_data!=0)
            # 忽略值为0的数据，不划线
            viz_line_data = viz_line_data[index_arr]
            x_range_in = x_range[index_arr]
            if i==0:
                opts={
                    'showlegend': True, 
                    'title': title,
                    'caption':desc,
                    'xlabel': "time", 
                    'ylabel': "values", 
                }           
                if ytickmin is not None:
                    opts["ytickmin"] = ytickmin
                    opts["ytickmax"] = ytickmax
                self.viz.line(
                    X=x_range_in,
                    Y=viz_line_data,
                    win=win,
                    name=line_name,
                    update=None,
                    opts=opts,
                )  
            else:
                self.viz.line(
                    X=x_range_in,
                    Y=viz_line_data,
                    win=win,
                    name=line_name,
                    update='append',
                )
                
    def viz_bar_compare(self,data,win="compare",rownames=None,title=None,legends=None):    
        """柱状图（多柱并列）"""

        opts=dict(
            stacked=False,
            legend=legends,
            rownames=rownames,
            title=title,
            ylabel='',
            xtickmin=0.4, 
            xtickstep=0.4
        )
        self.viz.bar(X=data,opts=opts,win=win)
     
    def viz_data_hist(self,data,numbins=10,win="histogram",title="histogram"):     
        """绘制柱状比较图"""
        
        length = data.shape[0]
        self.viz.histogram(X=data, win=win, opts=dict(numbins=numbins,title=title))  

    def viz_mpf_data(self,primary_data,pri_cols=None,sec_cols=None,target_title=None,file_path=None):
        """预测数据单条显示"""
        
        primary_data.index.name = "Time Index"
        primary_data.rename(columns={'OPEN':'Open','CLOSE':'Close','HIGH':'High','LOW':'Low','VOLUME_CLOSE':'Volume'},inplace=True)
        apds = []
        # 主要数据
        for index,col_name in enumerate(pri_cols):
            if index==0:
                color = "lime"
            if index==1:
                color = "c"                                                
            if index==2:
                color = "dimgray"                  
            mpf_data = mpf.make_addplot(primary_data[col_name],panel=0,color=color,secondary_y=False)
            apds.append(mpf_data)
        # 附加数据
        for index,col_name in enumerate(sec_cols):
            if index==0:
                color = "lime"
            if index==1:
                color = "c"   
            if index==2:
                color = "dimgray"                            
            mpf_data = mpf.make_addplot(primary_data[col_name],panel=1,color=color,secondary_y=False,y_on_right=True)
            apds.append(mpf_data)            
        mav = (5, 10, 20)
        mav = ()
        fig, axes = mpf.plot(primary_data, title=target_title,type='candle',addplot=apds, mav=mav,volume=False,returnfig=True,savefig=file_path,figsize=(12, 8))
        axes[0].legend(pri_cols,loc='upper left')        

if __name__ == "__main__":
    test_normal_vis()
    # reals_data_test()
    
       
    