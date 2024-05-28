from cus_utils.tensor_viz import TensorViz
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

VIZ_ITEM_NUMBER = 3


def clu_coords_viz(coords,imp_index=None,name="viz_coords_result",labels=None,save_path=None,att_data=None):
    """Viz for coords points"""
    
    plt.figure(name,figsize=(12,9))
    if labels is None:
        plt.scatter(coords[:,0],coords[:,1],marker='o')
    else:
        p_index = np.where(labels==3)[0]
        p2_index = np.where(labels==2)[0]
        n_index = np.where(labels==0)[0]
        n2_index = np.where(labels==1)[0]
        p = plt.scatter(coords[p_index,0],coords[p_index,1], marker='o',color="r", s=50)
        p2 = plt.scatter(coords[p2_index,0],coords[p2_index,1], marker='o',color="b", s=50)
        n = plt.scatter(coords[n_index,0],coords[n_index,1], marker='x',color="k", s=50)
        n2 = plt.scatter(coords[n2_index,0],coords[n2_index,1], marker='x',color="y", s=50)
        # 显示关注点
        if imp_index is not None and imp_index.shape[0]>0:
            plt.scatter(coords[imp_index,0],coords[imp_index,1], marker='p',color="m", s=80)
            # 给簇关注点打标注
            for i in range(imp_index.shape[0]):
                x = coords[imp_index[i]][0]
                y = coords[imp_index[i]][1]
                if att_data is None:
                    imp_desc = "{}".format(i)
                else:
                    imp_desc = "{}/{}".format(i,att_data[i])
                plt.annotate(imp_desc, xy = (x, y), xytext = (x+0.1, y+0.1))
        # plt.legend((p, p2,n2, n), ('P','P2','N2','N'), loc=2)       
        if save_path is None:
            save_path = './custom/data/results'
    plt.savefig("{}/{}".format(save_path,name))  
    return coords 

def ShowClsResult(W, B, X, Y, xt=None, yt=None,save_file=None):
    fig = plt.figure(figsize=(6,6))
    
    DrawFourCategoryPoints(X[:,0], X[:,1], Y[:], xlabel="x1", ylabel="x2", show=False)

    b12 = (B[0,1] - B[0,0])/(W[1,0] - W[1,1])
    w12 = (W[0,1] - W[0,0])/(W[1,0] - W[1,1])
    
    b13 = (B[0,0] - B[0,2])/(W[1,2] - W[1,0])
    w13 = (W[0,0] - W[0,2])/(W[1,2] - W[1,0])

    b14 = (B[0,0] - B[0,3])/(W[1,3] - W[1,0])
    w14 = (W[0,0] - W[0,3])/(W[1,3] - W[1,0])
    
    b23 = (B[0,2] - B[0,1])/(W[1,1] - W[1,2])
    w23 = (W[0,2] - W[0,1])/(W[1,1] - W[1,2])

    b24 = (B[0,3] - B[0,1])/(W[1,1] - W[1,3])
    w24 = (W[0,3] - W[0,1])/(W[1,1] - W[1,3])

    b34 = (B[0,3] - B[0,2])/(W[1,2] - W[1,3])
    w34 = (W[0,3] - W[0,2])/(W[1,2] - W[1,3])
    
    def create_xdata():
        return np.array([X[:,0].min(),X[:,0].max()])
    
    x = create_xdata()
    y = w13 * x + b13
    # p13, = plt.plot(x,y,c='r')

    x = create_xdata()
    y = w23 * x + b23
    p23, = plt.plot(x,y,c='b')

    x = create_xdata()
    y = w12 * x + b12
    p12, = plt.plot(x,y,c='black')

    x = create_xdata()
    y = w14 * x + b14
    # p14, = plt.plot(x,y,c='yellow')

    x = create_xdata()
    y = w24 * x + b24
    # p24, = plt.plot(x,y,c='black')

    x = create_xdata()
    y = w34 * x + b34
    p34, = plt.plot(x,y,c='r')
            
    # plt.legend([p12,p13,p14,p23,p24,p34], ["12","13","14","23","24","34"])
    plt.legend([p12,p23,p34], ["12","23","34"])
    plt.axis([X[:,0].min(),X[:,0].max(),X[:,1].min(),X[:,1].max()])
    
    # DrawFourCategoryPoints(xt[:,0], xt[:,1], yt[:], xlabel="x1", ylabel="x2", show=True, isPredicate=True)
    plt.savefig(save_file)

def DrawFourCategoryPoints(X1, X2, Y_label, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['black', 'y', 'b','r']
    shapes = ['x', 'x', 'o','o']
    assert(X1.shape[0] == X2.shape[0] == Y_label.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = Y_label[i]
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    #end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show() 

class VisUtil:
    
    def __init__(self):
        self.viz_target_pred = TensorViz(env="dataview_pred_target")
        self.viz_valid = TensorViz(env="dataview_validation")
        self.viz_input = TensorViz(env="dataview_input")          
    
    def viz_loss(self,out,x,y,index=0,loss=None,loss_value=None,step="training"):
        if index % 10 != 0:
            return
        print("do viz_training loss:",index)
        # 取得某个股票的预测值,并转换为预测类别数值
        pred_fw = loss.to_prediction(out.prediction)
        pred = pred_fw[VIZ_ITEM_NUMBER].unsqueeze(0)
        label = y[0][VIZ_ITEM_NUMBER].unsqueeze(0)
        target_pred = torch.cat((pred,label),0).cpu().numpy().transpose(1,0)
        names=["pred","target"]
        if loss_value is None:
            loss_value = 0
        else:
            loss_value = round(loss_value.cpu().item(), 2)
        win_target = "target_{}".format(str(index))
        title = "target_{} loss:{}".format(str(index),loss_value)
        desc = "target_{} loss:{}".format(str(index),loss_value)
        if step=="training":
            self.viz_target_pred.viz_matrix_var(target_pred,win=win_target,title=title,desc=desc,names=names)
        else:
            self.viz_valid.viz_matrix_var(target_pred,win=win_target,title=title,desc=desc,names=names)
        if index%100==0:
            print("iii")
        print("step viz")      
    
    def viz_loss_bar(self,out,x,y,epoch=0,index=0,loss=None,loss_value=None,step="training"):
        """图形显示预测与标签进行比较结果"""
        
        if step=="training" and index not in [3,6,9]:
            return
        if step=="validation" and index not in [0]:
            return        
        # 取得某个股票的预测值,并转换为预测类别数值
        pred_fw = loss.to_prediction(out.prediction)
        pred = pred_fw[VIZ_ITEM_NUMBER].unsqueeze(0)
        label = y[0][VIZ_ITEM_NUMBER].unsqueeze(0)
        target_pred = torch.cat((pred,label),0).cpu().numpy().transpose(1,0)
        names=["pred","target"]
        if loss_value is None:
            loss_value = 0
        else:
            loss_value = round(loss_value.cpu().item(), 2)
        win_target = "target_{}_{}".format(epoch,str(index))
        title = "target_{}_{} loss:{}".format(epoch,str(index),loss_value)
        desc = "target_{} loss:{}".format(str(index),loss_value)
        if step=="training":
            self.viz_target_pred.viz_data_bar(target_pred,win=win_target,title=title,desc=desc,names=names)
        else:
            self.viz_valid.viz_data_bar(target_pred,win=win_target,title=title,desc=desc,names=names)
        print("step viz")  
  
    def viz_target_data(self,data,epoch=0,index=0,loss=None,loss_value=None,type="training"):
        """图形显示结果数据"""
        
        if type=="training" and index not in [3,6,9]:
            return
        if type=="validation" and index not in [0]:
            return        
        
        pred = []
        results = data["results"]
        # 摘取出序列里的max值编号，进行后续比较
        for result in results:
            result = result[VIZ_ITEM_NUMBER,:]
            p, yi = result.topk(1)
            pred.append(yi)
        pred = torch.tensor([pred])
        label = data["target"][VIZ_ITEM_NUMBER].unsqueeze(0).cpu()
        target_pred = torch.cat((pred,label),0).numpy().transpose(1,0)
        names=["pred","target"]
        if loss_value is None:
            loss_value = 0
        else:
            loss_value = round(loss_value.cpu().item(), 2)
        win_target = "target_{}_{}".format(epoch,str(index))
        title = "target_{}_{} loss:{}".format(epoch,str(index),loss_value)
        desc = "target_{} loss:{}".format(str(index),loss_value)
        if type=="training":
            self.viz_target_pred.viz_data_bar(target_pred,win=win_target,title=title,desc=desc,names=names)
        else:
            self.viz_valid.viz_data_bar(target_pred,win=win_target,title=title,desc=desc,names=names)
        # print("step viz")  

    def viz_input_data(self,data,epoch=0,index=0,loss=None,loss_value=None,type="training"):
        """图形显示输入数据"""
        
        if type=="training" and index not in [3,6,9]:
            return
        if type=="validation" and index not in [0]:
            return        
        
        input_xc_data = data['inputs_xc'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_xw_data = data['inputs_xw'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_data = np.concatenate((np.expand_dims(input_xw_data,axis=1),input_xc_data), axis=1)
        names = ["xw"]
        names = names + ["xc_{}".format(i) for i in range(input_xw_data.shape[0])]
        win_input = "input_{}_{}".format(epoch,str(index))
        title = win_input
        
        if type=="training":    
            self.viz_input.viz_matrix_var(input_data,win=win_input,title=title,names=names)
        else:
            self.viz_valid.viz_matrix_var(input_data,win=win_input,title=title,names=names)         
             
    def viz_training(self,out,x,y,index=0):
        if index % 10 != 0:
            return
        print("do viz_training:",index)
        # 分位数图形
        pred_fw = out.prediction[VIZ_ITEM_NUMBER]
        names=["pred_{}".format(i) for i in range(7)]
        win_target = "target_" + str(index)
        self.viz_target_pred.viz_matrix_var(pred_fw,win=win_target,names=names)
        # 预测值与实际值图形      
        label = y[0][VIZ_ITEM_NUMBER].unsqueeze(-1)
        prediction = self.to_prediction(out, {})
        prediction = prediction[VIZ_ITEM_NUMBER].unsqueeze(-1)
        target_pred = torch.cat((prediction,label),1)
        names=["pred","label"]
        win_target = "pred_label_" + str(index)
        self.viz_target_pred.viz_matrix_var(target_pred,win=win_target,names=names)        
        # 输入值图形
        input_cats = x['encoder_cat'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_reals = x['encoder_cont'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_data = np.concatenate((input_reals,input_cats), axis=1)
        win_input = "input_" + str(index)
        inputs_names = self.reals + self.categoricals
        self.viz_target_pred.viz_matrix_var(input_data,win=win_input,names=inputs_names)
        if index%100==0:
            print("iii")
        print("step viz")  

    def viz_validation(self,out,x,y,index=0):
        if index % 10 != 0:
            return
        print("do viz_training:",index)
        # 分位数图形
        pred_fw = out.prediction[VIZ_ITEM_NUMBER].detach().cpu().numpy()
        names=["pred_{}".format(i) for i in range(7)]
        win_target = "target_" + str(index)
        self.viz_valid.viz_matrix_var(pred_fw,win=win_target,names=names)
        # 预测值与实际值图形      
        label = y[0][VIZ_ITEM_NUMBER].unsqueeze(-1)
        prediction = self.to_prediction(out, {})
        prediction = prediction[VIZ_ITEM_NUMBER].unsqueeze(-1)
        target_pred = torch.cat((prediction,label),1)
        names=["pred","label"]
        win_target = "pred_label_" + str(index)
        self.viz_valid.viz_matrix_var(target_pred,win=win_target,names=names)        
        # 输入值图形
        input_cats = x['encoder_cat'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_reals = x['encoder_cont'][VIZ_ITEM_NUMBER].detach().cpu().numpy()
        input_data = np.concatenate((input_reals,input_cats), axis=1)
        win_input = "input_" + str(index)
        inputs_names = self.reals + self.categoricals
        self.viz_valid.viz_matrix_var(input_data,win=win_input,names=inputs_names)
        if index%100==0:
            print("iii")
        print("step viz")  
               
    def viz_output(self,output,index=0,is_training=True):
        if index % 10 != 0:
            return     
        print("do viz_output:",index)   
        output = output[VIZ_ITEM_NUMBER]
        output = output.detach().cpu().numpy()
        names=["output_{}".format(i) for i in range(7)]
        win_output= "output_" + str(index)
        title = win_output
        if is_training:
            self.viz_target_pred.viz_matrix_var(output,win=win_output,title=title,names=names)
        else:
            self.viz_valid.viz_matrix_var(output,win=win_output,title=title,names=names)
    
       
        
        
    