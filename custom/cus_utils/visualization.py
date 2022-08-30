from cus_utils.tensor_viz import TensorViz
import torch
import torch.nn as nn
import numpy as np

VIZ_ITEM_NUMBER = 3

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
    
    