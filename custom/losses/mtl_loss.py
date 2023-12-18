from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
from torch.nn import TripletMarginWithDistanceLoss
import numpy as np
from tslearn.metrics import SoftDTWLossPyTorch

from cus_utils.common_compute import normalization
from cus_utils.encoder_cus import transform_slope_value
from tft.class_define import CLASS_SIMPLE_VALUE_SEC,CLASS_SIMPLE_VALUE_MAX,get_simple_class_weight

mse_weight = torch.tensor(np.array([1,2,3,4,5]))  
mse_scope_weight = torch.tensor(np.array([1,2,3,4]))  

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

distance = distances.LpDistance()
reducer = reducers.ThresholdReducer(low=0)


class MseLoss(_Loss):
    """自定义mse损失，用于设置类别权重"""
    
    __constants__ = ['reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(MseLoss, self).__init__(size_average, reduce, reduction)
        self.device = device
        if weight is None:
            weight = mse_weight      
        self.weight = weight.to(self.device)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss_arr_ori = (input - target) ** 2
        nag_jud_weight = (input>target).int()
        # 倾向于预测值小于实际值，在此进行加权
        loss_arr_pun = loss_arr_ori + loss_arr_ori * nag_jud_weight
        if self.reduction=="mean":
            mse_loss = torch.mean(loss_arr_pun)
        else:
            mse_loss = torch.sum(loss_arr_pun)
        return mse_loss  

class BatchScopeLoss(_Loss):
    """批量数据的涨跌幅度衡量"""
    
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(BatchScopeLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.device = device

    def forward(self, slope_input: Tensor, slope_target: Tensor) -> Tensor:
        # slope_target = transform_slope_value(target)
        loss = self.corr_loss_comp(slope_input,slope_target)
        return loss 

class TripletLoss(_Loss):
    """自定义三元组损失函数"""
    
    __constants__ = ['reduction']

    def __init__(self, hard_margin=1.5,semi_margin=1.2,dis_func=None,reduction: str = 'mean',device=None) -> None:
        super(TripletLoss, self).__init__(reduction=reduction)
        
        self.device = device
        self.reduction = reduction
        self.hard_triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=dis_func,margin=hard_margin)      
        self.semi_triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=dis_func,margin=semi_margin)  
                      
    def forward(self, output: Tensor, target: Tensor,labels=None,labels_value=None) -> Tensor:
        """使用度量学习，用Triplet Loss比较排序损失,使用实际的目标值作为锚点(Anchor)"""
        
        import_index = torch.where(labels==CLASS_SIMPLE_VALUE_MAX)[0]
        imp_sec_index = torch.where(labels==CLASS_SIMPLE_VALUE_SEC)[0]
        fail_index = torch.where(labels==0)[0]
        fail_sec_index = torch.where(labels==1)[0]
        # 把三元组数据分别遍为3组，(3,0) (3,1) (2,0),并分别计算三元组损失
        loss_total = 0.0
        for i in range(3):
            if i==0:
                pos_index,neg_index = self.combine_index(import_index, fail_index)
            if i==1:
                pos_index,neg_index = self.combine_index(import_index, fail_sec_index)       
            if i==2:
                pos_index,neg_index = self.combine_index(imp_sec_index, fail_index)        
            pos_data = torch.index_select(output, 0, pos_index)       
            neg_data = torch.index_select(output, 0, neg_index)      
            anchor_data = torch.index_select(labels_value, 0, pos_index)  
            # 不同等级之间的距离使用不同的margin   
            if i==0:     
                loss_item = self.hard_triplet_loss(anchor_data,pos_data,neg_data)
            else:
                loss_item = self.semi_triplet_loss(anchor_data,pos_data,neg_data)
            loss_total += loss_item
        return loss_total 
    
    def combine_index(self,pos_index,nag_index):
        size = pos_index.shape[0] if pos_index.shape[0]<nag_index.shape[0] else nag_index.shape[0]
        return pos_index[:size],nag_index[:size]
                  
class UncertaintyLoss(nn.Module):
    """不确定损失,包括mse，corr以及分类交叉熵损失等"""

    def __init__(self, mse_reduction="mean",device=None,loss_sigma=None):
        super(UncertaintyLoss, self).__init__()
        
        self.mse_reduction = mse_reduction
        # self.sigma = loss_sigma
        # sig_params = torch.ones(4, requires_grad=True).to(device)
        # self.sigma = torch.nn.Parameter(sig_params)
        
        self.device = device
        self.epoch = 0
        
        # 涨跌幅分类损失中，需要设置不同分类权重
        vr_loss_weight = get_simple_class_weight()
        vr_loss_weight = torch.from_numpy(np.array(vr_loss_weight)).to(device)
        self.mse_weight = mse_weight.to(device)
        self.batch_scope_loss = BatchScopeLoss(reduction=mse_reduction,device=device)
        self.vr_loss = nn.MSELoss()
        self.mse_loss = nn.MSELoss() # MseLoss(reduction=mse_reduction,device=device)
        
        self.triplet_loss = TripletLoss(dis_func=self.abs_dis,device=device)
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1,normalize=True)
        self.rankloss = TripletLoss(reduction='mean')
        # 设置损失函数的组合模式
        self.loss_mode = 0

    def forward(self, input_ori: Tensor, target_ori: Tensor,optimizers_idx=0,epoch=0):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        (input,vr_combine_class,vr_classes) = input_ori
        (target,target_class,target_info,y_transform) = target_ori
        label_class = target_class[:,0]
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        # 指标分类
        ce_loss = torch.tensor(0.0).to(self.device) 
        value_diff_loss = torch.tensor(0).to(self.device)         
        # 相关系数损失,多个目标
        for i in range(len(input)):
            if optimizers_idx==i or optimizers_idx==-1:
                input_item = input[i][:,:,0]
                target_item = target[:,:,i]    
                label_item = torch.Tensor(np.array([target_info[j]["raise_range"] for j in range(len(target_info))])).to(self.device)
                label_item = normalization(label_item,mode="torch")
                vr_item_class = vr_classes[i][:,0] 
                if i==1:
                    corr_loss_combine[i] = self.triplet_loss(vr_item_class, target_item,labels=label_class,labels_value=label_item)   
                    # corr_loss_combine[i] += self.ccc_loss_comp(input_item,target_item)
                elif i==2:
                    corr_loss_combine[i] = self.triplet_loss(vr_item_class, target_item,labels=label_class,labels_value=label_item)   
                    # corr_loss_combine[i] += self.ccc_loss_comp(input_item,target_item)
                else: 
                    corr_loss_combine[i] = self.triplet_loss(vr_item_class, target_item,labels=label_class,labels_value=label_item)   
                    # corr_loss_combine[i] += self.ccc_loss_comp(input_item,target_item)
                loss_sum = corr_loss_combine[i]
        # 二次目标损失部分
        # if optimizers_idx==len(input):
        #     # value_diff_loss = self.compute_dtw_loss(third_input,third_label) 
        #     ce_loss = self.rankloss(vr_class, target_class[:,0])
        #     loss_sum = ce_loss
        # 验证阶段，全部累加
        if optimizers_idx==-1:
            # ce_loss = nn.CrossEntropyLoss()(vr_class, target_class[:,0])
            loss_sum = torch.sum(corr_loss_combine) + ce_loss + value_diff_loss
        
        return loss_sum,[corr_loss_combine,ce_loss,value_diff_loss]

    def abs_dis(self,x1, x2):
        return torch.abs(x1 - x2)      
    
    def corr_loss_comp(self, input: Tensor, target: Tensor):
        
        num_outputs = input.shape[0]
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        input_corr = torch.squeeze(input).transpose(1,0)
        target_corr = torch.squeeze(target).transpose(1,0)
        corr_loss = pearson(input_corr, target_corr)   
        corr_loss = 1 - corr_loss
        return torch.mean(corr_loss)

    def batch_corr_loss_comp(self, input: Tensor, target: Tensor):
        
        pearson = torchmetrics.PearsonCorrCoef().to(self.device)
        input_corr = torch.squeeze(input)
        target_corr = torch.squeeze(target)
        corr_loss = pearson(input_corr, target_corr)   
        corr_loss = 1 - corr_loss
        return torch.mean(corr_loss)
     
    def ccc_loss_comp(self, input_ori: Tensor, target_ori: Tensor):
        """一致性相关系数计算"""
        
        input = input_ori.flatten()
        target = target_ori.flatten()
        corr_tensor = torch.stack([input,target],dim=0)
        cor = torch.corrcoef(corr_tensor)[0][1]
        var_true = torch.var(target)
        var_pred = torch.var(input)
        sd_true = torch.std(target)
        sd_pred = torch.std(input)
        numerator = 2*cor*sd_true*sd_pred
        mse_part = self.mse_loss(input_ori,target_ori)
        denominator = var_true + var_pred + mse_part
        ccc = numerator/denominator
        ccc_loss = 1 - ccc
        return ccc_loss

    def ccc(self,x,y):
        sxy = torch.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
        rhoc = 2*sxy / (torch.var(x) + torch.var(y) + (x.mean() - y.mean())**2)
        return rhoc
    
    def compute_dtw_loss(self,input,target):
        input_real = torch.unsqueeze(input,-1)
        target_real = torch.unsqueeze(target,-1)
        loss = self.dtw_loss(input_real, target_real).mean()       
        return loss
  
