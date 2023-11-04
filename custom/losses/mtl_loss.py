from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
import numpy as np
from tslearn.metrics import SoftDTWLossPyTorch

from cus_utils.common_compute import normalization
from cus_utils.encoder_cus import transform_slope_value
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES,get_simple_class_weight

mse_weight = torch.tensor(np.array([1,2,3,4,5]))  
mse_scope_weight = torch.tensor(np.array([1,2,3,4]))  

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

class PartLoss(_Loss):
    """自定义损失，只衡量部分数据"""
    
    __constants__ = ['reduction']

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_item_range = (input[:,-1] - input[:,-2])/input[:,-2]
        target_item_range = (target[:,-1] - target[:,-2])/target[:,-2]
        loss = torch.abs(input_item_range - target_item_range)
        return torch.mean(loss)
    
class LastClassifyLoss(nn.BCEWithLogitsLoss):
    """自定义二分类损失，计算最后一段预测上升还是下降的准确性"""
    
    __constants__ = ['reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(LastClassifyLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.device = device
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        last_sec_slop = (input[:,-1] - input[:,-2])/input[:,-2]
        last_sec_out = normalization(last_sec_slop,mode="torch",avoid_zero=False)
        last_sec_tar_bool = ((target[:,-1] - target[:,-2])>0).type(torch.float64)
        loss = super().forward(last_sec_out,last_sec_tar_bool)
        return loss  

class ScopeLoss(_Loss):
    """涨跌幅度的度量"""
    
    __constants__ = ['reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(ScopeLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.device = device
        if weight is None:
            weight = mse_scope_weight
        self.weight = weight.to(self.device)

    def forward(self, slope_input: Tensor, slope_target: Tensor) -> Tensor:
        # slope_target = transform_slope_value(target)
        slope_arr = torch.abs(slope_input - slope_target)
        if self.reduction=="mean":
            loss = torch.mean(slope_arr)
        else:
            loss = torch.sum(slope_arr)   
        mean_threhold = torch.mean(torch.stack((slope_input,slope_target)),dim=0).mean()
        return loss,mean_threhold 
               
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
        self.last_classify_loss = ScopeLoss(reduction=mse_reduction,device=device)
        # self.vr_loss = nn.CrossEntropyLoss(weight=vr_loss_weight)
        self.part_loss = PartLoss()
        self.vr_loss = nn.MSELoss()
        self.mse_loss = nn.MSELoss() # MseLoss(reduction=mse_reduction,device=device)
        self.tar_loss = nn.MSELoss()
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1,normalize=True)
        # 设置损失函数的组合模式
        self.loss_mode = 0

    def forward(self, input_ori: Tensor, target_ori: Tensor,optimizers_idx=0,epoch=0):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        (input,vr_class,tar_class) = input_ori
        (target,target_class,target_info,y_transform) = target_ori
        # slope_target = (target[:,-1] - target[:,0])/target[:,0]
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        # 指标分类
        ce_loss = torch.tensor(0).to(self.device) 
        value_diff_loss = torch.tensor(0).to(self.device)         
        # 相关系数损失,多个目标
        for i in range(len(input)):
            if optimizers_idx==i or optimizers_idx==-1:
                input_item = input[i][:,:,0]
                label_item = target[:,:,i]            
                if i==1:
                    corr_loss_combine[i] = self.ccc_loss_comp(input_item, label_item)   
                elif i==2:
                    corr_loss_combine[i] = self.ccc_loss_comp(input_item, label_item)   
                else: 
                    corr_loss_combine[i] = self.ccc_loss_comp(input_item, label_item)
                loss_sum = corr_loss_combine[i]
        # 二次目标损失部分
        # if optimizers_idx==len(input) or optimizers_idx==-1:
        #     # value_diff_loss = self.compute_dtw_loss(third_input,third_label) 
        #     ce_loss = self.vr_loss(vr_class, y_transform)
        #     loss_sum = ce_loss
        # if optimizers_idx==len(input)+1 or optimizers_idx==-1:
        #     # value_diff_loss = self.compute_dtw_loss(third_input,third_label) 
        #     value_diff_loss = self.tar_loss(tar_class, raise_range)
        #     loss_sum = value_diff_loss
        # 验证阶段，全部累加
        if optimizers_idx==-1:
            loss_sum = torch.sum(corr_loss_combine) + ce_loss + value_diff_loss
        
        return loss_sum,[corr_loss_combine,ce_loss,value_diff_loss]
    
    def corr_loss_comp(self, input: Tensor, target: Tensor):
        
        num_outputs = input.shape[0]
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        input_corr = torch.squeeze(input).transpose(1,0)
        target_corr = torch.squeeze(target).transpose(1,0)
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
          
# if __name__ == '__main__':
#     weighted_loss_func = UncertaintyLoss(2)
#     weighted_loss_func.to(device)
#     optimizer = torch.optim.Adam(
#         filter(lambda x: x.requires_grad, list(model.parameters()) + list(weighted_loss_func.parameters())),
#         betas=(0.9, 0.98), eps=1e-09)
#
#     if epoch < 10:
#         loss = loss1
#     else:
#         loss = weighted_loss_func(loss1, loss2)
