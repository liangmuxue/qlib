from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
import numpy as np

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
        sig_params = torch.ones(4, requires_grad=True).to(device)
        self.sigma = torch.nn.Parameter(sig_params)
        
        self.device = device
        self.epoch = 0
        
        # 涨跌幅分类损失中，需要设置不同分类权重
        vr_loss_weight = get_simple_class_weight()
        vr_loss_weight = torch.from_numpy(np.array(vr_loss_weight)).to(device)
        self.mse_weight = mse_weight.to(device)
        self.last_classify_loss = ScopeLoss(reduction=mse_reduction,device=device)
        # self.vr_loss = nn.CrossEntropyLoss(weight=vr_loss_weight)
        self.vr_loss = nn.CrossEntropyLoss()
        self.mse_loss = MseLoss(reduction=mse_reduction,device=device)
        self.scope_loss = nn.CrossEntropyLoss()

    def forward(self, input_ori: Tensor, target_ori: Tensor,outer_loss=None,epoch=0):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        (input,slope_out,second_class,third_class) = input_ori
        (target,future_target,target_class,slope_target) = target_ori
        # slope_target = (target[:,-1] - target[:,0])/target[:,0]
        vr_target= target_class[:,0]
        last_vr_target = target_class[:,1]
        first_input = input[:,:,0]
        first_label = target[:,:,0]
        second_input = input[:,:,1]
        second_label = target[:,:,1]     
        third_input = input[:,:,2]
        third_label = target[:,:,2]          
        # 如果是似然估计下的数据，需要取中间值
        if len(input.shape)==4:
            input = torch.mean(input[:,:,0,:],dim=-1)
        else:
            input = torch.squeeze(input,-1)
        # 相关系数损失
        corr_loss = self.ccc_loss_comp(first_input, first_label)   
        # 第二指标分类
        ce_loss = 0.0 # self.vr_loss(second_class, vr_target)
        # 第三指标计算
        mse_loss = self.ccc_loss_comp(second_input, second_label)     
        # 第三指标分类 
        value_diff_loss = self.ccc_loss_comp(third_input,third_label)  
        # value_diff_loss = 0.0
        mean_threhold = 0.0 
        # if slope_out.max()>1:
        #     max_item = slope_out.max(dim=1)[0]
        #     print("slope_out weight >1 cnt:{}".format(torch.sum(max_item>1)))
        # 使用不确定性损失模式进行累加
        # loss_sum += 1/3 / (self.sigma[0] ** 2) * ce_loss + torch.log(1 + self.sigma[0] ** 2)
        # loss_sum += 1/2 / (self.sigma[1] ** 2) * value_range_loss + torch.log(1 + self.sigma[1] ** 2)
        # loss_sum += 1/2 / (self.sigma[2] ** 2) * corr_loss + torch.log(1 + self.sigma[2] ** 2)
        # loss_sum += 1/2 / (self.sigma[3] ** 2) * mse_loss + torch.log(1 + self.sigma[3] ** 2)
        loss_sum = value_diff_loss
        # loss_sum = ce_loss + value_diff_loss
        
        return loss_sum,(mse_loss,value_diff_loss,corr_loss,ce_loss,mean_threhold)
    
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
