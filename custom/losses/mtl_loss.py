from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
import numpy as np

from darts.utils import likelihood_models
from tft.class_define import CLASS_VALUES,CLASS_SIMPLE_VALUES,get_simple_class_weight

class CorrLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, num_outputs=5, size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(CorrLoss, self).__init__(size_average, reduce, reduction)
        self.device = device
        # self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse_loss = F.mse_loss(input, target, reduction=self.reduction)
        num_outputs = input.shape[0]
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        input_corr = torch.squeeze(input).transpose(1,0)
        target_corr = torch.squeeze(target).transpose(1,0)
        corr_loss = pearson(input_corr, target_corr)
        return corr_loss  
    
class UncertaintyLoss(nn.Module):
    """不确定损失,包括mse，corr以及分类交叉熵损失等"""

    def __init__(self, mse_reduction="mean",device=None,loss_sigma=None):
        super(UncertaintyLoss, self).__init__()
        
        self.mse_reduction = mse_reduction
        self.sigma = loss_sigma
        self.device = device
        self.epoch = 0
        
        # 涨跌幅分类损失中，需要设置不同分类权重
        vr_loss_weight = get_simple_class_weight()
        vr_loss_weight = torch.from_numpy(np.array(vr_loss_weight)).to(device)
        self.classify_loss = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(weight=vr_loss_weight)]

    def forward(self, input_ori: Tensor, target_ori: Tensor,outer_loss=None,epoch=0):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        input = input_ori[0]
        input_classify = input_ori[1]
        vr_class = input_ori[2]
        target = target_ori[0]
        target_classify = target_ori[1]
        # 均线形态分为2个部分
        input_classify_arr = [input_classify[:,0,:],input_classify[:,1,:]]
        # 如果是似然估计下的数据，需要取中间值
        if len(input.shape)==4:
            input = torch.mean(input[:,:,0,:],dim=-1)
        else:
            input = torch.squeeze(input,-1)
        # 相关系数损失
        corr_loss = self.corr_loss_comp(input, target)   
        # 针对均线2个部分，分别计算交叉熵损失并相加 
        ce_loss = 0
        for i in range(2):
            ce_loss += self.classify_loss[i](input_classify_arr[i],target_classify[:,i])
        # 第3个部分为幅度范围分类，计算交叉熵损失 
        value_range_loss = self.classify_loss[2](vr_class[:,0,:],target_classify[:,2])              
        # 整体MSE损失
        # mse_loss = F.mse_loss(input, target[:,:,0], reduction=self.mse_reduction)     
        # 只衡量第一个和最后一个数值
        value_diff_loss = F.mse_loss(input[:,[0,-1]], target[:,[0,-1],0], reduction=self.mse_reduction)    
        
        loss_sum = 0
        # 使用不确定性损失模式进行累加
        loss_sum += 1/4 / (self.sigma[0] ** 2) * value_range_loss + torch.log(1 + self.sigma[0] ** 2)
        loss_sum += 1/4 / (self.sigma[1] ** 2) * ce_loss + torch.log(1 + self.sigma[1] ** 2)
        loss_sum += 1/4 / (self.sigma[2] ** 2) * corr_loss + torch.log(1 + self.sigma[2] ** 2)
        loss_sum += 1/4 / (self.sigma[2] ** 2) * value_diff_loss + torch.log(1 + self.sigma[2] ** 2)
        return loss_sum
    
    def corr_loss_comp(self, input: Tensor, target: Tensor):
        
        num_outputs = input.shape[0]
        pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        input_corr = torch.squeeze(input).transpose(1,0)
        target_corr = torch.squeeze(target).transpose(1,0)
        corr_loss = pearson(input_corr, target_corr)   
        corr_loss = 1 - corr_loss
        return torch.mean(corr_loss)
        
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
