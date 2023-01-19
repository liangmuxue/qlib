from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics

from torch import Tensor
from darts.utils import likelihood_models

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
    """不确定损失"""

    def __init__(self, mse_reduction="mean",device=None,loss_sigma=None):
        super(UncertaintyLoss, self).__init__()
        
        v_num = 3
        self.mse_reduction = mse_reduction
        self.sigma = loss_sigma
        self.v_num = v_num
        self.device = device
        self.epoch = 0

    def forward(self, input_ori: Tensor, target: Tensor,outer_loss=None,epoch=0):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        # 如果是似然估计下的数据，需要取中间值
        if len(input_ori.shape)==4:
            input = torch.mean(input_ori[:,:,0,:],dim=-1)
        else:
            input = input_ori
        # 相关系数损失
        corr_loss = self.corr_loss_comp(input, target)   
        # return corr_loss  
        # MSE损失
        if outer_loss is not None:
            compute_loss = outer_loss
        else:
            compute_loss = F.mse_loss(input, target, reduction=self.mse_reduction)     
        
        # 前后数值差的距离衡量
        # input_value_diff = input[:,-1] - input[:,0]
        # target_value_diff = target[:,-1] - target[:,0]
        # value_diff_loss = F.mse_loss(input[:,[0,-1]], target[:,[0,-1],0], reduction=self.mse_reduction)    
        loss_sum = 0
        # 使用不确定性损失模式进行累加
        loss_sum += 0.5 / (self.sigma[0] ** 2) * corr_loss + torch.log(1 + self.sigma[0] ** 2)
        loss_sum += 0.5 / (self.sigma[1] ** 2) * compute_loss + torch.log(1 + self.sigma[1] ** 2)
        # loss_sum += 0.5 / (self.sigma[2] ** 2) * value_diff_loss + torch.log(1 + self.sigma[2] ** 2)
        # loss_sum = 5 * corr_loss + mse_loss
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
