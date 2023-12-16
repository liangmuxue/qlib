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

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

distance = distances.LpDistance()
reducer = reducers.ThresholdReducer(low=0)


def listMLE(y_pred, y_true, eps=1e-9, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

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

class RankLoss(_Loss):
    """自定义排序损失函数"""
    
    __constants__ = ['reduction']

    def __init__(self, margin=1,size_average=None, reduce=None, reduction: str = 'mean',device=None) -> None:
        super(RankLoss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.device = device
        self.reduction = reduction
        
        self.loss_func = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=margin, distance=distance, type_of_triplets="semihard"
        )
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """使用度量学习，用Triplet Loss比较排序损失"""
        
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss_func(embeddings, labels, indices_tuple)        
        return loss 
                   
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
        self.part_loss = PartLoss()
        self.vr_loss = nn.MSELoss()
        self.mse_loss = nn.MSELoss() # MseLoss(reduction=mse_reduction,device=device)
        self.tar_loss = nn.MSELoss()
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1,normalize=True)
        self.rankloss = RankLoss(reduction='mean',margin=1)
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
                target_item_mean = torch.mean(target_item,-1)
                label_item = torch.Tensor(np.array([target_info[j]["raise_range"] for j in range(len(target_info))])).to(self.device)
                vr_item_class = vr_classes[i]     
                if i==1:
                    corr_loss_combine[i] = self.rankloss(vr_item_class, label_item)   
                elif i==2:
                    corr_loss_combine[i] = self.rankloss(vr_item_class, target_item_mean)   
                else: 
                    corr_loss_combine[i] = self.rankloss(vr_item_class, label_item)
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
  
