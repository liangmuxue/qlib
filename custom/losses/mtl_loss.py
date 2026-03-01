from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
from torch.nn import TripletMarginWithDistanceLoss
import numpy as np
import random
import math
from functools import reduce
from tslearn.metrics import SoftDTWLossPyTorch
import itertools
from cus_utils.common_compute import normalization,batch_normalization,adjude_seq_eps,compute_price_range,intersect1d
from cus_utils.encoder_cus import transform_slope_value
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_SEC,CLASS_SIMPLE_VALUE_MAX,get_simple_class_weight
from numba.cuda.cudadrv import ndarray
from .quanlity_loss import QuanlityLoss

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchmetrics.regression import ConcordanceCorrCoef
from networkx.algorithms import similarity

def similarity_consistency_loss(pred, target):
    """
    多通道相似度一致性损失：约束预测通道间的余弦相似度与真实一致
    Args:
        pred: 预测值，shape=[batch_size, seq_len, n_channels]
        target: 真实值，shape=[batch_size, seq_len, n_channels]
    Returns:
        sim_loss: 相似度一致性损失值
    """
    # 维度调整：[batch*seq_len, n_channels]，方便计算通道间相似度
    batch_seq = pred.shape[0] * pred.shape[1]
    pred_flat = pred.reshape(batch_seq, -1)  # [batch*seq_len, n_channels]
    target_flat = target.reshape(batch_seq, -1)
    
    # 计算真实通道间的余弦相似度矩阵
    target_sim = F.cosine_similarity(
        target_flat.unsqueeze(1), target_flat.unsqueeze(2), dim=-1
    )  # [batch*seq_len, n_channels, n_channels]
    
    # 计算预测通道间的余弦相似度矩阵
    pred_sim = F.cosine_similarity(
        pred_flat.unsqueeze(1), pred_flat.unsqueeze(2), dim=-1
    )
    
    # 约束相似度矩阵一致
    sim_loss = F.mse_loss(pred_sim, target_sim, reduction='mean')
    return sim_loss

class MinerLoss(_Loss):
    """具备挖掘功能的损失函数"""

    def __init__(self,pos_margin=0.1,neg_margin=0.3,dis_func=None):
        super(MinerLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.dis_func = dis_func

    def forward(self, output, target,labels):
        dist_data = self.dis_func(output, target)
        target_index = self.hard_sample_minering(dist_data, labels)
        return torch.mean(dist_data[target_index])

    def hard_sample_minering(self,dist_data,labels):
        """挖掘困难样本"""
        
        import_index = torch.where(labels==CLASS_SIMPLE_VALUE_MAX)[0]
        fail_index = torch.where(labels==0)[0]
        fail_sec_index = torch.where(labels==1)[0]
        
        def miner_pair(index_tar,type=1):
            dist_tar = dist_data[index_tar]
            if type==1:
                # 对于涨幅达标样本，要求输出尽量靠近
                miner_data_idx = torch.where(dist_tar>self.pos_margin)[0]
            else:
                # 对于负面样本，要求输出尽量不在正样本中
                miner_data_idx = torch.where(dist_tar>self.neg_margin)[0]
            dist_index = index_tar[miner_data_idx]
            return dist_index
        
        # 针对不同类别的数据，分别挖掘出困难样本
        import_data_index = miner_pair(import_index)  
        neg_data_index = miner_pair(fail_index)
        sec_neg_data_index = miner_pair(fail_sec_index) 
        # 合并
        target_index = intersect1d(import_data_index,neg_data_index)
        target_index = intersect1d(target_index,sec_neg_data_index)
    
class TripletLoss(_Loss):
    """自定义三元组损失函数"""
    
    __constants__ = ['reduction']

    def __init__(self, hard_margin=0.6,semi_margin=0.4,dis_func=None,anchor_target=False,mode=1,
                 reduction: str = 'mean',device=None,type_of_triplets="semihard") -> None:
        super(TripletLoss, self).__init__(reduction=reduction)
        
        self.device = device if device is not None else "cpu"
        self.reduction = reduction
        self.anchor_target = anchor_target
        self.hard_margin = hard_margin
        self.semi_margin = semi_margin
        self.hard_triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=dis_func,margin=hard_margin)      
        self.semi_triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=dis_func,margin=semi_margin)
        self.dis_func = dis_func  
        self.mode = mode
        self.type_of_triplets = type_of_triplets
                      
    def forward(self, output: Tensor, target: Tensor,labels=None,labels_value=None) -> Tensor:
        """使用度量学习，用Triplet Loss比较排序损失,使用实际的目标值作为锚点(Anchor)"""
        
        import_index = torch.where(labels==CLASS_SIMPLE_VALUE_MAX)[0]
        imp_sec_index = torch.where(labels==CLASS_SIMPLE_VALUE_SEC)[0]
        fail_index = torch.where(labels==0)[0]
        fail_sec_index = torch.where(labels==1)[0]
        # 把三元组数据分别遍为3组，(3,0) (3,1) (2,0),并分别计算三元组损失
        loss_total = 0.0
        acc_total = 0
        # 使用与目标的距离值进行挖掘
        dist_data = self.dis_func(output,target)
        for i in range(4):
            if i==0:
                index_pair = (import_index, fail_index)
            if i==1:
                index_pair = (import_index, fail_sec_index)
            if i==2:
                index_pair = (imp_sec_index, fail_index)
            if i==3:
                index_pair = (fail_index, import_index)
            if self.anchor_target:   
                # 使用与目标的距离值进行挖掘
                a_idx,p_idx,n_idx = self.minering_index_pair_by_target_value(index_pair[0], index_pair[1],dist_data=dist_data,margin=self.semi_margin)   
            else:
                a_idx,p_idx,n_idx = self.minering_index_pair(index_pair[0], index_pair[1],output=output,target=target,margin=self.semi_margin)     
            # 需要保证批次内的可学习样本大于1
            if a_idx.shape[0]>1:
                # 根据配置决定是否使用目标值作为锚定标的   
                if self.anchor_target:
                    pos_data = torch.index_select(output, 0, index_pair[0])  
                    anchor_data = torch.index_select(target, 0, index_pair[0])  
                    neg_data = torch.index_select(target, 0, index_pair[0])   
                else:
                    pos_data = torch.index_select(output, 0, p_idx)  
                    anchor_data = torch.index_select(output, 0, a_idx)
                    neg_data = torch.index_select(output, 0, n_idx)   
                # 可以在不同等级之间的距离使用不同的margin   
                if i==0 or i==3:
                    loss_item = self.hard_triplet_loss(anchor_data,pos_data,neg_data)
                else:
                    loss_item = self.semi_triplet_loss(anchor_data,pos_data,neg_data)
                # 计算准确率
                p_dis = self.dis_func(anchor_data,pos_data)
                n_dis = self.dis_func(anchor_data,neg_data)
                if i==0:
                    acc = torch.sum((n_dis - p_dis - self.hard_margin)>0)/p_dis.shape[0]
                else:
                    acc = torch.sum((n_dis - p_dis - self.semi_margin)>0)/p_dis.shape[0]
                loss_total += loss_item
                acc_total += acc
            else:
                print("no find semihard")
                acc_total += 1
        acc = acc_total/4
        return loss_total,acc 

    def minering_index_pair(self,pos_index,nag_index,output=None,target=None,max_iter_time=1000,margin=0.3):
        """生成positive和negitive数据的配对索引"""
        
        size = pos_index.shape[0] if pos_index.shape[0]<nag_index.shape[0] else nag_index.shape[0]
        target_size = size * max_iter_time
        
        # 根据参数增加到原来的n倍，然后随机打乱顺序
        p_index = pos_index.repeat(1,math.ceil(target_size/pos_index.shape[0])).squeeze()
        p_idx = torch.randperm(p_index.shape[0])
        p_index = p_index[p_idx]
        n_index = nag_index.repeat(1,math.ceil(target_size/nag_index.shape[0])).squeeze()
        n_idx = torch.randperm(n_index.shape[0])
        n_index = n_index[n_idx]
        # 取较小的size进行对齐
        p_index = p_index[:target_size].to(self.device)
        n_index = n_index[:target_size].to(self.device)
        a_index = p_index[torch.randperm(p_index.shape[0])]
        
        # 根据配置决定是否使用同一个pos作为anchor锚点
        if self.anchor_target:
            anchor_data = target[a_index] 
        else:
            anchor_data = output[a_index] 
        pos_data = output[p_index] 
        neg_data = output[n_index] 
        # 根据距离进行筛选
        ap_dist = self.dis_func(anchor_data,pos_data)
        an_dist = self.dis_func(anchor_data,neg_data)
        triplet_margin = (an_dist - ap_dist)         
        threshold_condition = triplet_margin <= margin
        if self.type_of_triplets == "hard":
            threshold_condition &= triplet_margin <= 0
        elif self.type_of_triplets == "semihard":
            threshold_condition &= triplet_margin > 0             
        return (
            a_index[threshold_condition],
            p_index[threshold_condition],
            n_index[threshold_condition],
        ) 

    def minering_index_pair_by_target_value(self,pos_index,nag_index,max_iter_time=2000,dist_data=None,margin=0.3):
        """根据目标值，生成positive和negitive数据的配对索引"""
        
        size = pos_index.shape[0] if pos_index.shape[0]<nag_index.shape[0] else nag_index.shape[0]
        target_size = size * max_iter_time
        
        # 根据参数增加到原来的n倍，然后随机打乱顺序
        p_index = pos_index.repeat(1,math.ceil(target_size/pos_index.shape[0])).squeeze()
        p_idx = torch.randperm(p_index.shape[0])
        p_index = p_index[p_idx]
        n_index = nag_index.repeat(1,math.ceil(target_size/nag_index.shape[0])).squeeze()
        n_idx = torch.randperm(n_index.shape[0])
        n_index = n_index[n_idx]
        # 取较小的size进行对齐
        p_index = p_index[:target_size].to(self.device)
        n_index = n_index[:target_size].to(self.device)
        a_index = p_index
        # 锚定目标距离，进行挖掘
        ap_dist = dist_data[p_index] 
        an_dist = dist_data[n_index] 
        triplet_margin = (an_dist - ap_dist)         
        threshold_condition = triplet_margin <= margin
        if self.type_of_triplets == "hard":
            threshold_condition &= triplet_margin <= 0
        elif self.type_of_triplets == "semihard":
            threshold_condition &= triplet_margin > 0        
        return (
            a_index[threshold_condition],
            p_index[threshold_condition],
            n_index[threshold_condition],
        )                 
           
    def minering_index(self,pos_index,nag_index):
        """生成三元组数据，两两配对positive数据，然后配对到anchors和negitive数据"""
        
        pos_miner = list(itertools.permutations(pos_index.cpu().numpy().tolist(), 2))
        size = nag_index.shape[0]
        pos_miner = random.sample(pos_miner, 10*size)
        a_idx = list(set([item[0] for item in pos_miner]))
        a_idx = torch.Tensor(np.array(a_idx)).long().to(self.device)
        p_idx = [item[1] for item in pos_miner[:size]]
        p_idx = torch.Tensor(np.array(p_idx)).long().to(self.device)
        real_size = min(size,p_idx.shape[0],a_idx.shape[0])
        p_idx = p_idx[:real_size]
        a_idx = a_idx[:real_size]
        n_idx = nag_index[:real_size]
        return a_idx,p_idx,n_idx

    def combine_index(self,pos_index,nag_index):
        """生成三元组数据，使用原数据序列值作为锚点值"""
        
        size = nag_index.shape[0] if nag_index.shape[0]<pos_index.shape[0] else pos_index.shape[0]
        return pos_index[:size],pos_index[:size],nag_index[:size]

class UncertaintyLoss(nn.Module):
    """不确定损失,包括mse，corr以及分类交叉熵损失等"""

    def __init__(self, mse_reduction="mean",device=None,ref_model=None):
        super(UncertaintyLoss, self).__init__()
        
        self.ref_model = ref_model
        self.mse_reduction = mse_reduction
        # self.sigma = loss_sigma
        # sig_params = torch.ones(4, requires_grad=True).to(device)
        # self.sigma = torch.nn.Parameter(sig_params)
        
        self.device = device
        self.epoch = 0
        
        # 涨跌幅分类损失中，需要设置不同分类权重
        vr_loss_weight = get_simple_class_weight()
        vr_loss_weight = torch.from_numpy(np.array(vr_loss_weight)).to(device)
        self.vr_losses = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()]
        
        self.triplet_loss_combine = [TripletLoss(dis_func=self.ccc_distance_torch,type_of_triplets="hard",anchor_target=True,hard_margin=0.4,semi_margin=0.4,device=device),
                                     TripletLoss(dis_func=self.ccc_distance_torch,type_of_triplets="hard",anchor_target=True,hard_margin=0.4,semi_margin=0.4,device=device),
                                     TripletLoss(dis_func=self.ccc_distance_torch,type_of_triplets="hard",anchor_target=True,hard_margin=0.4,semi_margin=0.4,device=device)]
        self.miner_loss_combine = [MinerLoss(pos_margin=0.1,neg_margin=0.3,dis_func=self.ccc_distance_torch),
                                   MinerLoss(pos_margin=0.2,neg_margin=0.4,dis_func=self.ccc_distance_torch),
                                   MinerLoss(pos_margin=0.2,neg_margin=0.4,dis_func=self.ccc_distance_torch)]
        self.dtw_loss = SoftDTWLossPyTorch(gamma=0.1,normalize=True)
        self.rankloss = TripletLoss(reduction='mean',dis_func=self.mse_dis,anchor_target=True,device=device)
        # 设置损失函数的组合模式
        self.loss_mode = 0

    def forward(self, input_ori: Tensor, target_ori: Tensor,optimizers_idx=0,epoch=0,cluster_centers=None):
        """使用MSE损失+相关系数损失，连接以后，使用不确定损失来调整参数"""
 
        (input,vr_combine_class,vr_classes) = input_ori
        (target,target_class,target_info,y_transform) = target_ori
        label_class = target_class[:,0]
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        corr_acc_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        similarity_value = [None,None,None]
        triplet_loss_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        classify_loss_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        # 指标分类
        ce_loss = torch.tensor(0.0).to(self.device) 
        value_diff_loss = torch.tensor(0).to(self.device)        
        label_item = torch.Tensor(np.array([target_info[j]["raise_range"] for j in range(len(target_info))])).to(self.device)
        label_item = normalization(label_item,mode="torch")     
        price_array = torch.Tensor(np.array([target_info[j]["price_array"] for j in range(len(target_info))])).to(self.device)    
        price_range_arr = compute_price_range(price_array)/10
        price_range_arr = price_range_arr[:,-5:]
        # 相关系数损失,多个目标
        for i in range(len(input)):
            if optimizers_idx==i or optimizers_idx==-1:
                input_item = input[i]
                input_item_norm = torch.softmax(input_item.squeeze(),dim=1)
                target_item = target[:,:,i]    
                vr_item_class = vr_classes[i]
                corr_loss_combine[i] += self.miner_loss_combine[i](input_item.squeeze(),target_item,label_class)
                # triplet_loss_combine[i] += self.triplet_online(input_item.squeeze(),label_class,
                #                                     dist_func=self.ccc_distance_torch,target=target_item,margin=0.3)                     
                triplet_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                                                                    target_item,labels=label_class,labels_value=label_item)                    
                # triplet_loss_combine[i] += self.triplet_online(input_item.squeeze(),label_class,
                #                                     dist_func=self.ccc_distance_torch,target=target_item,margin=0.3)                        
                # if i==1:
                #     # corr_loss_combine[i] += self.ccc_loss_comp(input_item.squeeze(),target_item)
                #
                #     # triplet_loss_combine[i] += self.ms_loss(input_item_norm,label_class,dist_func=self.ccc_distance_torch)                 
                #     # triplet_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                #     #                                                 target_item,labels=label_class,labels_value=label_item)                       
                # elif i==2:
                #     # corr_loss_combine[i] = self.ccc_loss_comp(input_item.squeeze(),target_item)
                #     # triplet_loss_combine[i] += self.triplet_online(input_item.squeeze(),label_class,
                #     #                                     dist_func=self.ccc_distance_torch,target=target_item,margin=0.3)   
                #     # triplet_loss_combine[i] += self.ms_loss(input_item_norm,label_class,dist_func=self.ccc_distance_torch)                       
                #     # triplet_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                #     #                                                 target_item,labels=label_class,labels_value=label_item)      
                # else: 
                #     # corr_loss_combine[i] = self.ccc_loss_comp(input_item.squeeze(),target_item)
                #     # triplet_loss_combine[i] += self.triplet_online(input_item.squeeze(),label_class,
                #     #                                     dist_func=self.ccc_distance_torch,target=target_item,margin=0.2)      
                #     # triplet_loss_combine[i] += self.ms_loss(input_item_norm,label_class,dist_func=self.ccc_distance_torch)            
                #     # triplet_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                #     #                                                 target_item,labels=label_class,labels_value=label_item)      
                loss_sum = corr_loss_combine[i] + triplet_loss_combine[i] + classify_loss_combine[i]
                
        # 验证阶段，全部累加
        if optimizers_idx==-1:
            # ce_loss,acc = self.rankloss(vr_combine_class, price_range_arr,label_class)
            # ce_loss = self.triplet_online(vr_combine_class, label_class,target=price_range_arr)
            loss_sum = torch.sum(corr_loss_combine+triplet_loss_combine) + ce_loss + value_diff_loss
            
        return loss_sum,[corr_loss_combine,triplet_loss_combine,classify_loss_combine]
            
    def mse_loss(self,x1, x2,weighted_data=None):
        return torch.mean(self.mse_dis(x1,x2))

    def weighted_mse_loss(self,pred, target, weight, reduction='mean'):
        """
        pred: 预测值，形状 (N, *) 或任意形状
        target: 目标值，形状与 pred 相同
        weight: 权重，形状必须能与 pred 广播（通常与 pred 相同或为 (N,1) 等）
        reduction: 'mean'（默认，除以权重和）、'sum'（直接求和）、'none'（返回逐元素加权误差）
        """
        # 逐元素平方误差
        squared_error = (pred - target) ** 2
        # 应用权重
        weighted_squared_error = weight * squared_error
    
        if reduction == 'mean':
            return weighted_squared_error.sum() / weight.sum()
        elif reduction == 'sum':
            return weighted_squared_error.sum()
        elif reduction == 'none':
            return weighted_squared_error
        else:
            raise ValueError(f"Unknown reduction type: {reduction}")
        
    def cos_loss(self,x1, x2):
        similarity = torch.cosine_similarity(x1, x2, dim=1)
        loss = 1 - similarity
        return loss
        
    def triplet_dis(self,x1, x2):
        return self.ccc_distance(x1,x2)      

    def mse_dis(self,input: Tensor, target: Tensor) -> Tensor:
        loss_arr = (input - target) ** 2
        distance = torch.mean(loss_arr,dim=1)
        return distance       
       
    def pearson_dis(self, input: Tensor, target: Tensor,eps=False):
        if len(input.shape)==1:
            num_outputs = 1
            pearson = torchmetrics.PearsonCorrCoef().to(self.device)
            input_corr = input
            target_corr = target
        else:           
            num_outputs = input.shape[0]
            pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
            input_corr = input.transpose(1,0)
            target_corr = target.transpose(1,0)
            # 为了避免NAN，需要进行校验和调整
            if eps:
                input_corr = adjude_seq_eps(input_corr.permute(1,0)).permute(1,0)
                target_corr = adjude_seq_eps(target_corr.permute(1,0)).permute(1,0)
        distance = 1 - pearson(input_corr, target_corr).pow(2)
        return distance

    def corr_distance(self, input: Tensor, target: Tensor):
        distance = self.pearson_dis(input,target)
        return distance
                    
    def corr_loss_comp(self, input: Tensor, target: Tensor):
        distance = self.pearson_dis(input,target)
        return torch.mean(distance)

    def ccc_loss_comp(self, input: Tensor, target: Tensor):
        """一致性相关系数计算"""
        ccc_loss = self.ccc_distance(input, target)
        return torch.mean(ccc_loss)

    def ccc_distance_torch(self,x,y):
        flag_numpy = 0
        if isinstance(x,np.ndarray):
            flag_numpy = 1
            x = torch.Tensor(x)
            y = torch.Tensor(y)
        if len(x.shape)>1:
            x = x.transpose(1,0)
            y = y.transpose(1,0)
            concordance = ConcordanceCorrCoef(num_outputs=x.shape[-1]).to(self.device)
        else:
            concordance = ConcordanceCorrCoef().to(self.device)
        dis = 1 - concordance(x, y)
        if flag_numpy==1:
            dis = dis.cpu().numpy()
        return dis

    def ccc_distance_3d(self,x,y):
        x_reshape = x.reshape(-1,x.shape[2])
        y_reshape = y.reshape(-1,y.shape[2])
        dis = self.ccc_distance_torch(x_reshape, y_reshape)
        dis = dis.reshape(x.shape[0],x.shape[1],1)
        return dis
       
    def ccc_distance(self,input_ori,target_ori):
        if len(input_ori.shape)==1:
            input_with_dims = input_ori.unsqueeze(0)
        else:
            input_with_dims = input_ori
        if len(target_ori.shape)==1:
            target_with_dims = target_ori.unsqueeze(0)    
        else:
            target_with_dims = target_ori                    
        input = input_with_dims.flatten()
        target = target_with_dims.flatten()
        corr_tensor = torch.stack([input,target],dim=0)
        cor = torch.corrcoef(corr_tensor)[0][1]
        var_true = torch.var(target)
        var_pred = torch.var(input)
        sd_true = torch.std(target)
        sd_pred = torch.std(input)
        numerator = 2*cor*sd_true*sd_pred
        mse_part = self.mse_dis(input_with_dims,target_with_dims)
        denominator = var_true + var_pred + mse_part
        if len(input_ori.shape)==1 and denominator==0:
            return torch.tensor(0.0).to(numerator.device) 
        compli_index = torch.where(denominator!=0)[0]
        ccc = numerator/denominator[compli_index]
        ccc_loss = 1 - ccc
        return ccc_loss     
    
    def ccc_distance_noinverse(self,input_ori,target_ori):
        if len(input_ori.shape)==1:
            input_with_dims = input_ori.unsqueeze(0)
        else:
            input_with_dims = input_ori
        if len(target_ori.shape)==1:
            target_with_dims = target_ori.unsqueeze(0)    
        else:
            target_with_dims = target_ori                    
        input = input_with_dims.flatten()
        target = target_with_dims.flatten()
        corr_tensor = torch.stack([input,target],dim=0)
        cor = torch.corrcoef(corr_tensor)[0][1]
        var_true = torch.var(target)
        var_pred = torch.var(input)
        sd_true = torch.std(target)
        sd_pred = torch.std(input)
        numerator = 2*cor*sd_true*sd_pred
        mse_part = self.mse_dis(input_with_dims,target_with_dims)
        denominator = var_true + var_pred + mse_part
        ccc = numerator/denominator
        return ccc  
        
    def compute_dtw_loss(self,input,target):
        input_real = torch.unsqueeze(input,-1)
        target_real = torch.unsqueeze(target,-1)
        loss = self.dtw_loss(input_real, target_real).mean()       
        return loss
    
    def triplet_online(self,embeddings,labels,target=None,dist_func=None,margin=0.3):
        """在线triplet挖掘及损失计算"""
        
        if dist_func is None:
            dist_func = distances.LpDistance(p=2)
        miner = TripletTargetMiner(type_of_triplets="semihard",distance=dist_func,margin=margin)
        loss_func = TripletTargetLoss(margin=margin,distance=dist_func)        
        hard_pairs = miner(embeddings, labels,ref_target=target)
        loss = loss_func(embeddings, labels, hard_pairs,ref_target=target)      
        # acc = torch.sum((n_dis - p_dis - self.semi_margin)>0)/p_dis.shape[0] 
        return loss 
    
    def hsan_loss(self,output,index=0):    
        """HSAN模式的损失计算"""   
        
        Z1, Z2, E1, E2 = output
        loss,S = self.hsan_loss_combine[index](Z1, Z2, E1, E2)
        # 同时返回综合相似度数值后续使用
        return loss,S
        
    def kl_loss(self,input,target): 
        kl = F.kl_div(input.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='mean')
        return kl

    def weighted_l1_loss(self,inputs, targets, weights=None):
        loss = F.l1_loss(inputs, targets, reduction='none')
        if weights is not None:
            loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss   

def weighted_spearman_loss(
    pred_scores, true_labels, external_weights,temperature=0.01
):
    """
    可微的带权Spearman损失：用soft排序+gather保留梯度
    Args:
        pred_scores: (batch_size, docs_per_query) - 模型预测分数（带梯度）
        true_labels: (batch_size, docs_per_query) - 真实标签
        external_weights: (batch_size, docs_per_query) - 外部权重
        temperature: 温度系数（越小，soft排序越接近硬排序，梯度越稳定）
    Returns:
        avg_loss: 批次平均损失（可微）
    """
    batch_size, docs_per_query = pred_scores.shape
    
    # -------------------------- 关键：可微排名计算（替代argsort） --------------------------
    def soft_rank(scores, descending=True, temp=temperature):
        """
        可微排名计算：用softmax近似排序，返回每个元素的“软排名”（1-based）
        Args:
            scores: (batch_size, docs_per_query)
            descending: 是否降序
            temp: 温度系数
        Returns:
            soft_ranks: (batch_size, docs_per_query) - 可微的软排名
        """
        # 步骤1：生成两两比较的矩阵 (batch_size, docs_per_query, docs_per_query)
        scores_expanded = scores.unsqueeze(2)  # (B, N, 1)
        scores_tiled = scores.unsqueeze(1)     # (B, 1, N)
        
        # 步骤2：计算两两比较的相似度（可微）
        if descending:
            diff = scores_expanded - scores_tiled  # 降序：score_i > score_j → 排名更靠前
        else:
            diff = scores_tiled - scores_expanded
        
        # 步骤3：用sigmoid近似“是否大于”的判断（可微）
        pairwise_similarity = torch.sigmoid(diff / temp)
        
        # 步骤4：计算软排名 = 1 + 所有比当前分数高的样本数（可微）
        soft_ranks = 1.0 + torch.sum(pairwise_similarity, dim=2) - 0.5  # 减0.5修正自身比较
        
        return soft_ranks
    
    # -------------------------- 计算软排名（可微） --------------------------
    # 预测分数的软排名（降序）
    pred_soft_ranks = soft_rank(pred_scores, descending=True)
    # 真实标签的软排名（降序，标签越高排名越靠前）
    true_soft_ranks = soft_rank(true_labels, descending=True)
    
    # -------------------------- 用gather确认梯度链路（可选，强化梯度保留） --------------------------
    # 对pred_scores按软排名排序（gather保留梯度）
    sorted_indices = torch.argsort(pred_soft_ranks, dim=1, descending=False)  # 排名1在前
    pred_scores_sorted = torch.gather(pred_scores, dim=1, index=sorted_indices)
    # 仅用于激活梯度（gather不改变值，但保留链路）
    pred_soft_ranks = pred_soft_ranks + (pred_scores_sorted - pred_scores_sorted).detach()
    
    # -------------------------- 加权Spearman损失计算（全可微） --------------------------
    # 排名差值 d_i（可微）
    d = pred_soft_ranks - true_soft_ranks  # (B, N)
    
    # 加权平方差 sum(w_i * d_i²)（可微）
    weighted_d2 = external_weights * (d ** 2)
    sum_weighted_d2 = torch.sum(weighted_d2, dim=1)  # (B,)
    
    # 加权Spearman系数（可微）
    sum_weights = torch.sum(external_weights, dim=1)  # (B,)
    denominator = sum_weights * (docs_per_query ** 2 - 1)  # (B,)
    spearman_r = 1 - (6 * sum_weighted_d2) / (denominator + 1e-10)  # (B,)
    
    # 损失 = 1 - 相关系数（最小化损失=最大化排名一致性）
    loss_per_query = 1 - spearman_r  # (B,)
    avg_loss = torch.mean(loss_per_query)  # 批次平均（可微）
    
    return avg_loss

class WeightedSpearmanLoss(nn.Module):
    def __init__(self, tau=1.0, reduction='mean'):
        """
        tau: 温度参数，控制软排序的平滑程度。tau 越小越接近真实排序，但梯度越稀疏。
        reduction: 'mean' 返回批次中所有样本的损失平均值；'sum' 返回总和。
        """
        super().__init__()
        self.tau = tau
        self.reduction = reduction

    def _soft_rank(self, scores):
        """
        输入: scores 形状 (N, )，一维张量
        输出: soft_ranks 形状 (N, )，每个元素的近似排名（从1开始）
        """
        N = scores.size(0)
        # 计算两两差值矩阵: diff[i][j] = scores[j] - scores[i]   (j 与 i 比较)
        diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (N, N)
        # 使用 sigmoid 近似计数比 i 大的元素个数
        # 对于升序排名：排名 = 1 + (比 i 小的元素个数) = 1 + sum_{j != i} [s_j < s_i]
        # 等价于 1 + sum_{j != i} sigmoid((s_i - s_j)/tau)
        # 这里用 (s_i - s_j)/tau 而不是 (s_j - s_i)，因为 sigmoid(x) 当 x>0 时≈1
        soft_ranks = 1 + torch.sigmoid((scores.unsqueeze(1) - scores.unsqueeze(0)) / self.tau).sum(dim=1)
        # 减去自身项（当 j=i 时，s_i - s_i = 0，sigmoid(0)=0.5，需要去掉）
        soft_ranks = soft_ranks - 0.5  # 因为对角线贡献了 0.5
        return soft_ranks

    def forward(self, pred, target, weight=None):
        """
        pred: 预测值，形状 (N, ) 或 (batch_size, ...)，将展平为 (N,)
        target: 目标值，形状与 pred 相同
        weight: 样本权重，形状与 pred 相同，或为 None（等权重）
        """
        pred = pred.view(-1)
        target = target.view(-1)
        N = pred.size(0)

        if weight is None:
            weight = torch.ones_like(pred)
        else:
            weight = weight.view(-1)

        # 计算软排名
        pred_rank = self._soft_rank(pred)
        target_rank = self._soft_rank(target)

        # 计算加权平方差
        diff = pred_rank - target_rank
        weighted_diff_sq = weight * (diff ** 2)

        # 计算分母：加权总样本数的立方减去加权总样本数
        sum_w = weight.sum()
        denom = sum_w ** 3 - sum_w
        if denom == 0:
            # 避免除零（例如所有权重相等且只有一个样本？但这种情况很少见）
            denom = 1e-8

        # 计算加权 Spearman 系数
        numerator = 6 * weighted_diff_sq.sum()
        rho_w = 1 - numerator / denom

        # 损失：我们希望最大化 Spearman，所以取 1 - rho_w（最小化）
        loss = 1 - rho_w

        if self.reduction == 'mean':
            return loss.mean() if loss.numel() > 1 else loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
