from  torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch
import torchmetrics
from torch import Tensor
from torch.nn import TripletMarginWithDistanceLoss
import numpy as np
import random
from tslearn.metrics import SoftDTWLossPyTorch
import itertools
from cus_utils.common_compute import normalization,adjude_seq_eps
from cus_utils.encoder_cus import transform_slope_value
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_SEC,CLASS_SIMPLE_VALUE_MAX,get_simple_class_weight
from numba.cuda.cudadrv import ndarray


mse_weight = torch.tensor(np.array([1,2,3,4,5]))  
mse_scope_weight = torch.tensor(np.array([1,2,3,4]))  

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchmetrics.regression import ConcordanceCorrCoef

distance = distances.LpDistance()
reducer = reducers.ThresholdReducer(low=0)


class NPairLoss(_Loss):

    def __init__(self, l2_reg=1,dis_func=None):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.dis_func = dis_func

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.dis_func(anchors, positives)

        return losses.sum()

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]
    
class TripletLoss(_Loss):
    """自定义三元组损失函数"""
    
    __constants__ = ['reduction']

    def __init__(self, hard_margin=0.6,semi_margin=0.4,dis_func=None,anchor_target=False,mode=1,
                 reduction: str = 'mean',device=None,caller=None) -> None:
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
                      
    def forward(self, output: Tensor, target: Tensor,labels=None,labels_value=None) -> Tensor:
        """使用度量学习，用Triplet Loss比较排序损失,使用实际的目标值作为锚点(Anchor)"""
        
        import_index = torch.where(labels==CLASS_SIMPLE_VALUE_MAX)[0]
        imp_sec_index = torch.where(labels==CLASS_SIMPLE_VALUE_SEC)[0]
        fail_index = torch.where(labels==0)[0]
        fail_sec_index = torch.where(labels==1)[0]
        other_index = torch.where(labels!=CLASS_SIMPLE_VALUE_MAX)[0]
        # 把三元组数据分别遍为3组，(3,0) (3,1) (2,0),并分别计算三元组损失
        loss_total = 0.0
        acc_total = 0
        for i in range(4):
            if i==0:
                a_idx,p_idx,n_idx = self.minering_index(import_index, fail_index)
            if i==1:
                a_idx,p_idx,n_idx = self.minering_index(import_index, fail_sec_index)       
            if i==2:
                a_idx,p_idx,n_idx = self.minering_index(imp_sec_index, fail_index)     
            if i==3:
                a_idx,p_idx,n_idx = self.minering_index(fail_index, import_index)                    
            pos_data = torch.index_select(output, 0, p_idx)    
            # 根据配置决定是否使用目标值作为锚定标的   
            if self.anchor_target:
                anchor_data = torch.index_select(target, 0, a_idx)  
                neg_data = torch.index_select(target, 0, n_idx)   
            else:
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
            # # 不同目标值，使用不同的挖掘数据指标
            # if self.mode==1 and i!=3:
            #     # 对于前两个目标值，不使用反向指标
            #     loss_total += loss_item
            # if self.mode==2 and i==3:
            #     # 对于第三个目标值,只使用反向指标
            #     loss_total += loss_item                
            acc_total += acc
        acc = acc_total/4
        return loss_total,acc 
    
    def minering_index(self,pos_index,nag_index):
        """生成三元组数据，两两配对positive数据，然后配对到anchors和negitive数据"""
        
        pos_miner = list(itertools.permutations(pos_index.cpu().numpy().tolist(), 2))
        size = nag_index.shape[0]
        pos_miner = random.sample(pos_miner, size)
        a_idx = torch.Tensor(np.array([item[0] for item in pos_miner[:size]])).long().to(self.device)
        p_idx = torch.Tensor(np.array([item[1] for item in pos_miner[:size]])).long().to(self.device)
        n_idx = nag_index[:size]
        return a_idx,p_idx,n_idx

    def combine_index(self,pos_index,nag_index):
        """生成三元组数据，使用原数据序列值作为锚点值"""
        
        size = nag_index.shape[0] if nag_index.shape[0]<pos_index.shape[0] else pos_index.shape[0]
        return pos_index[:size],pos_index[:size],nag_index[:size]

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
        self.vr_loss = nn.MSELoss()
        
        self.triplet_loss_combine = [TripletLoss(dis_func=self.ccc_distance_torch,anchor_target=True,hard_margin=0.5,semi_margin=0.3,device=device),
                                     TripletLoss(dis_func=self.ccc_distance_torch,anchor_target=True,hard_margin=0.5,semi_margin=0.3,device=device),
                                     TripletLoss(dis_func=self.mse_dis,anchor_target=True,hard_margin=0.5,semi_margin=0.3,device=device,mode=2)]
        self.n_paire_loss = NPairLoss(dis_func=self.pearson_dis)
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
        corr_acc_combine = torch.Tensor(np.array([0 for i in range(len(input))])).to(self.device)
        # 指标分类
        ce_loss = torch.tensor(0.0).to(self.device) 
        value_diff_loss = torch.tensor(0).to(self.device)         
        # 相关系数损失,多个目标
        for i in range(len(input)):
            if optimizers_idx==i or optimizers_idx==-1:
                input_item = input[i]
                target_item = target[:,:,i]    
                label_item = torch.Tensor(np.array([target_info[j]["raise_range"] for j in range(len(target_info))])).to(self.device)
                label_item = normalization(label_item,mode="torch")
                vr_item_class = vr_classes[i][:,0] 
                if i==1:
                    corr_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                                                                    target_item,labels=label_class,labels_value=label_item)   
                    # corr_loss_combine[i] += self.ccc_loss_comp(input_item.squeeze(),target_item)
                elif i==2:
                    corr_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                                                                    target_item,labels=label_class,labels_value=label_item) 
                    # corr_loss_combine[i] = self.n_paire_loss(input_item.squeeze(),label_class)
                else: 
                    corr_loss_combine[i],corr_acc_combine[i] = self.triplet_loss_combine[i](input_item.squeeze(-1), 
                                                                    target_item,labels=label_class,labels_value=label_item)   
                    # corr_loss_combine[i] = self.n_paire_loss(input_item.squeeze(),label_class)
                # corr_loss_combine[i] += self.mse_loss(input_item.squeeze(),target_item)
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
            
        return loss_sum,[corr_loss_combine,ce_loss,corr_acc_combine]

    def mse_loss(self,x1, x2):
        return torch.mean(self.mse_dis(x1,x2))
    
    def triplet_dis(self,x1, x2):
        return self.ccc_distance(x1,x2)      

    def mse_dis(self,input: Tensor, target: Tensor) -> Tensor:
        loss_arr = (input - target) ** 2
        distance = torch.mean(loss_arr,dim=1)
        return distance       
       
    def pearson_dis(self, input: Tensor, target: Tensor):
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
            input_corr = adjude_seq_eps(input_corr.permute(1,0)).permute(1,0)
            target_corr = adjude_seq_eps(target_corr.permute(1,0)).permute(1,0)
        distance = 1 - pearson(input_corr, target_corr) 
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
        ccc = numerator/denominator
        ccc_loss = 1 - ccc
        return ccc_loss     
    
    def compute_dtw_loss(self,input,target):
        input_real = torch.unsqueeze(input,-1)
        target_real = torch.unsqueeze(target,-1)
        loss = self.dtw_loss(input_real, target_real).mean()       
        return loss
  
