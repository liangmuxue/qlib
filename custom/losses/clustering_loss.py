import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import pairwise_distances,batch_cov_comp

class ClusteringLoss(UncertaintyLoss):
    """基于聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(ClusteringLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):
        """套用dinknet中的聚类损失计算，使用具有梯度的聚类中心参数进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,future_target_adj,pca_target) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        kl_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        fake_loss = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        label_class = target_class[:,:,0].long()
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[:,:,:,i]
                # real_target = past_target[:,:,i]
                output_item = output[i] 
                x_bar, q, pred, pred_value,z = output_item  
                x_bar = x_bar.view(x_bar.shape[0]*x_bar.shape[1],-1)
                real_target = real_target.view(real_target.shape[0]*real_target.shape[1],-1)
                corr_loss_combine[i] = self.corr_loss_comp(x_bar,real_target)
                if mode=="pretrain":
                    # 如果属于特征值阶段，则只比较特征距离
                    loss_sum = loss_sum + corr_loss_combine[i]
                else:  
                    # 使用相关距离，进行横向损失比较
                    ce_loss[i] = self.mse_loss(pred[:,:,0],pca_target)   
                    loss_sum = loss_sum + corr_loss_combine[i] + kl_loss[i] + ce_loss[i]
        return loss_sum,[corr_loss_combine,kl_loss,ce_loss]

    def mahalanobis_distance(self,x,y): 
        """计算马氏距离"""
        
        delta = x - y
        delta_T = delta.permute(0,2,1)
        # 计算协方差矩阵
        S = batch_cov_comp(x,y)
        np.save("custom/data/asis/KDJ_2000_202010/s_test.npy",S.detach().cpu().numpy())
        # 逆矩阵
        S_inv = torch.linalg.pinv(S)
        # 向量与矩阵点乘
        bm1_test = torch.bmm(delta_T, S)
        bm1 = torch.bmm(delta_T, S_inv)
        m = torch.bmm(bm1, delta)
        return torch.sqrt(m)
           
    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    
    
class MtgLoss(UncertaintyLoss):
    """基于距离矩阵的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(MtgLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0):

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,target_info,past_target) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        triplet_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i][...,0]
                output_item = output_item.permute(0,2,1)
                output_item = torch.reshape(output_item,(output_item.shape[0]*output_item.shape[1],output_item.shape[2]))
                target_item = target[:,:,:,i]    
                target_item = torch.reshape(target_item,(target_item.shape[0]*target_item.shape[1],target_item.shape[2]))
                # 分别计算输出值与目标值的距离矩阵
                # mat_output = pairwise_distances(output_item,distance_func=self.ccc_distance_torch)
                # mat_target = pairwise_distances(target_item,distance_func=self.ccc_distance_torch)
                # 计算距离矩阵的MSE损失作为实际损失
                # corr_loss_combine[i] = self.mse_loss(mat_output, mat_target)                
                corr_loss_combine[i] = self.ccc_loss_comp(output_item, target_item)     
                loss_sum += corr_loss_combine[i] 
        return loss_sum,[corr_loss_combine,triplet_loss_combine,similarity_value[i]]
            
