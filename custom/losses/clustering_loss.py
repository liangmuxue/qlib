import numpy as np
import torch
from torch.nn.modules.loss import _Loss
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import pairwise_distances,pairwise_compare

class ClusteringLoss(UncertaintyLoss):
    """基于聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(ClusteringLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,cluster_centers=None,optimizers_idx=0):
        """套用dinknet中的聚类损失计算，使用具有梯度的聚类中心参数进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,target_info,y_transform) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        triplet_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                cluster_center = cluster_centers[i]
                output_item = output[i][:,:,0]
                target_item = target[:,:,i]    
                # 实现与聚类簇心的距离计算，以及损失计算
                sample_center_distance = pairwise_compare(cluster_center,output_item,distance_func=self.ccc_distance_torch)
                # 取每个类别距离的最小值，视为属于这个类别的点
                sample_center_distance = torch.min(sample_center_distance,dim=0)[0]
                center_distance = pairwise_compare(cluster_center,cluster_center,distance_func=self.ccc_distance_torch)
                self.no_diag(center_distance, cluster_center.shape[0])
                corr_loss_combine[i] = 10 * sample_center_distance.mean() - center_distance.mean()
                loss_sum += corr_loss_combine[i] 
        return loss_sum,[corr_loss_combine,triplet_loss_combine,similarity_value[i]]
            
    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()    
    
class DistanceMetricLoss(UncertaintyLoss):
    """基于距离矩阵的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(DistanceMetricLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,cluster_centers=None):

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,target_info,y_transform) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        triplet_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i]
                target_item = target[:,:,i]    
                # 分别计算输出值与目标值的距离矩阵
                mat_output = pairwise_distances(output_item,distance_func=self.ccc_distance_torch)
                mat_target = pairwise_distances(target_item,distance_func=self.ccc_distance_torch)
                # 计算距离矩阵的MSE损失作为实际损失
                corr_loss_combine[i] = self.mse_loss(mat_output, mat_target)                
                loss_sum += corr_loss_combine[i] 
        return loss_sum,[corr_loss_combine,triplet_loss_combine,similarity_value[i]]
            
