import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import batch_cov,batch_cov_comp,eps_rebuild,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from cus_utils.metrics import pca_apply,guass_cdf,sampler_normal
from .fds_loss import weighted_mse_loss
import geomloss
from .quanlity_loss import QuanlityLoss

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def calculate_gaussian_kl_divergence(m1,m2,std1,std2):
    v1 = torch.square(std1)
    v2 = torch.square(std2)
    return torch.log(std1 / std2) + torch.div(torch.add(v1, torch.square(m1 - m2)), 2 * v2 ) - 0.5

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum(w[ind[0],ind[1]])*1.0/Y_pred.size, w

class ClusteringLoss(UncertaintyLoss):
    """基于聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(ClusteringLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
        self.ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):
        """套用dinknet中的聚类损失计算，使用具有梯度的聚类中心参数进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,future_target_slope,pca_target,price_range) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        kl_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        fake_loss = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        label_class = target_class[:,:,0].long()
        ot_loss_detail = []
        
        price_range = price_range.squeeze(-1)
        price_range_t = F.softmax(price_range, dim=1)
                
        for i in range(len(output)):
            
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[:,:,:,i]
                # pca_target_item = pca_target[:,:,:,i]
                # real_target = past_target[:,:,i]
                output_item = output[i] 
                x_bar, q, pred, pred_value,z = output_item  
                x_bar = x_bar.view(x_bar.shape[0]*x_bar.shape[1],-1)
                pre_target = real_target.view(real_target.shape[0]*real_target.shape[1],-1)
                corr_loss_combine[i] = self.mse(x_bar,pre_target)
                if mode=="pretrain":
                    # 如果属于特征值阶段，则只比较特征距离
                    loss_sum = loss_sum + corr_loss_combine[i]
                else:  
                    # 批量计算协方差矩阵，使用单条记录用于比较分布规律
                    cov_matrix = batch_cov(target[...,i])   
                    cov_target = cov_matrix[:,:,0]                     
                    pred = pred.squeeze(-1)
                    pred_t = F.log_softmax(pred, dim=1)
                    cov_target_t = F.softmax(cov_target, dim=1)
                    kl_loss[i] = nn.KLDivLoss(reduction="batchmean")(pred_t, cov_target_t)
                    # kl_loss[i] = self.ccc_loss_comp(pred, cov_target)
                    # 使用OT距离，进行横向损失比较
                    # ot_loss_detail.append(ot_loss)
                    # ce_loss[i] = torch.mean(ot_loss)  
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
    
class VadeLoss(UncertaintyLoss):
    """基于变分聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(VadeLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):

        """套用vade中的聚类损失计算，使用聚类模式进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,past_target,past_covariates) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        elbu_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        fake_loss = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        label_class = target_class[:,0].long()
        ot_loss_detail = []
        
        for i in range(len(output)):
            
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                # pca_target_item = pca_target[:,:,:,i]
                # real_target = past_target[:,:,i]
                output_item = output[i] 
                x_bar, mu, log_sigma2, _,_elbu_loss = output_item  
                if mode=="pretrain":
                    detail_loss = None
                    x_bar = x_bar.permute(1,0,2).squeeze()
                    corr_loss_combine[i] = self.ccc_loss_comp(x_bar,real_target)
                    # 如果属于特征值阶段，则只比较特征距离
                    loss_sum = loss_sum + corr_loss_combine[i]
                else:  
                    elbu_loss[i],detail_loss,yita_c = _elbu_loss
                    ce_loss[i] = nn.CrossEntropyLoss()(yita_c,label_class)
                    # 使用elbo中的预测损失
                    corr_loss_combine[i] = detail_loss[0]
                    # 使用ELBO损失，最大化下界
                    loss_sum = loss_sum + ce_loss[i] + corr_loss_combine[i]
        return loss_sum,[corr_loss_combine,ce_loss,detail_loss]

class VaRELoss(UncertaintyLoss):
    """基于变分聚类的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(VaRELoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):

        """套用vade中的聚类损失计算，使用聚类模式进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,past_target,past_covariates) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        elbu_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        comb_detail_loss = []
        label_class = target_class[:,0].long()
        
        for i in range(len(output)):
            
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                
                output_item = output[i] 
                x_bar, mu, log_sigma2, _,loss_tup= output_item  
                x_bar = x_bar.permute(1,0,2).squeeze()
                p_conv = past_covariates[i]
                corr_loss_combine[i] += self.ccc_loss_comp(x_bar,real_target)
                if mode=="pretrain":
                    # 如果属于特征值阶段，则只比较特征距离
                    loss_sum = loss_sum + corr_loss_combine[i]
                else:  
                    elbu_loss[i] = loss_tup
                    # 使用ELBO损失，最大化下界
                    loss_sum = loss_sum + elbu_loss[i] + corr_loss_combine[i]
        return loss_sum,[corr_loss_combine,elbu_loss,comb_detail_loss]

class SdcnLoss(UncertaintyLoss):
    """基于位置关系比对的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(SdcnLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):

        """使用预测标准距离结合位置比对模式进行计算"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,pca_target) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        cls_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        label_class = target_class[:,0].long()
        
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                pca_target_item = pca_target[...,i]
                output_item = output[i] 
                x_bar, z_ca, lattend, _,output_class = output_item  
                corr_loss_combine[i] = self.ccc_loss_comp(x_bar,real_target)
                # 追加计算目标数据的分类损失
                cls_loss[i] = nn.CrossEntropyLoss()(output_class,label_class)                
                if mode=="pretrain":
                    # 如果属于特征值阶段，则只比较特征距离
                    loss_sum = loss_sum + corr_loss_combine[i] + cls_loss[i]
                else:  
                    # 对降维后的二维数据，进行位置关系比较,同时使用分类损失
                    ce_loss[i] = self.mse_loss(z_ca, pca_target_item) + cls_loss[i]
                    loss_sum = loss_sum + ce_loss[i] + corr_loss_combine[i] + cls_loss[i]
        return loss_sum,[corr_loss_combine,ce_loss,cls_loss,None]

class CovCnnLoss(UncertaintyLoss):
    """基于协方差关系比对的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(CovCnnLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):
        """协方差结果进行比对"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,pca_target) = target_ori
        corr_loss_combine = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        similarity_value = [None,None,None]
        cls_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标
        label_class = target_class[:,0].long()
        
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                pca_target_item = pca_target[...,i]
                output_item = output[i] 
                cls,features,_ = output_item  
                # 计算目标数据的分类损失
                cls_loss[i] = nn.CrossEntropyLoss()(cls,label_class)       
                # 降维目标之间的欧氏距离         
                ce_loss[i] = self.mse_loss(features, pca_target_item) + cls_loss[i]
                loss_sum = loss_sum + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss_combine,ce_loss,cls_loss,None]

class MlpLoss(UncertaintyLoss):
    """基于MLP的损失函数"""
    
    def __init__(self,ref_model=None,device=None):
        super(MlpLoss, self).__init__(ref_model=ref_model,device=device)
        self.ref_model = ref_model
        self.device = device  
        
        reducer = reducers.ThresholdReducer(low=0)
        self.triplet_loss = losses.TripletMarginLoss(margin=0.03, reducer=reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=0.03,type_of_triplets="semihard"
        )     
        self.quan_loss = QuanlityLoss(device=device)   
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):
        """Multiple Loss Combine"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,last_target,price_target) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        tar_cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 相关系数损失,多个目标R
        label_class = target_class[:,0].long()
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                last_target_item = last_target[...,i:i+1]
                # pca_target_item = normalization(pca_target_item, mode="torch")
                output_item = output[i] 
                x_bar,z,cls,tar_cls,x_smo = output_item  
                # 预测值的一致性损失
                corr_loss[i] = self.ccc_loss_comp(x_bar, real_target)
                # 计算价格区间损失
                # cls_loss[i] = 100 * self.mse_loss(cls, price_target)
                             
                # 降维目标之间的欧氏距离         
                ce_loss[i] = 10 * self.mse_loss(x_smo, last_target_item)
                # ce_loss[i] = self.quan_loss.compute_loss(x_smo.unsqueeze(1).unsqueeze(1), last_target_item)
                loss_sum = loss_sum + corr_loss[i] + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]    


class Mlp3DLoss(UncertaintyLoss):
    """基于MLP的损失函数，以日期维度进行整合的3D版本"""
    
    def __init__(self,ins_dim,ref_model=None,device=None):
        super(Mlp3DLoss, self).__init__(ref_model=ref_model,device=device)
        
        # 股票数量维度
        self.ins_dim = ins_dim
        self.ref_model = ref_model
        self.device = device  
        
        reducer = reducers.ThresholdReducer(low=0)
        self.triplet_loss = losses.TripletMarginLoss(margin=0.03, reducer=reducer)
        self.mining_func = miners.TripletMarginMiner(
            margin=0.03,type_of_triplets="semihard"
        )     
        self.quan_loss = QuanlityLoss(device=device)   
        
    def forward(self, output_ori,target_ori,optimizers_idx=0,mode="pretrain"):
        """Multiple Loss Combine"""

        (output,vr_combine_class,vr_classes) = output_ori
        (target,target_class,last_target) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 忽略目标缺失值的损失计算,找出符合比较的索引
        keep_index = torch.where(target_class>=0)[0]
        target_class_3d = target_class.reshape(int(target_class.shape[0]/self.ins_dim),self.ins_dim)
        batch_size = target_class_3d.shape[0]
        # 多个目标损失
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                last_target_item = last_target[...,i:i+1]
                output_item = output[i] 
                x_bar,z,cls,tar_cls,x_smo = output_item  
                # 预测值的一致性损失,忽略目标缺失值的损失计算
                corr_loss[i] = self.ccc_loss_comp(x_bar, real_target[keep_index])
                # 降维目标之间的欧氏距离   ,忽略目标缺失值的损失计算      
                ce_loss[i] = 10 * self.mse_loss(x_smo, last_target_item[keep_index])
                
                # last_target_item_3d = last_target_item.reshape(batch_size,self.ins_dim)
                # whole_cls = torch.tensor(0.0).to(self.device) 
                # continue_index = 0
                # # 分别针对每批次反向计算出当前有效数量
                # keep_size_mapping = [0]
                # for j in range(batch_size):
                #     ingore_index_inner = torch.where(target_class_3d[j]<0)[0]
                #     keep_num = self.ins_dim - ingore_index_inner.shape[0]
                #     continue_index = continue_index + keep_num   
                #     keep_size_mapping.append(continue_index)               
                # # 计算总体比较的一致性损失,需要剔除缺失值后比较
                # for j in range(batch_size):
                #     keep_index_inner = torch.where(target_class_3d[j]>=0)[0]
                #     x_smo_inner = x_smo[keep_size_mapping[j]:keep_size_mapping[j+1]]
                #     last_target_inner = last_target_item_3d[j]
                #     whole_cls += self.ccc_loss_comp(x_smo_inner.squeeze(), last_target_inner[keep_index_inner])
                # cls_loss[i] = whole_cls/batch_size
                loss_sum = loss_sum + corr_loss[i] + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]    
    