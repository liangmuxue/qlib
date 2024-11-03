import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import batch_cov,batch_cov_comp,eps_rebuild,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from darts_pro.data_extension.industry_mapping_util import IndustryMappingUtil
import geomloss
from .quanlity_loss import QuanlityLoss

from pytorch_metric_learning import distances, losses, miners, reducers, testers

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
        (target,target_class,last_target,rank_data) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 忽略目标缺失值的损失计算,找出符合比较的索引
        keep_index = torch.where(target_class.reshape(-1)>=0)[0]
        rank_data = rank_data.reshape(-1)[keep_index]
        # 多个目标损失
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                real_target = real_target.reshape(-1,real_target.shape[-1])[keep_index]
                last_target_item = last_target[...,i].reshape(-1)[keep_index]
                output_item = output[i] 
                x_bar,sv_combine = output_item  
                x_bar = x_bar.squeeze(-1)
                x_bar = x_bar.reshape(-1,x_bar.shape[-1])[keep_index]
                sv = sv_combine[...,0].reshape(-1)[keep_index]
                rank_v = sv_combine[...,1].reshape(-1)[keep_index]
                # 预测值的一致性损失,忽略目标缺失值的损失计算
                corr_loss[i] = self.ccc_loss_comp(x_bar.unsqueeze(-1), real_target.unsqueeze(-1))
                # 分别计算排名损失，以及总体数据损失     
                ce_loss[i] = 10 * self.mse_loss(sv.unsqueeze(-1), last_target_item.unsqueeze(-1))
                cls_loss[i] = 10 * self.mse_loss(rank_v.unsqueeze(-1), rank_data.unsqueeze(-1))
                
                loss_sum = loss_sum + corr_loss[i] + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]    
    
class Indus3DLoss(UncertaintyLoss):
    """基于行业分类整合的损失函数，以日期维度进行整合的3D版本"""
    
    def __init__(self,ins_dim,ref_model=None,device=None):
        super(Indus3DLoss, self).__init__(ref_model=ref_model,device=device)
        
        # 股票数量维度
        self.ins_dim = ins_dim
        self.ref_model = ref_model
        self.device = device  
        
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,mode="pretrain"):
        """Multiple Loss Combine"""

        (output,_,_) = output_ori
        (target,target_class,last_targets,indus_targets) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([1 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 忽略目标缺失值的损失计算,找出符合比较的索引
        keep_index_bool_flatten = target_class.reshape(-1)>=0
        keep_index_flatten = torch.where(keep_index_bool_flatten)[0]
        # 缺失值长度，需要在分类内部按照批次区分
        keep_index_bool = keep_index_bool_flatten.reshape(target_class.shape[0],-1)
        # 多个目标损失
        map_size = len(sw_ins_mappings)
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                real_target = real_target.reshape(-1,real_target.shape[-1])[keep_index_flatten]
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                x_bar,sv_instru,sv_indus = output_item  
                x_bar = x_bar.squeeze(-1)
                x_bar = x_bar.reshape(-1,x_bar.shape[-1])[keep_index_flatten]
                # 预测值的一致性损失,忽略目标缺失值的损失计算
                corr_loss[i] = self.ccc_loss_comp(x_bar.unsqueeze(-1), real_target.unsqueeze(-1))    
                last_bar = x_bar[:,-1] - x_bar[:,-2]
                last_target = real_target[:,-1] - real_target[:,-2]
                fds_loss[i] = self.mse_loss(last_bar.unsqueeze(-1), last_target.unsqueeze(-1))               
                # 遍历行业分类，对每个分类下的股票列表预测损失进行计算
                loss_total = [[] for _ in range(map_size)]
                loss_combine = None
                ins_indus_targets = last_targets[...,i]
                # for j in range(map_size):
                #     # 取得索引对照，并映射到输出值和结果集上
                #     idx_list = torch.Tensor(IndustryMappingUtil.get_sw_industry_instrument(sw_ins_mappings[j])).to(ins_indus_targets.device).long()
                #     # 还需要忽略缺失值,需要按照批次分别衡量
                #     for k in range(keep_index_bool.shape[0]):
                #         keep_index = torch.where(keep_index_bool[k])[0]
                #         idx_list_bool = torch.isin(idx_list,keep_index)
                #         idx_list = idx_list[idx_list_bool]
                #         # 有可能对应目标当天全部没有数据，则跳过
                #         if idx_list.shape[0]==0:
                #             continue
                #         sv_item = sv_instru[j][k].squeeze(-1)
                #         # 输出结果也需要同步过滤
                #         sv_item = sv_item[torch.where(idx_list_bool)[0]]
                #         ins_indus_target = ins_indus_targets[k,idx_list,0]
                #         # 计算相关性损失
                #         if idx_list.shape[0]==1:
                #             # 如果只有一个值，无法进行相关性损失计算，改为mse损失
                #             loss_item = self.mse_loss(sv_item.unsqueeze(0),ins_indus_target.unsqueeze(0))
                #         else:
                #             loss_item = self.ccc_loss_comp(sv_item.unsqueeze(0),ins_indus_target.unsqueeze(0))      
                #         loss_total[j].append(loss_item)   
                #     if len(loss_total[j])>0:
                #         loss_total[j] = torch.stack(loss_total[j])
                #         if loss_combine is None:
                #             loss_combine = loss_total[j]
                #         else:
                #             loss_combine = torch.concat([loss_combine,loss_total[j]])
                # ce_loss[i] = loss_combine.mean()
                # 行业分类总体损失，使用相关性损失
                ava_index = torch.where(torch.all(indus_targets>-1,dim=-1))[0]
                # 如果存在缺失值，则忽略，不比较
                indus_targets_real = indus_targets[ava_index]
                sv_indus = sv_indus[ava_index]
                cls_loss[i] = self.ccc_loss_comp(sv_indus,indus_targets_real)  
                
                loss_sum = loss_sum + corr_loss[i] + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]     
    
    
class IndusAloneLoss(UncertaintyLoss):
    """基于行业分类整合的损失函数，以日期维度进行整合的3D版本"""
    
    def __init__(self,indus_dim,ref_model=None,device=None):
        super(IndusAloneLoss, self).__init__(ref_model=ref_model,device=device)
        
        # 股票数量维度
        self.indus_dim = indus_dim
        self.ref_model = ref_model
        self.device = device  
        
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,mode="pretrain"):
        """Multiple Loss Combine"""

        (output,_,_) = output_ori
        (target,target_class,future_round_targets,future_sw_index_target) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 忽略目标缺失值的损失计算,找出符合比较的索引
        keep_index_bool_flatten = target_class.reshape(-1)>=0
        keep_index_flatten = torch.where(keep_index_bool_flatten)[0]
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                real_target_exi = real_target[:,0,:]
                index_target_item = future_sw_index_target[:,i]
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                x_bar,sv,sw_index_data = output_item  
                x_bar = x_bar.squeeze(-1)
                x_bar_exi = x_bar[:,0,:]
                # 只使用指数数据进行corr走势预测
                # corr_loss[i] = self.ccc_loss_comp(x_bar_exi, real_target_exi)        
                # fds_loss[i] = self.ccc_loss_comp(torch.mean(x_bar[:,1:,:],dim=1), real_target_exi) 
                x_bar_flat = x_bar.reshape(-1,x_bar.shape[-1])  
                real_target_flat = real_target.reshape(-1,real_target.shape[-1])  
                last_bar = x_bar_flat[:,-1] - x_bar_flat[:,-2]
                last_target = real_target_flat[:,-1] - real_target_flat[:,-2]
                fds_loss[i] = self.mse_loss(last_bar.unsqueeze(-1), last_target.unsqueeze(-1))                   
                corr_loss[i] += self.ccc_loss_comp(x_bar_flat, real_target_flat)           
                # 行业分类总体损失
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    round_targets_item = future_round_targets[j,keep_index,i]
                    # 总体目标值最后几位(pred_len)会是0，不进行计算
                    if torch.any(round_targets_item==0):
                        continue
                    sv_indus = sv[j,keep_index]
                    # cls_loss[i] += self.mse_loss(sv_indus,round_targets_item.unsqueeze(-1))  
                    cls_loss[i] += self.ccc_loss_comp(sv_indus.squeeze(-1),round_targets_item)  
                    
                # 指标整体数值损失,忽略缺失值
                for k in range(index_target_item.shape[0]):
                    item = index_target_item[k]
                    op_data = sw_index_data[k,0]
                    if item>0:
                        ce_loss[i] += torch.abs(op_data-item)
                if i<=1:
                    loss_sum = loss_sum + ce_loss[i] + cls_loss[i]
                elif i==2:
                    loss_sum = loss_sum + cls_loss[i] 
                elif i==3:
                    loss_sum = loss_sum + corr_loss[i]                     
                else:
                    loss_sum = loss_sum + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]     
    
        