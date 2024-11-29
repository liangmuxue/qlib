import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import batch_cov,batch_cov_comp,eps_rebuild,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
import geomloss

from cus_utils.common_compute import tensor_intersect

from pytorch_metric_learning import distances, losses, miners, reducers, testers

class FuturesCombineLoss(UncertaintyLoss):
    """基于期货品种和行业分类整合的损失函数，并以日期维度进行整合"""
    
    def __init__(self,indus_dim,ref_model=None,device=None):
        super(FuturesCombineLoss, self).__init__(ref_model=ref_model,device=device)
        
        # 股票数量维度
        self.indus_dim = indus_dim
        self.ref_model = ref_model
        self.device = device  
        
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0):
        """Multiple Loss Combine"""

        (output,_,_) = output_ori
        (target,target_class,future_round_targets,target_info) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        ce_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 忽略目标缺失值的损失计算,找出符合比较的索引
        keep_index_bool_flatten = target_class.reshape(-1)>=0
        keep_index_flatten = torch.where(keep_index_bool_flatten)[0]
        # 取得所有品种排序号
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        indus_data_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
        
        for i in range(len(output)):
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                real_target_exi = real_target[:,0,:]
                index_target_item = future_round_targets[:,indus_data_index,i]
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                x_bar,sv,sw_index_data = output_item  
                x_bar = x_bar.squeeze(-1)
                x_bar_exi = x_bar[:,indus_data_index,:]
                # corr走势预测
                x_bar_flat = x_bar.reshape(-1,x_bar.shape[-1])  
                real_target_flat = real_target.reshape(-1,real_target.shape[-1])  
                corr_loss[i] += self.ccc_loss_comp(x_bar_flat, real_target_flat)           
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    # 只比较期货品种，不比较分类
                    keep_index = tensor_intersect(keep_index,torch.Tensor(instrument_index).to(keep_index.device))
                    round_targets_item = future_round_targets[j,keep_index,i]
                    # 总体目标值最后几位(pred_len)会是0，不进行计算
                    if torch.any(round_targets_item==0):
                        continue
                    if round_targets_item.shape[0]<=1:
                        continue                    
                    sv_indus = sv[j,keep_index]
                    # cls_loss[i] += self.mse_loss(sv_indus,round_targets_item.unsqueeze(-1))  
                    cls_loss_item = self.ccc_loss_comp(sv_indus.squeeze(-1),round_targets_item)  
                    cls_loss[i] += cls_loss_item
                    
                # 板块分类指标整体数值损失,板块间使用相关系数损失
                for k in range(index_target_item.shape[0]):
                    item = index_target_item[k]
                    keep_idx = torch.where(item!=0)[0]
                    if keep_idx.shape[0]<=1:
                        continue
                    op_data = sw_index_data[k]
                    if i!=2 and False:
                        ce_loss_item = self.ccc_loss_comp(item[keep_idx],op_data[keep_idx])
                    else:
                        ce_loss_item = self.mse_loss(item[keep_idx].unsqueeze(-1),op_data[keep_idx].unsqueeze(-1))
                    ce_loss[i] += ce_loss_item
                if i<=3:
                    loss_sum = loss_sum + ce_loss[i] + cls_loss[i]
                # elif i>=4:
                #     loss_sum = loss_sum + corr_loss[i]                     
                else:
                    loss_sum = loss_sum + ce_loss[i] + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]     
    
        