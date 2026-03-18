import sys
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss,WeightedSpearmanLoss,weighted_spearman_loss
from cus_utils.common_compute import batch_cov,batch_cov_comp,eps_rebuild,normalization
from tft.class_define import get_simple_class
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from cus_utils.common_compute import tensor_intersect,normalization_axis,pairwise_compare,normalization_standard,all_elements_same,is_same_elements
from .feature_loss import AdaptiveSingleFeatureLoss
from .triplet_loss import AdaptiveSemiHardTripletLoss,ContinuousSemiHardTripletLoss
from .triplet_miner import ContinuousTripletLossWithMemory,ContinuousTripletConfig
from .contrastive_regression_loss import TripletContrastiveRegressionLoss,ContrastiveRegressionLoss,PairwiseContrastiveRegressionLoss
from .arc_loss import RobustArcFaceRegression
from .rank_loss import attention_approx_ndcg_loss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from audioop import minmax

def attention_concentration_loss(attn_weights, top_k, loss_type="entropy"):
    """
    注意力集中度损失：约束权重向Top-K集中
    :param attn_weights: 注意力权重 (batch, n_head, seq_len, seq_len)
    :param top_k: 目标Top-K数量
    :param loss_type: 损失类型：
        - "entropy": 熵损失（熵越小，分布越集中）
        - "topk_ratio": Top-K占比损失（最大化Top-K权重占比）
    :return: 标量损失值
    """
    # 维度调整：合并batch和head维度，方便计算 (batch*n_head, seq_len, seq_len)
    attn_flat = attn_weights.reshape(-1, attn_weights.size(-2), attn_weights.size(-1))
    seq_len = attn_flat.size(-1)
    
    if loss_type == "entropy":
        """方案1：熵损失（推荐）
        熵公式：H = -Σ(p_i * log(p_i + ε))
        熵越小，分布越集中，因此损失目标是最小化熵
        """
        eps = 1e-8  # 避免log(0)
        entropy = -torch.sum(attn_flat * torch.log(attn_flat + eps), dim=-1)  # (batch*n_head, seq_len)
        # 对所有位置的熵取平均，作为最终损失
        loss = torch.mean(entropy)
        
    elif loss_type == "topk_ratio":
        """方案2：Top-K占比损失
        计算每个位置Top-K权重的总和，目标是最大化这个总和（即最小化 1 - 总和）
        """
        # 取每个位置的Top-K权重 (batch*n_head, seq_len, top_k)
        top_k_vals, _ = torch.topk(attn_flat, k=top_k, dim=-1)
        # 计算Top-K权重占比 (batch*n_head, seq_len)
        top_k_ratio = torch.sum(top_k_vals, dim=-1)
        # 损失 = 1 - 占比 的平均值（占比越接近1，损失越小）
        loss = torch.mean(1 - top_k_ratio)
    
    return loss

class FuturesIndustryLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,embedding_size=16,lock_epoch_num=0
                 ,opt_size=1,num_mixtures=5,output_chunk_length=2,cut_len=2,loss_weights=None):
        
        super(FuturesIndustryLoss, self).__init__(ref_model=ref_model,device=device)
        
        self.lock_epoch_num = lock_epoch_num
        self.ref_model = ref_model
        self.device = device  
        self.target_mode = target_mode
        
        self.output_chunk_length = output_chunk_length 
        self.cut_len = cut_len
        self.log_vars = nn.Parameter(torch.zeros(len(loss_weights)))
        self.loss_weights = loss_weights
        self.num_mixtures = num_mixtures
        self.embedding_size = embedding_size
        self.opt_size = opt_size
        
        # 整体及品种特征值回归损失函数
        self.index_feature_loss = AdaptiveSingleFeatureLoss(loss_type='welsch', device=self.device)
        self.ins_feature_loss = AdaptiveSingleFeatureLoss(loss_type='cauchy', device=self.device)
        
        # 基于有监督回归的比对损失函数，用于衡量整体指数
        # self.contrast_loss = SupervisedContrastiveRegressionLoss(contrast_weight=0.5,base_loss='mse',temperature=0.5,device=self.device)
        
        # self.contrast_loss = TripletContrastiveRegressionLoss(distance_func=self.ccc_distance,margin=1.0,device=self.device)
        # self.contrast_loss = ContinuousSemiHardTripletLoss(pairwise_distance=self.ccc_distance,device=self.device)
        self.contrast_loss = WeightedSpearmanLoss()
        # config = ContinuousTripletConfig(
        #     memory_size=1024,
        #     embedding_dim=embedding_size,
        #     base_margin=1.0,
        #     similarity_threshold=0.5,
        #     mining_strategy='semi-hard',
        #     semi_hard_ratio_target=0.3,
        #     device=device
        # )
        # self.contrast_loss = ContinuousTripletLossWithMemory(config)
        # self.contrast_loss = PairwiseContrastiveRegressionLoss(device=self.device)
        # self.rank_loss = GlobalLambdaRankLoss(reduction='none')

    def compute_main_loss(self,pred,target):
        """计算主要损失"""
        
        # 主体损失，斯皮尔逊相关性
        if all_elements_same(target) or all_elements_same(pred,eps=1e-6):
            main_loss = 0
            if all_elements_same(pred):
                print("all_elements_same for pred:{}".format(pred))
        else:
            main_loss = self.ccc_loss_comp(pred, target)
        return main_loss
        
    def compute_top_loss(self,pred,target,top_num=3,no_real_dis=True):
        """计算top损失"""
        
        top_pred, top_pred_index = torch.topk(pred, k=top_num, dim=0)
        top_pred_inverse, top_pred_inverse_index = torch.topk(pred, k=top_num, largest=False, dim=0)
        top_target = torch.gather(target, 0, top_pred_index)
        top_target_inverse = torch.gather(target, 0, top_pred_inverse_index)
        top_pred_data = torch.cat([top_pred,top_pred_inverse])
        top_target_data = torch.cat([top_target,top_target_inverse])
        
        if all_elements_same(top_target_data) or all_elements_same(top_pred_data):
            top_loss = self.mse_loss(top_pred_data.unsqueeze(0), top_target_data.unsqueeze(0)) 
        else:
            top_loss = self.ccc_loss_comp(top_pred_data, top_target_data)
        # 优化：对前k名候选的目标值进行比对，实现额外加权（强制区分核心候选）
        top_real,target_topk_index = torch.topk(target, k=top_num, dim=0)
        top_real_inverse,target_topk_insverse_index = torch.topk(target, k=top_num,largest=False, dim=0)
        top_real_index = torch.cat([target_topk_index,target_topk_insverse_index])
        top_pred_ref_data = torch.gather(pred, 0, top_real_index)
        top_real_target_data = torch.cat([top_real,top_real_inverse])
        # 使用当前索引对应的实际数据与实际排名靠前的数据作差，作为加权的权重
        if all_elements_same(top_real_target_data) or no_real_dis:
            real_dis_weights = 0
        else:
            real_dis_weights = self.ccc_loss_comp(top_target_data,top_real_target_data)
        top_loss = top_loss + real_dis_weights
        if not no_real_dis:
            top_loss = top_loss/2
        return top_loss
       
    def filter_top_index_bidi(self,pred,top_num=3,ins_rel_index=None):
        pred_index = torch.argwhere(pred!=0)[:,0]
        # 拆分出排名靠前和靠后的分组
        sort_index = pred_index[torch.argsort(-pred[pred_index])]
        pred_index_long = sort_index[:top_num]
        pred_index_short = sort_index[-top_num:]
        # 当前可用品种的再次筛选
        pred_index_long = tensor_intersect(pred_index_long,ins_rel_index)
        pred_index_short = tensor_intersect(pred_index_short,ins_rel_index)
        
        return pred_index_long,pred_index_short

    def filter_top_index(self,pred,top_num=3,ins_rel_index=None,mask_mode=False):
        # 数组中已经把非目标置零，因此通过下标还原实际目标
        if mask_mode:
            pred_index = torch.argwhere(pred!=0)[:,0]
        else:
            pred_index = torch.argsort(-pred)[:top_num]
        # 当前可用品种的再次筛选
        pred_index = tensor_intersect(pred_index,ins_rel_index)
        
        return pred_index

    def compute_att_main_loss(self,pred,target,top_num=3,att_long=None,att_short=None):
        """配合注意力机制，计算损失"""
        
        # 主体损失，斯皮尔逊相关性
        if all_elements_same(target) or all_elements_same(pred,eps=1e-6):
            main_loss = 0
            if all_elements_same(pred):
                print("all_elements_same for pred:{}".format(pred))
        else:
            # 使用带权重的mse或ccc损失
            # long_loss = self.weighted_mse_loss(pred,target,att_long,reduction='sum')
            # short_loss = self.weighted_mse_loss(pred,target,att_short,reduction='sum')
            # main_loss = (long_loss + short_loss)/2
            main_loss = weighted_spearman_loss(pred.unsqueeze(0), target.unsqueeze(0),att_long.unsqueeze(0))     
            main_loss = main_loss + weighted_spearman_loss(pred.unsqueeze(0), target.unsqueeze(0),att_short.unsqueeze(0))    
            main_loss = main_loss/2
        return main_loss

    def compute_att_top_loss(self,pred,target,topk_mask_weights=None,ins_rel_index=None,top_num=3):
        """配合注意力机制，计算TOP损失"""
        
        # 分别选取注意力权重多方和空方的前几个
        pred_index_long = torch.argwhere(topk_mask_weights==1)[:,0]
        pred_index_long = tensor_intersect(pred_index_long,ins_rel_index)
        pred_index_short = torch.argwhere(topk_mask_weights==-1)[:,0]
        pred_index_short = tensor_intersect(pred_index_short,ins_rel_index)
        if pred_index_long.shape[0]==0 or pred_index_short.shape[0]==0:
            return self.mse_loss(pred.unsqueeze(0),target.unsqueeze(0))
        top_pred = pred[pred_index_long]
        top_pred_inverse = pred[pred_index_short]        
        top_pred_data = torch.cat([top_pred,top_pred_inverse])
        top_target = torch.gather(target, 0, pred_index_long)
        top_target_inverse = torch.gather(target, 0, pred_index_short)
        top_target_data = torch.cat([top_target,top_target_inverse])
        
        if all_elements_same(top_target_data) or all_elements_same(top_pred_data):
            top_loss = self.mse_loss(top_pred_data.unsqueeze(0), top_target_data.unsqueeze(0)) 
        elif pred_index_long.shape[0]!=top_num or pred_index_short.shape[0]!=top_num:
            top_loss = self.mse_loss(top_pred_data.unsqueeze(0), top_target_data.unsqueeze(0)) 
        else:
            top_loss = self.ccc_loss_comp(top_pred_data, top_target_data)
            # 叠加目标数值的差距最大化损失
            # ts_index = top_pred_data.argsort(descending=True)
            # top_target_resort = top_target_data[ts_index]
            # margin = (top_target_resort[:top_num] - top_target_resort[top_num:]).mean()
            # weights = (1 - torch.sigmoid(margin)) * 2
            # top_loss = top_loss * weights            
        return top_loss
       


    def dual_highk_loss(self,attention_weights, topk_indices, bottomk_indices, k):
        """
        双高权重损失函数：让前k和后k位置的权重都尽可能高
        :param attention_weights: 原始注意力权重，[batch_size, seq_len]
        :param topk_indices: 排序前k的索引，[batch_size, k]
        :param bottomk_indices: 排序后k的索引，[batch_size, k]
        :param k: 前k/后k的k值
        :return: 总损失
        """
        batch_size, seq_len = attention_weights.shape
        
        # 1. 构建"前k+后k"的掩码：这2k个位置为1，其余为0
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        # 初始化掩码
        dual_k_mask = torch.zeros_like(attention_weights)
        # 标记前k位置
        dual_k_mask[batch_indices, topk_indices] = 1.0
        # 标记后k位置
        dual_k_mask[batch_indices, bottomk_indices] = 1.0
        
        # 2. 约束：前k+后k的权重之和尽可能接近1（核心目标）
        dual_k_sum = (attention_weights * dual_k_mask).sum(dim=-1)
        loss_dual_high = F.mse_loss(dual_k_sum, torch.ones_like(dual_k_sum))
        
        # 3. 约束：前k+后k内部权重尽可能均匀（避免某一部分独占权重）
        # 提取2k个位置的权重
        batch_indices_combine = torch.cat([batch_indices,batch_indices],-1)
        dual_k_weights = attention_weights[batch_indices_combine, torch.cat([topk_indices, bottomk_indices], dim=1)]
        # 计算2k个权重的方差（方差越小，分布越均匀）
        dual_k_var = torch.var(dual_k_weights, dim=-1)
        loss_dual_balance = dual_k_var.mean()  # 方差越小越好
        
        # 4. 惩罚中间区域（非前k且非后k）的权重，让其尽可能小
        middle_mask = 1 - dual_k_mask
        middle_sum = (attention_weights * middle_mask).sum(dim=-1)
        loss_middle_suppress = F.mse_loss(middle_sum, torch.zeros_like(middle_sum))
        
        # 总损失：三个部分加权（可根据需求微调，核心保证前k+后k高权重）
        total_loss = loss_dual_high + 0.1 * loss_dual_balance + loss_middle_suppress
        
        return total_loss
                   
            
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,epoch_num=0):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,future_covs,target_class,future_round_targets,index_round_targets,price_targets,long_diff_seq_targets,target_info) = target_ori
        future_index_round_target = index_round_targets[:,:,-self.output_chunk_length:,:]
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.zeros([len(output)]).to(self.device)
        fds_loss = torch.zeros([len(output)]).to(self.device)
        ce_loss = torch.zeros([len(output)]).to(self.device)
        loss_sum = torch.tensor(0.0).to(self.device) 
        
        # 取得所有品种排序号
        ins_in_indus_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        # 行业分类排序号
        indus_data_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        # 总体指标序号
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        main_index_abs = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        ins_index_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        predictions = None
        loop_size = self.opt_size
        
        for i in range(loop_size):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                dec_out,sv,sw_index_data = output_item  
                future_round_targets_factor = future_round_targets[...,i]
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                price_index_total = []
                sw_index_total = []
                index_target_total = []
                future_covs_main_total = []
                target_info_total = []
                sv_out_total = []
                target_total = []
                batch_size = 0
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    index_target_item = future_index_round_target[j,indus_rel_index,:,i]
                    indus_index = tensor_intersect(keep_index,indus_data_index).to(keep_index.device)
                    inner_class_item = target_class_item[indus_data_index]                            
                    sv_out_item_batch = sv[0][j]
                    ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                    if ins_rel_index.shape[0]<3:
                        continue
                    round_targets_item = future_round_targets_factor[j,ins_rel_index,self.cut_len-1]  
                    # 样本太少则忽略
                    if round_targets_item.shape[0]<=3:
                        continue      
                    target_info_total.append(target_info[j])             
                    future_covs_ins = future_covs[i][j,ins_rel_index,:,:]
                    future_covs_main = future_covs[i][j,main_index_abs,-1,:]
                    # 记录主指数的多个指标特征，后续计算对比损失
                    future_covs_main_total.append(future_covs_main)   
                    price_diff_range_real = price_targets[j][ins_all]
                    sv_out_item_real = sv[0][j]
                    # price_diff_range = target_info_inbatch[main_index_abs]['diff_range'][-self.output_chunk_length:]
                    
                    # 不同模式的损失计算                          
                    if target_mode==0:
                        if indus_index.shape[0]<2:
                            continue
                        # 板块整体损失计算
                        ce_loss[i] += self.ccc_loss_comp(sw_index_data[j,ins_rel_index],index_target_item[:,-1])/10
                        # if target_mode==1: 
                        #     ce_loss[i] += self.mse_loss(sw_index_data[j].unsqueeze(-1),time_diff_targets.unsqueeze(-1))  
                    elif target_mode==2:
                        # 比较全部品种，辅助整体指数比较
                        # target_len = -1
                        target_len = -self.output_chunk_length + self.cut_len - 1
                        # 借用其他目标作为整体走势衡量
                        ref_indicator = 0 
                        ref_indicator2 = 2                          
                        dec_out_item = dec_out[j,:,:,:][ins_rel_index].squeeze(-1)
                        # dec_out_item = dec_out[j,ins_rel_index,:,0]
                        # sv_out_item_att = sv[1][j][ins_rel_index]
                        # sv_out_item_att2 = sv[2][j][ins_rel_index]
                        # sv_out_item_att2 = sv[2][j][ins_rel_index]
                        # dec_out_item_att = dec_out[j,:,1][ins_rel_index]
                        # 使用价格指标作为主要指标
                        price_diff_range = price_targets[j,ins_rel_index]  
                        round_targets_att = future_round_targets[j,ins_rel_index,:,:]
                        round_targets_item = future_round_targets[j,ins_rel_index,target_len,ref_indicator]
                        # round_targets_item_att = future_round_targets[j,ins_rel_index,target_len,1]
                        # round_targets_item_att2 = future_round_targets[j,ins_rel_index,target_len,2]
                        # round_targets_item_att2 = future_round_targets[j,ins_rel_index,:,ref_indicator2]
                        # if not is_same_elements(att_tar,price_diff_range,eps=1e-3):
                        #     print("not same")
                        # cls_loss[i] += self.compute_top_loss(sv_out_item, price_diff_range, no_real_dis=False)   
                        node_num = ins_all.shape[0]
                        sv_out_item = sv_out_item_real[:node_num]
                        topk_mask_weights = sv_out_item_real[node_num:2*node_num]
                        sv_out_item_att = sv_out_item_real[2*node_num:3*node_num][ins_rel_index]
                        attention_weights = sv_out_item_real[3*node_num:4*node_num]
                        cls_loss[i] += self.compute_main_loss(sv_out_item[ins_rel_index],price_diff_range)  
                        ce_loss[i] += self.compute_top_loss(sv_out_item[ins_rel_index],price_diff_range)  
                        fds_loss[i] += self.compute_att_top_loss(sv_out_item,price_diff_range_real,ins_rel_index=ins_rel_index,
                                        top_num=top_num,topk_mask_weights=topk_mask_weights) 
                        # 计算双高权重损失函数：让前k和后k位置的权重都尽可能高
                        topk_indices = torch.argwhere(topk_mask_weights==1)[:,0]
                        bottomk_indices = torch.argwhere(topk_mask_weights==-1)[:,0]
                        corr_loss[i] += self.dual_highk_loss(attention_weights.unsqueeze(0), topk_indices.unsqueeze(0), bottomk_indices.unsqueeze(0), top_num)
                        # 计算top损失
                        # ce_loss[i] += self.compute_top_loss(sv_out_item, price_diff_range, top_num=top_num,no_real_dis=True)
                        target_item = target[j,ins_rel_index,:,ref_indicator2]
                        # 辅助目标的损失
                        # fds_loss[i] += self.ccc_loss_comp(dec_out_item,target_item)    
                        # corr_loss[i] += self.compute_main_loss(sv_out_item_att,price_diff_range)    
                        batch_size += 1                    
                    elif target_mode==3:
                        ref_indicator = 1 
                        # 整体损失计算
                        target_item = target[j,ins_rel_index,:,ref_indicator]
                        cls_loss[i] += self.ccc_loss_comp(sw_index_data[j],target_item.mean(dim=0))
                        batch_size += 1
                        
                if target_mode in [0]:
                    loss_sum = loss_sum + ce_loss[i]           
                if target_mode in [2,3]:
                    cls_loss[i] = cls_loss[i]/batch_size
                    ce_loss[i] = ce_loss[i]/batch_size
                    fds_loss[i] = fds_loss[i]/batch_size
                    corr_loss[i] = corr_loss[i]/batch_size
                    loss_sum = loss_sum + cls_loss[i] + ce_loss[i] + fds_loss[i] + corr_loss[i]
                    
                if target_mode in [6]:
                    # 衡量目标值与前面各段已知结果比较的相对位置，作为优化目标，整体指标模式
                    diff_target = long_diff_seq_targets[:,main_index,-1,i]
                    ce_loss[i] = torch.abs(sw_index_data.squeeze()-diff_target).mean()        
                    loss_sum = loss_sum + ce_loss[i]      
                           
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss,predictions]    




    ############################### Data Compute Relate ####################################

    def compute_diff_range_class(self,target_info,target_info_arr=None,is_main=False):
        """根据实际涨跌数据计算类别"""
        
        # 对于整体指标，不能使用开盘和收盘价格直接计算，使用原数据（所有品种收盘价差的均值,之前的dataset中已经设置好了）
        if is_main:
            # 使用所有品种的均值进行计算
            diff_range_total = np.array([(pr['open_array'][-self.output_chunk_length+self.cut_len-1] - pr['open_array'][-self.output_chunk_length])
                                          /pr['open_array'][-self.output_chunk_length]*100 for pr in target_info_arr])
            diff_range = diff_range_total.mean()
            open_array = np.stack([pr["open_array"] for pr in target_info_arr])
            diff_range_arr = np.array((open_array[:,1:] - open_array[:,:-1])/open_array[:,:-1]*100)
            diff_range_arr = np.mean(diff_range_arr,0)
        else:
            open_array = target_info["open_array"]
            # price_array = target_info["price_array"] 
            # 收盘与前收盘价差作为衡量指标
            # diff_range = (price_array[-self.output_chunk_length+self.cut_len-1] - price_array[-self.output_chunk_length])/price_array[-self.output_chunk_length]*100
            # 预测l结束日期的开盘与预测开始日期的开盘价差作为衡量指标
            diff_range = (open_array[-self.output_chunk_length+self.cut_len-1] - open_array[-self.output_chunk_length])/open_array[-self.output_chunk_length]*100
            # 价差展示，从过去一直延续到预测当日，未包含最后一条记录
            diff_range_arr = np.array((open_array[1:] - open_array[:-1])/open_array[:-1]*100)            
        range_class = get_simple_class(diff_range)
        
        return diff_range,range_class,diff_range_arr
    
    def compute_batch_last_distance(self,features,targets):
        """计算批次内最后一条数据与前面数据的特征距离和实际目标距离的匹配度"""
        
        fea_distance_vec = pairwise_compare(features[-1:],features[:-1],self.ccc_distance).squeeze(0)
        tar_distance_vec = torch.abs(targets[-1] - targets[:-1]) 
        distance = self.ccc_loss_comp(fea_distance_vec, tar_distance_vec)
        return distance
        
            
