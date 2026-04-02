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
# import torchsort

from cus_utils.common_compute import tensor_intersect,normalization_axis,pairwise_compare,normalization_standard,all_elements_same,is_same_elements
from .feature_loss import AdaptiveSingleFeatureLoss
from .triplet_loss import AdaptiveSemiHardTripletLoss,ContinuousSemiHardTripletLoss
from .triplet_miner import ContinuousTripletLossWithMemory,ContinuousTripletConfig
from .contrastive_regression_loss import TripletContrastiveRegressionLoss,ContrastiveRegressionLoss,PairwiseContrastiveRegressionLoss
from .arc_loss import RobustArcFaceRegression
from .rank_loss import attention_approx_ndcg_loss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from audioop import minmax

def gaussian_loss(z):
    mu = z.mean()
    sigma = z.std()
    return (mu **2 + (sigma - 1)**2).mean()
    
def softrank_ndcg_loss(scores, labels, k=None, sigma=0.1, gain_fn=None):
    """
    scores: (batch, n) 预测得分
    labels: (batch, n) 真实相关性（非负实数）
    k: 截断位置，若为None则使用全部
    sigma: 温度参数，控制排名的平滑程度（越小越接近硬排序）
    gain_fn: 增益函数，默认取 labels 本身；若需指数增益可用 lambda x: 2**x - 1
    """
    batch_size, n = scores.shape
    device = scores.device
    
    if gain_fn is None:
        gain_fn = lambda x: x
    gains = gain_fn(labels)                 # (batch, n)
    
    # 1. 计算两两比较概率
    # p_ij = P(i 排在 j 前面)
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)   # (batch, n, n)
    p_ij = torch.sigmoid(diff / sigma)                 # (batch, n, n)
    # 对角线为0.5，不影响后续求和
    
    # 2. 计算每个文档的期望排名（从1开始）
    # rank_i = 1 + sum_{j != i} p_{ji}
    # p_{ji} 即 j 排在 i 前面的概率 = p_ij[j, i] 的转置
    # 使用 sum_{j} p_ij[j, i] 但需要排除 i=j，但 p_ij[i,i]=0.5，故需要减去0.5
    p_ji = p_ij.transpose(1, 2)            # (batch, n, n)
    expected_ranks = 1 + p_ji.sum(dim=2)   # (batch, n)  注意包含了 i=j 时的0.5，实际正好是1+sum_{j!=i}p_ji
    # 实际上，p_ji.sum(dim=2) 包含了 j=i 时 p_ji[i,i]=0.5，所以结果就是 1 + sum_{j!=i} p_ji
    # 但为了更精确，也可以手动减去0.5，但效果差异不大
    
    # 3. 计算平滑 DCG
    # 使用折扣函数 1 / log2(1 + rank)
    disc = 1.0 / torch.log2(1.0 + expected_ranks)   # (batch, n)
    # 对于超过 k 的位置，折扣设为0（只计算前k个）
    if k is not None:
        # 这里需要知道哪些文档排在前k位，但由于排名是连续的，我们直接用 soft top-k 近似
        # 简单做法：使用 sigmoid 对排名进行截断，例如 mask = sigmoid((k+0.5 - expected_ranks) / tau)
        # 但为了简洁，这里直接对期望排名小于等于 k 的文档计算 DCG
        mask = (expected_ranks <= k).float()
        dcg = (gains * disc * mask).sum(dim=1)
    else:
        dcg = (gains * disc).sum(dim=1)
    
    # 4. 计算 IDCG
    # 按真实相关性降序排序，得到理想排名
    sorted_labels, sorted_indices = torch.sort(labels, dim=1, descending=True)
    ideal_gains = gain_fn(sorted_labels)
    ideal_ranks = torch.arange(1, n+1, device=device).float().unsqueeze(0).expand(batch_size, -1)
    ideal_disc = 1.0 / torch.log2(1.0 + ideal_ranks)
    if k is not None:
        ideal_mask = (ideal_ranks <= k).float()
        idcg = (ideal_gains * ideal_disc * ideal_mask).sum(dim=1)
    else:
        idcg = (ideal_gains * ideal_disc).sum(dim=1)
    
    # 避免除零
    idcg = torch.clamp(idcg, min=1e-8)
    ndcg = dcg / idcg
    loss = 1 - ndcg
    return loss.mean()

class FuturesIndustryLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,embedding_size=16,lock_epoch_num=0,scale_dict=None
                 ,opt_size=1,num_mixtures=5,output_chunk_length=2,cut_len=2,loss_weights=None,combine_nodes=None):
        
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
        self.combine_nodes = combine_nodes
        self.scale_dict = scale_dict
        
        # 整体及品种特征值回归损失函数
        self.index_feature_loss = AdaptiveSingleFeatureLoss(loss_type='welsch', device=self.device)
        self.ins_feature_loss = AdaptiveSingleFeatureLoss(loss_type='cauchy', device=self.device)
        self.contrast_loss = WeightedSpearmanLoss()

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

    def compute_tar_top_loss(self,pred,target,top_num=3,r=0.5):
        """使用目标视角，计算top损失"""

        # 尺度约束：[-1, 1]
        pred = torch.tanh(pred)
        # 分布匹配：标准正态
        pred = (pred - pred.mean()) / (pred.std() + 1e-8)
                
        _, top_target_index = torch.topk(target, k=top_num, dim=0)
        _, top_target_inverse_index = torch.topk(target, k=top_num, largest=False, dim=0)
        _, top_pred_index = torch.topk(pred, k=top_num, dim=0)
        _, top_pred_inverse_index = torch.topk(pred, k=top_num, largest=False, dim=0)        

        def _compute_margin_loss(p_index,t_index,mode=0):
            # 取得目标排名靠前的下标，并对照预测值进行比较
            top_pred = torch.gather(pred, 0, t_index)
            top_target = torch.gather(target, 0, t_index)
            # 计算当前对应的预测值与实际最大预测值的差距，并使用名次加权
            top_pred_real_norm = torch.gather(pred, 0, p_index)     
            top_pred_norm = torch.gather(pred, 0, t_index)          
            if mode==0:
                magin_loss = top_target.mean() - top_pred.mean()     
                pred_magin = top_pred_real_norm.mean() - top_pred_norm.mean()
            else:
                magin_loss = top_pred.mean() - top_target.mean()    
                pred_magin = top_pred_norm.mean() - top_pred_real_norm.mean()      
            magin_loss = torch.clamp(magin_loss, min=0.1)
            # 需要强制输出服从先验分布，否则会全部集中到某个数值
            pred_magin = pred_magin + gaussian_loss(pred)
            # 根据名次是否匹配再次加权
            match_num = tensor_intersect(top_pred_index,top_target_index).shape[0]
            magin_loss = r * magin_loss +  (1-r) * (top_num - match_num) * pred_magin 
            return magin_loss
                    
        magin_loss = _compute_margin_loss(top_pred_index,top_target_index,mode=0)
        magin_loss_inverse = _compute_margin_loss(top_pred_inverse_index,top_target_inverse_index,mode=1)
        loss = magin_loss + magin_loss_inverse
        
        return loss
       
    def compute_indus_top_loss(self,pred,target,sw_ins_mappings=None,ins_rel_index=None):
        """按照行业计算top损失"""
        
        indus_scale_arr = self.scale_arr['indus_scale']
        # 按照行业取内部的最大和最小
        top_real_index = []
        for i,instruments in enumerate(indus_scale_arr):
            if i==0:
                top_num = 2
            else:
                top_num = 1            
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)

        return top_loss

    def compute_nt_top_loss(self,pred,target,sw_ins_mappings=None,ins_rel_index=None):
        """按照是否包含夜盘计算top损失"""
        
        nt_scale_arr = self.scale_dict['nt_scale']
        top_real_index = []
        for i,instruments in enumerate(nt_scale_arr):
            if i==0:
                top_num = 1
            else:
                top_num = 2
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss

    def compute_combine_top_loss(self,pred,target,ins_rel_index=None):
        """按照指定规则分组计算top损失"""
        
        scale_arr = self.scale_arr[0]
        top_real_index = []
        for i,instruments in enumerate(scale_arr):
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=1)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss
    
    def compute_mr_top_loss(self,pred,target,sw_ins_mappings=None,ins_rel_index=None):
        """按照交易保证金比率计算top损失"""
        
        mr_scale_arr = self.scale_arr['mr_scale']
        top_real_index = []
        for i,instruments in enumerate(mr_scale_arr):
            if i==0:
                top_num = 1
            else:
                top_num = 2            
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss

    def compute_cy_top_loss(self,pred,target,sw_ins_mappings=None,ins_rel_index=None):
        """按照品种创建年份计算top损失"""
        
        cy_scale_arr = self.scale_arr['cy_scale']
        top_real_index = []
        for i,instruments in enumerate(cy_scale_arr):
            if i==0:
                top_num = 2
            else:
                top_num = 1
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss
      
    def compute_rank_top_loss(self,pred,target,top_num=0,sigma=0.1):
        """按照排序模式，计算top损失"""
        
        loss_long = softrank_ndcg_loss(pred, target, k=top_num, sigma=sigma, gain_fn=lambda x: 2**x - 1)
        loss_short = softrank_ndcg_loss(-pred, target, k=top_num, sigma=sigma, gain_fn=lambda x: 2**x - 1)
        loss = loss_long + loss_short
        return loss
       
    def compute_indus_loss(self,pred,target,ins_rel_index=None,sw_ins_mappings=None):
        """按照行业计算损失"""
        
        indus_data_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        loss = 0
        for index in indus_data_index:
            instruments = FuturesMappingUtil.get_instrument_rel_index_within_industry(sw_ins_mappings,index)
            instruments = torch.Tensor(instruments).to(pred.device)
            instruments = tensor_intersect(instruments,ins_rel_index).long()
            if instruments.shape[0]<2 or all_elements_same(pred[instruments]) or all_elements_same(target[instruments]):
                loss += self.mse_loss(pred.unsqueeze(0), target.unsqueeze(0))
            else:
                loss += self.ccc_loss_comp(pred[instruments], target[instruments])
        loss = loss/len(indus_data_index)
        
        return loss
    
    def compute_exchange_top_loss(self,pred,target,sw_ins_mappings=None,ins_rel_index=None):
        """按照交易所计算top损失"""
    
        exchange_ids = FuturesMappingUtil.get_exchange_ids(sw_ins_mappings)
        u_exc_ids = np.unique(exchange_ids)
        top_real_index = []
        for exchange_id in u_exc_ids:
            instruments = np.where(exchange_id==exchange_ids)[0]
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=1)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
        top_real_index = torch.Tensor(np.array(top_real_index)).to(pred.device).long() 
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss
    
    
    def _compute_top_loss(self,pred,target,top_real_index):
        if top_real_index.shape[0]<2:
            return self.mse_loss(pred.unsqueeze(0),target.unsqueeze(0))
        if all_elements_same(pred[top_real_index]) or all_elements_same(target[top_real_index]):
            top_loss = self.mse_loss(pred[top_real_index].unsqueeze(0),target[top_real_index].unsqueeze(0))
        else:
            top_loss = self.ccc_loss_comp(pred[top_real_index],target[top_real_index])  
        return top_loss    
             
    def filter_top_index_bidi(self,pred,top_num=3,ins_rel_index=None):
        pred_index = torch.argwhere(pred!=0)[:,0]
        # 拆分出排名靠前和靠后的分组
        sort_index = pred_index[torch.argsort(-pred[pred_index])]
        pred_index_long = sort_index[:top_num]
        pred_index_short = sort_index[-top_num:]
        # 当前可用品种的再次筛选
        if ins_rel_index is not None:
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
                        # 使用价格指标作为主要指标
                        price_diff_range = price_targets[j,ins_rel_index]  
                        price_diff_range_all = price_targets[j,ins_all]  
                        round_targets_item = future_round_targets[j,ins_all,target_len,0]
                        node_num = ins_all.shape[0]
                        sv_out_item_nt = sv[0]['nt_scale'][j]
                        sv_out_item_indus = sv[0]['indus_scale'][j]
                        target_item = target[j,ins_all,target_len,0]
                        # cls_loss[i] += self.compute_main_loss(attention_scores[ins_rel_index],target_item)  
                        # cls_loss[i] += self.compute_top_loss(sv_out_item[ins_rel_index],target_item[ins_rel_index])    
                        cls_loss[i] += self.compute_nt_top_loss(sv_out_item_nt,target_item,ins_rel_index=ins_rel_index,sw_ins_mappings=sw_ins_mappings)    
                        # ce_loss[i] += self.compute_cy_top_loss(sv_out_item_att,target_item,ins_rel_index=ins_rel_index,sw_ins_mappings=sw_ins_mappings)  
                        # ce_loss[i] += self.compute_combine_top_loss(sv_out_item_att,target_item,ins_rel_index=ins_rel_index)  
                        # corr_loss[i] += self.compute_indus_loss(sv_out_item_att2,target_item,ins_rel_index=ins_rel_index,sw_ins_mappings=sw_ins_mappings) 
                        # ce_loss[i] += self.compute_indus_top_loss(sv_out_item_att,target_item,ins_rel_index=ins_rel_index,sw_ins_mappings=sw_ins_mappings)  
                        ce_loss[i] += self.compute_indus_loss(sv_out_item_indus,target_item,ins_rel_index=ins_rel_index,sw_ins_mappings=sw_ins_mappings) 
                        
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
        
            
