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

from cus_utils.common_compute import tensor_intersect,normalization_axis,batch_normalization

from pytorch_metric_learning import distances, losses, miners, reducers, testers

class FuturesCombineLoss(UncertaintyLoss):
    """基于期货品种和行业分类整合的损失函数，并以日期维度进行整合"""
    
    def __init__(self,indus_dim,ref_model=None,device=None,target_mode=None):
        super(FuturesCombineLoss, self).__init__(ref_model=ref_model,device=device)
        
        # 股票数量维度
        self.indus_dim = indus_dim
        self.ref_model = ref_model
        self.device = device  
        self.target_mode = target_mode
        
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0):
        """Multiple Loss Combine"""

        (output,_,_) = output_ori
        (target,target_class,future_round_targets,last_targets,target_info) = target_ori
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
        indus_data_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        
        for i in range(len(output)):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                real_target = target[...,i]
                real_target_exi = real_target[:,0,:]
                index_target_item = future_round_targets[:,indus_data_index,i]
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                x_bar,sv,sw_index_data = output_item  
                x_bar = x_bar.squeeze(-1)
                # corr走势预测
                x_bar_flat = x_bar.reshape(-1,x_bar.shape[-1])  
                real_target_flat = real_target.reshape(-1,real_target.shape[-1])  
                # corr_loss[i] += self.ccc_loss_comp(x_bar_flat, real_target_flat)      
                # corr_loss[i] += self.mse_loss(x_bar_flat, real_target_flat)     
                corr_loss[i] += self.cos_loss(x_bar_flat,real_target_flat).mean()
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                counter = 0
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    # 只比较期货品种，不比较分类
                    keep_index = tensor_intersect(keep_index,torch.Tensor(instrument_index).to(keep_index.device))
                    round_targets_item = future_round_targets[j,keep_index,i]
                    last_target_item = last_targets[j,keep_index,i]
                    # 总体目标值最后几位(pred_len)会是0，不进行计算
                    if torch.any(round_targets_item==0):
                        continue
                    if round_targets_item.shape[0]<=1:
                        continue                    
                    sv_indus = sv[j,keep_index]
                    # 根据标志，决定比较整体涨跌幅，还是最后一段涨跌幅
                    if target_mode==0:
                        cls_loss[i] += 10 * self.mse_loss(sv_indus,round_targets_item.unsqueeze(-1))  
                    else:
                        cls_loss[i] += 10 * self.mse_loss(sv_indus,last_target_item.unsqueeze(-1))  
                    # cls_loss[i] += self.cos_loss(sv_indus.transpose(1,0),round_targets_item.unsqueeze(0))[0] 
                    # cls_loss[i] += self.ccc_loss_comp(sv_indus.squeeze(-1),round_targets_item)  
                    counter += 1
                cls_loss[i] = cls_loss[i]/counter
                
                loss_sum = loss_sum + cls_loss[i]
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]     
    

class FuturesStrategyLoss(FuturesCombineLoss):
    """基于策略选取的损失"""

    def __init__(self,indus_dim,ref_model=None,device=None,target_mode=None,lock_epoch_num=0):
        
        super().__init__(indus_dim,ref_model=ref_model,target_mode=target_mode,device=device)
        self.lock_epoch_num = lock_epoch_num
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,epoch_num=0):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,target_class,future_round_targets,index_round_target,price_targets) = target_ori
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        fds_loss = torch.tensor(0.0).to(self.device)
        ce_loss = torch.Tensor(np.array([0 for _ in range(len(output))])).to(self.device)
        # 指标分类
        loss_sum = torch.tensor(0.0).to(self.device) 
        # 取得所有品种排序号
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        indus_data_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
        
        for i in range(len(output)):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                x_bar,sv,sw_index_data = output_item  
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                counter = 0
                sv_mean = []
                round_targets_indus = []
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    # 只比较期货品种，不比较分类
                    keep_index = tensor_intersect(keep_index,torch.Tensor(instrument_index).to(keep_index.device))
                    round_targets_item = future_round_targets[j,keep_index,i]
                    last_target_item = index_round_target[j,i]
                    # 总体目标值最后几位(pred_len)会是0，不进行计算
                    if torch.any(round_targets_item==0):
                        continue
                    if round_targets_item.shape[0]<=1:
                        continue                    
                    sv_indus = sv[j,keep_index]
                    # sv_mean.append(sv_indus.mean())
                    sv_mean.append(sw_index_data[j,0])
                    round_targets_indus.append(last_target_item)
                    # round_targets_indus.append(round_targets_item.mean())
                    if target_mode==0 or target_mode==1:
                        cls_loss[i] += self.ccc_loss_comp(sv_indus.squeeze(-1),round_targets_item)     
                        counter += 1
                    if target_mode==3:
                        cls_loss[i] += self.mse_loss(sv_indus,round_targets_item.unsqueeze(-1))     
                        counter += 1                        
                if target_mode!=2:
                    cls_loss[i] = cls_loss[i]/counter
                    loss_sum = loss_sum + cls_loss[i]
                if target_mode>0: 
                    # 复用last_targets字段，作为单独品种归一化的总体数值，进行总体损失判断
                    round_targets_indus = torch.stack(round_targets_indus)
                    sw_indus = torch.stack(sv_mean)
                    if target_mode==1 or target_mode==3:
                        cl = self.mse_loss(sw_indus.unsqueeze(-1),round_targets_indus.unsqueeze(-1))  
                    else:
                        cl = self.ccc_loss_comp(sw_indus,round_targets_indus)  
                    ce_loss[i] += cl
                    loss_sum = loss_sum + ce_loss[i]
                    
        if epoch_num>=self.lock_epoch_num:
            # 综合策略损失评判
            if optimizers_idx==(len(output)) or optimizers_idx==-1:
                target = price_targets[:,instrument_index,:] 
                round_target = target[...,-1] - target[...,0]
                # 网络输出值包括预测数值，以及对应的多空索引
                choice,trend_value,combine_index = vr_class
                l_index = torch.where(trend_value)[0]
                s_index = torch.where(~trend_value)[0]
                choice_index = torch.zeros([choice.shape[0],top_num]).long().to(choice.device)
                choice_gather = torch.zeros([choice.shape[0],top_num]).to(choice.device)
                # 二次筛选更靠前的数据进行loss比对
                l_sort_idx = choice[l_index].argsort(descending=True,dim=1)[:,:top_num]
                s_sort_idx = choice[s_index].argsort(descending=False,dim=1)[:,:top_num]
                choice_index[l_index] = torch.gather(combine_index,1,l_sort_idx)
                choice_index[s_index] = torch.gather(combine_index,1,s_sort_idx)
                target_gather = torch.gather(round_target, 1, choice_index)  
                choice_gather[l_index] = torch.gather(choice, 1, l_sort_idx) 
                choice_gather[s_index] = torch.gather(choice, 1, s_sort_idx) 
                # 分别取得空头输出和多头输出,比较空头目标和多头目标
                fds_loss += 10*nn.MSELoss(reduction="mean")(choice_gather,target_gather)
                loss_sum = loss_sum + fds_loss
            
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]       

class ListwiseRegressionLoss(nn.Module):
    def __init__(self, tau=1.0, reduction='mean'):
        """
        Plackett-Luce排序损失
        :param tau: 温度系数，控制排序置信度（默认1.0）
        :param reduction: 损失聚合方式（'mean'或'sum'）
        """
        super().__init__()
        self.tau = tau
        self.reduction = reduction

    def forward(self, y_pred, y_true, group_ids=None):
        """
        :param y_pred: 模型预测值，形状为 [batch_size]
        :param y_true: 真实标签值，形状为 [batch_size]
        :param group_ids: 组标识（如查询ID），形状为 [batch_size]
        :return: 排序损失值
        """
        loss = 0.0
        unique_groups = torch.unique(group_ids)
        
        for group in unique_groups:
            # 提取当前组的预测和标签
            mask = (group_ids == group)
            s_group = y_pred[mask]  # 预测得分
            y_group = y_true[mask]  # 真实标签
            
            # 按真实标签降序排列的索引
            _, sorted_indices = torch.sort(y_group, descending=True)
            s_sorted = s_group[sorted_indices]
            
            # 计算Plackett-Luce损失（向量化实现）
            n = len(s_sorted)
            if n == 0:
                continue
                
            # 构造下三角掩码矩阵 [n, n]
            mask = torch.tril(torch.ones(n, n, device=y_pred.device)).bool()
            
            # 扩展得分矩阵 [n, n]
            s_expanded = s_sorted.unsqueeze(1).expand(-1, n)
            
            # 计算logits并应用温度系数 [n, n]
            logits = s_expanded / self.tau
            
            # 仅保留当前步骤之后的候选
            logits = logits.masked_fill(~mask, -1e9)
            
            # 计算log概率：log(exp(s_k) / sum_{j>=k} exp(s_j))
            log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            
            # 仅取对角线元素（每一步选择的正确项的log概率）
            group_loss = -log_probs.diagonal().sum()
            
            loss += group_loss
        
        if self.reduction == 'mean':
            return loss / len(unique_groups)
        else:
            return loss

def listMLE(y_pred, y_true, eps=1e-5, padded_value_indicator=-1):
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

class FuturesIndustryLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,lock_epoch_num=0,tau=0.1,output_chunk_length=2,cut_len=2):
        
        super(FuturesIndustryLoss, self).__init__(ref_model=ref_model,device=device)
        
        self.lock_epoch_num = lock_epoch_num
        self.ref_model = ref_model
        self.device = device  
        self.target_mode = target_mode
        
        # 排序损失部分，用于整体走势衡量
        self.listwise = ListwiseRegressionLoss(tau=tau)
        self.output_chunk_length = output_chunk_length 
        self.cut_len = cut_len
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,epoch_num=0):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,target_class,future_round_targets,index_round_targets,long_diff_seq_targets) = target_ori
        future_index_round_target = index_round_targets[:,:,-self.output_chunk_length:,:]
        diff_index_round_target = index_round_targets[:,:,-1:,:].repeat(1,1,self.cut_len,1) - index_round_targets[:,:,-self.cut_len-self.output_chunk_length:-self.output_chunk_length,:]
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
        ins_index_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        
        for i in range(len(output)):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                target_item = target[:,indus_data_index,:,i]
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                dec_out,sv,sw_index_data = output_item  
                future_round_targets_factor = future_round_targets[...,i]
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                counter = 0
                for j in range(target_class.shape[0]):
                    time_diff_targets = long_diff_seq_targets[j,indus_data_index,:,i]
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    index_target_item = future_index_round_target[j,indus_rel_index,:,i]
                    indus_index = tensor_intersect(keep_index,indus_data_index).to(keep_index.device)
                    inner_class_item = target_class_item[indus_data_index]                            
                    # 对应预测数据中的有效索引
                    inner_index = torch.where(inner_class_item>=0)[0]                    
                    if target_mode==0:
                        if indus_index.shape[0]<2:
                            continue
                        # 板块整体损失计算
                        ce_loss[i] += self.ccc_loss_comp(sw_index_data[j],index_target_item[:,-1])/10  
                        # if target_mode==1: 
                        #     ce_loss[i] += self.mse_loss(sw_index_data[j].unsqueeze(-1),time_diff_targets.unsqueeze(-1))  
                    elif target_mode==3:
                        # 比较全部品种
                        sv_out_item = sv[0][j]
                        ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                        if ins_rel_index.shape[0]<3:
                            continue
                        sv_out_item = sv_out_item[ins_rel_index]
                        ins_all_inner = torch.Tensor(ins_all).to(sv_out_item.device).long()
                        ins_class_item = target_class_item[ins_all_inner]
                        inner_index = ins_all_inner[torch.where(ins_class_item>=0)[0]]
                        round_targets_item = future_round_targets_factor[j,inner_index]    
                        cls_loss[i] += self.ccc_loss_comp(sv_out_item,round_targets_item)           
                    elif target_mode==2:
                        # 在行业内进行品种比较    
                        inner_class_item = target_class_item[ins_all]
                        # 对应预测数据中的有效索引
                        inner_index = torch.where(inner_class_item>=0)[0]
                        class_item = (target_class_item[inner_index]>-1)+0
                        # sv_indus = sv[0][j]
                        # sv_indus = sv_indus[inner_index]
                        # 使用第一类输出数据，进行行业内品种比较
                        for k,ind_idx in enumerate(indus_rel_index):
                            sv_out_item = sv[k][j].squeeze(-1)
                            # 对应目标数据中的有效索引
                            ins_index = ins_in_indus_index[k]
                            ins_rel_index = torch.where(target_class_item[ins_index]>=0)[0].long()
                            # 对应预测数据中的有效索引
                            ins_index = tensor_intersect(keep_index,ins_index)
                            round_targets_item = future_round_targets_factor[j,ins_index]  
                            # 总体目标值最后几位(pred_len)会是0，不进行计算
                            if round_targets_item.shape[0]<=3 or torch.any(round_targets_item==0) or torch.sum(round_targets_item<1e-4)>2 or torch.sum(round_targets_item>=0.9999)>=5:
                                continue
                            cls_loss[i] += self.ccc_loss_comp(sv_out_item[ins_rel_index],round_targets_item)    
                if target_mode in [0]:
                    loss_sum = loss_sum + ce_loss[i]
                if target_mode==1: 
                    # 衡量目标与前面各段已知结果的差值，作为优化目标
                    ce_loss[i] = self.mse_loss(sw_index_data,long_diff_seq_targets[:,indus_data_index,:,i])        
                    loss_sum = loss_sum + ce_loss[i]             
                if target_mode in [2]:
                    cls_loss[i] = cls_loss[i]/10
                    loss_sum = loss_sum + cls_loss[i]               
                if target_mode in [3]:
                    loss_sum = loss_sum + cls_loss[i]      
                if target_mode in [5]:
                    # 衡量目标值与前面各段已知结果比较的相对位置，作为优化目标
                    diff_target = long_diff_seq_targets[:,indus_data_index,-1,i]
                    ce_loss[i] = self.mse_loss(sw_index_data,diff_target)        
                    loss_sum = loss_sum + ce_loss[i]   
                                                                 
        # if epoch_num>=self.lock_epoch_num:
        #     # 综合策略损失评判
        #     if optimizers_idx==(len(output)) or optimizers_idx==-1:
        #         target = price_targets[:,instrument_index,:] 
        #         round_target = target[...,-1] - target[...,0]
        #         # 网络输出值包括预测数值，以及对应的多空索引
        #         choice,trend_value,combine_index = vr_class
        #         l_index = torch.where(trend_value)[0]
        #         s_index = torch.where(~trend_value)[0]
        #         choice_index = torch.zeros([choice.shape[0],top_num]).long().to(choice.device)
        #         choice_gather = torch.zeros([choice.shape[0],top_num]).to(choice.device)
        #         # 二次筛选更靠前的数据进行loss比对
        #         l_sort_idx = choice[l_index].argsort(descending=True,dim=1)[:,:top_num]
        #         s_sort_idx = choice[s_index].argsort(descending=False,dim=1)[:,:top_num]
        #         choice_index[l_index] = torch.gather(combine_index,1,l_sort_idx)
        #         choice_index[s_index] = torch.gather(combine_index,1,s_sort_idx)
        #         target_gather = torch.gather(round_target, 1, choice_index)  
        #         choice_gather[l_index] = torch.gather(choice, 1, l_sort_idx) 
        #         choice_gather[s_index] = torch.gather(choice, 1, s_sort_idx) 
        #         # 分别取得空头输出和多头输出,比较空头目标和多头目标
        #         fds_loss += 10*nn.MSELoss(reduction="mean")(choice_gather,target_gather)
        #         loss_sum = loss_sum + fds_loss
            
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]    


class FuturesIndustryDRollLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,lock_epoch_num=0,tau=0.1,lambda_reg=0):
        
        super(FuturesIndustryDRollLoss, self).__init__(ref_model=ref_model,device=device)
        
        self.lock_epoch_num = lock_epoch_num
        self.ref_model = ref_model
        self.device = device  
        self.target_mode = target_mode
        
        # 排序损失部分，用于整体走势衡量
        self.listwise = ListwiseRegressionLoss(tau=tau)
        self.lambda_reg = lambda_reg 
        
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,epoch_num=0):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,target_class,combine_index_round_target) = target_ori
        
        corr_loss = torch.Tensor(np.array([0 for i in range(len(output))])).to(self.device)
        cls_loss = torch.zeros([len(output)]).to(self.device)
        fds_loss = torch.zeros([len(output)]).to(self.device)
        ce_loss = torch.zeros([len(output)]).to(self.device)
        loss_sum = torch.tensor(0.0).to(self.device) 
        
        industry_index = [i for i in range(target_class.shape[-1])]
        industry_rel_index = [i for i in range(target_class.shape[-1])]
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        industry_rel_index.remove(main_index)
        index_round_target = combine_index_round_target[:,:,main_index,:]
        
        for i in range(len(output)):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                _,sv,sw_index_data = output_item  
                round_targets_item = index_round_target[...,i]
                target_class_single = target_class[:,:,main_index]
                round_targets_last = []
                sw_index_last = []
                for j in range(target_class_single.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class_single[j]
                    keep_index = torch.where(target_class_item>=0)[0]   
                    if keep_index.shape[0]<3:
                        continue
                    round_targets_inner = round_targets_item[j,keep_index]     
                    round_targets_last.append(round_targets_inner[-1])
                    # 分别衡量每个行业的时间段差分数值
                    idx = 0
                    indus_sv = []
                    indus_target = []
                    for indus_index in industry_index:
                        if indus_index==main_index:
                            continue
                        target_class_indus = target_class[j,:,indus_index]
                        keep_index = torch.where(target_class_indus>=0)[0]
                        if keep_index.shape[0]<3:
                            continue                          
                        sv_item = sv[j,:,idx,i]
                        idx += 1
                        sv_item = sv_item[keep_index]
                        if keep_index[-1]==(combine_index_round_target.shape[1]-1):
                            indus_sv.append(sv_item[-1])    
                            indus_target.append(combine_index_round_target[j,-1,indus_index,i])                     
                        indus_round_target_item = combine_index_round_target[j,keep_index,indus_index,i]
                        cls_loss[i] += self.ccc_loss_comp(sv_item,indus_round_target_item)
                cls_loss[i] = cls_loss[i]/target_class_single.shape[0]
                # 行业类别的横向比较
                indus_target = combine_index_round_target[:,-1,industry_rel_index,i]
                # indus_target = normalization(indus_target,mode="torch",axis=1)
                ce_loss[i] = self.mse_loss(sw_index_data,indus_target)
                if target_mode==0:
                    loss_sum = loss_sum + cls_loss[i] # + ce_loss[i]
            
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss]          
            