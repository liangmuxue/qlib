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
    
class FuturesIndustryLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,embedding_size=16,lock_epoch_num=0,scale_dict=None,
                 trend_threhold=None,opt_size=1,num_mixtures=5,output_chunk_length=2,cut_len=2,loss_weights=None,combine_nodes=None):
        
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
        self.trend_threhold = trend_threhold
        
        self.scale_dict = {}
        for key in scale_dict.keys():
            scale_arr = scale_dict[key]
            scale_arr = [torch.Tensor(ins).to(device).long() for ins in scale_arr]
            self.scale_dict[key] = scale_arr
        
        # 整体及品种特征值回归损失函数
        self.index_feature_loss = AdaptiveSingleFeatureLoss(loss_type='welsch', device=self.device)
        self.ins_feature_loss = AdaptiveSingleFeatureLoss(loss_type='cauchy', device=self.device)
        self.contrast_loss = WeightedSpearmanLoss()
    
    def judge_topNum_from_trend(self,trend_value,top_num=1):
        """根据趋势数值，生成top选取数量"""
        
        long_top_num = top_num
        short_top_num = top_num
        if trend_value<self.trend_threhold['min']:
            long_top_num = 0
            short_top_num = top_num*2
        elif trend_value>=self.trend_threhold['min'] and trend_value<self.trend_threhold['short']:
            long_top_num = top_num - 1
            short_top_num = top_num + 1            
        elif trend_value>=self.trend_threhold['short'] and trend_value<self.trend_threhold['long']:
            long_top_num = top_num
            short_top_num = top_num
        elif trend_value>=self.trend_threhold['long'] and trend_value<self.trend_threhold['max']:
            long_top_num = top_num + 1
            short_top_num = top_num - 1
        else:
            long_top_num = top_num*2
            short_top_num = 0               
        
        return (long_top_num,short_top_num)    
    
    def reset_device(self):
        for key in self.scale_dict.keys():
            scale_arr = self.scale_dict[key]
            scale_arr = [torch.Tensor(ins).to(self.device).long() for ins in scale_arr]
            self.scale_dict[key] = scale_arr        
    
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
        
        indus_scale_arr = self.scale_dict['indus_scale']
        # 按照行业取内部的最大和最小
        top_real_index = []
        for i,instruments in enumerate(indus_scale_arr):
            if i==0:
                top_num = 2
            else:
                top_num = 1            
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            mid_index = self.filter_middle_index(pred[instruments], mid_num=top_num, ins_rel_index=ins_rel_index)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
            top_real_index.append(instruments[mid_index])
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
            mid_index = self.filter_middle_index(pred[instruments], mid_num=top_num, ins_rel_index=ins_rel_index)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
            top_real_index.append(instruments[mid_index])
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
        
        mr_scale_arr = self.scale_dict['mr_scale']
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
        
        cy_scale_arr = self.scale_dict['cy_scale']
        top_real_index = []
        for i,instruments in enumerate(cy_scale_arr):
            if i==0:
                top_num = 2
            else:
                top_num = 1
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num)
            mid_index = self.filter_middle_index(pred[instruments], mid_num=top_num, ins_rel_index=ins_rel_index)
            top_real_index.append(instruments[pred_index_long])
            top_real_index.append(instruments[pred_index_short])
            top_real_index.append(instruments[mid_index])
        top_real_index = torch.cat(top_real_index)
        top_real_index = tensor_intersect(top_real_index,ins_rel_index)
        top_loss = self._compute_top_loss(pred, target, top_real_index)
        
        return top_loss

    def compute_popu_top_loss(self,pred,target,key=None,ins_rel_index=None,top_num=1,trend_ref=None):
        """通用业务分支的TOP损失"""

        scale_arr = self.scale_dict[key]
        long_index_total = torch.Tensor([]).to(pred.device) 
        short_index_total = torch.Tensor([]).to(pred.device) 
        for i,instruments in enumerate(scale_arr):
            instruments = torch.Tensor(instruments).to(pred.device).long()
            # 进行多种多空组合选取，并根据趋势置信度实现权重损失比较
            trend_value = trend_ref[i]
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num*2)
            pred_index_long = instruments[pred_index_long]
            pred_index_short = instruments[pred_index_short]
            long_top_num,short_top_num = self.judge_topNum_from_trend(trend_value, top_num)
            if long_top_num>0:
                long_index = pred_index_long[:long_top_num]
            else:
                long_index = torch.Tensor([]).to(pred.device)
            if short_top_num>0:
                short_index = pred_index_short[:short_top_num]    
            else:
                short_index = torch.Tensor([]).to(pred.device)      
            long_index_total = torch.cat([long_index_total,long_index])
            short_index_total = torch.cat([short_index_total,short_index])
        
        combine_index = torch.cat([long_index_total,short_index_total])   
        combine_index = tensor_intersect(combine_index,ins_rel_index).long()
        top_loss = self._compute_top_loss(pred, target, combine_index)
        
        return top_loss        
        
               
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

    def filter_middle_index(self,pred,mid_num=2,ins_rel_index=None):
        pred_index = torch.argwhere(pred!=0)[:,0]
        sort_index = pred_index[torch.argsort(-pred[pred_index])]
        mid_pos = pred.shape[0]//2
        pred_index_middle = sort_index[mid_pos:mid_pos+mid_num]
        # 当前可用品种的再次筛选
        if ins_rel_index is not None:
            pred_index_middle = tensor_intersect(pred_index_middle,ins_rel_index)
            
        return pred_index_middle
    
    def filter_top_index(self,pred,top_num=3,ins_rel_index=None,mode='long'):
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
    
    def build_scale_output(self,sv,batch_no):
        
        scale_output = {}
        for key in self.scale_dict.keys():
            sv_out_item = sv[key][batch_no]
            scale_output[key] = sv_out_item
        return scale_output
    
    def build_scale_trend_output(self,sw_index,batch_no,trend_output=None):

        # if "global_trend_feature" not in trend_output:
        #     trend_output["global_trend_feature"] = sw_index["global_trend_feature"][batch_no]
        # else:
        #     trend_output["global_trend_feature"] = torch.cat([trend_output["global_trend_feature"],sw_index["global_trend_feature"][batch_no]])
        for key in self.scale_dict.keys():
            index_data = sw_index[key][batch_no].unsqueeze(0)
            if key not in trend_output:
                trend_output[key] = index_data
            else:
                trend_output[key] = torch.cat([trend_output[key],index_data],dim=0)
        return trend_output        

    def build_scale_trend_target(self,target_info,ins_rel_index=None,trend_target=None):

        for key in self.scale_dict.keys():
            scale_arr = self.scale_dict[key]
            scale_target = []
            scale_value = torch.Tensor(target_info['scale_arr'][key]).to(ins_rel_index.device)
            for i,instruments in enumerate(scale_arr):
                instruments = tensor_intersect(instruments,ins_rel_index)
                if instruments.shape[0]==0:
                    scale_target.append(torch.tensor(0).to(ins_rel_index.device))
                else:
                    if scale_value.shape[0]<i+1:
                        scale_target_mean = torch.tensor(0).to(ins_rel_index.device)
                    else:
                        scale_target_mean = scale_value[i]
                    scale_target.append(scale_target_mean)                    
            scale_target = torch.stack(scale_target).unsqueeze(0)
            if key not in trend_target:
                trend_target[key] = scale_target
            else:
                trend_target[key] = torch.cat([trend_target[key],scale_target],dim=0)
        return trend_target        
           
    
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
        target_len = -self.output_chunk_length + self.cut_len - 1
        
        for i in range(loop_size):
            target_mode = self.target_mode[i]
            if optimizers_idx==i or optimizers_idx==-1:
                output_item = output[i] 
                # 输出值分别为未来目标走势预测、分类目标幅度预测、行业分类总体幅度预测
                sw_index_logits,sv,sw_index_data = output_item  
                future_round_targets_factor = future_round_targets[...,i]
                # 分批次，按照不同分类，分别衡量类内期货品种总体损失
                target_info_total = []
                batch_size = 0
                trend_output = {}
                trend_target = {}
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    index_target_item = future_index_round_target[j,indus_rel_index,:,i]
                    indus_index = tensor_intersect(keep_index,indus_data_index).to(keep_index.device)
                    inner_class_item = target_class_item[indus_data_index]                            
                    ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                    target_item = target[j,ins_all,target_len,0]
                    target_info_item = target_info[j][main_index_abs]
                    # 收集趋势业务数据
                    self.build_scale_trend_output(sw_index_data[0],j,trend_output=trend_output)
                    self.build_scale_trend_target(target_info_item,trend_target=trend_target,ins_rel_index=ins_rel_index)   
                                     
                    if ins_rel_index.shape[0]<3:
                        continue
                    round_targets_item = future_round_targets_factor[j,ins_rel_index,self.cut_len-1]  
                    # 样本太少则忽略
                    if round_targets_item.shape[0]<=3:
                        continue      
                    target_info_total.append(target_info[j])             
                    
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
                        ref_indicator = 0 
                        # 使用价格指标作为主要指标
                        price_diff_range = price_targets[j,ins_rel_index]  
                        price_diff_range_all = price_targets[j,ins_all]  
                        round_targets_item = future_round_targets[j,ins_all,target_len,0]
                        node_num = ins_all.shape[0]
                        sv_out_item = sv[0]['global_feature'][j]
                        # 根据网络输出，生成针对性业务分支输出,并依次计算损失
                        scale_output = self.build_scale_output(sv[0],j)
                        # 业务分支top损失计算                      
                        # for sidx, key in enumerate(scale_output.keys()):
                        #     sv_out_item = scale_output[key]
                        #     # 参考趋势输出，作为top选取参数
                        #     if sidx==0:
                        #         cls_loss[i] += self.compute_popu_top_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index,trend_ref=trend_output[key][j])
                        #     if sidx==1:
                        #         ce_loss[i] += self.compute_popu_top_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index,trend_ref=trend_output[key][j])   
                        # batch_size += 1                    
                    elif target_mode==3:
                        ref_indicator = 1 
                        # 整体损失计算
                        # target_item = target[j,ins_rel_index,:,ref_indicator]
                        # cls_loss[i] += self.ccc_loss_comp(sw_index_data[j],target_item.mean(dim=0))
                        # batch_size += 1
                        
                if target_mode in [0]:
                    loss_sum = loss_sum + ce_loss[i]           
                if target_mode in [2,3]:
                    # cls_loss[i] = cls_loss[i]/batch_size
                    # ce_loss[i] = ce_loss[i]/batch_size
                    # 批次内计算各个业务分支的整体损失
                    inner_batch = 0
                    scale_out_total = []
                    scale_target_total = []
                    for key in trend_output.keys():
                        scale_output = trend_output[key]
                        scale_target = trend_target[key]
                        if len(scale_output.shape)==1:
                            scale_out_total.append(scale_output.unsqueeze(-1))
                            scale_target_total.append(scale_target.unsqueeze(-1))
                        else:
                            scale_out_total.append(scale_output)
                            scale_target_total.append(scale_target)
                    scale_out_total = torch.cat(scale_out_total,-1).transpose(1,0)
                    scale_target_total = torch.cat(scale_target_total,-1).transpose(1,0)
                    # scale_out_norm = normalization_axis(scale_out_total,axis=-1)
                    # scale_target_norm = normalization_axis(scale_target_total,axis=-1)
                    cls_loss[i] += self.ccc_loss_comp(scale_out_total, scale_target_total)
                    
                    # corr_loss[i] = corr_loss[i]/batch_size
                    loss_sum = loss_sum + cls_loss[i] + ce_loss[i] + fds_loss[i] + corr_loss[i]
                           
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
    
            
