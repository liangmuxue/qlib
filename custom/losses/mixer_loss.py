import sys
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss,WeightedSpearmanLoss,HuberLoss,WeightedHuberLoss
from tft.class_define import get_simple_class
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from darts_pro.tft_futures_dataset import get_scale_conf
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import f1_score
from darts_pro.tft_futures_dataset import concat_scale_arr,emb_scale_arr

from cus_utils.common_compute import tensor_intersect,normalization_axis,scale_value,normalization_standard,all_elements_same,torch_intersect_indices
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
                 trend_threhold=None,opt_size=1,num_mixtures=5,input_chunk_length=28,output_chunk_length=2,cut_len=2,loss_weights=None,combine_nodes=None):
        
        super(FuturesIndustryLoss, self).__init__(ref_model=ref_model,device=device)
        
        self.lock_epoch_num = lock_epoch_num
        self.ref_model = ref_model
        self.device = device  
        self.target_mode = target_mode
        
        self.output_chunk_length = output_chunk_length 
        self.input_chunk_length = input_chunk_length
        self.cut_len = cut_len
        self.log_vars = nn.Parameter(torch.zeros(len(loss_weights)))
        self.loss_weights = loss_weights
        self.num_mixtures = num_mixtures
        self.embedding_size = embedding_size
        self.opt_size = opt_size
        self.combine_nodes = combine_nodes
        self.trend_threhold = trend_threhold
        
        
        self.scale_dict = emb_scale_arr(scale_dict)
        self.scale_arr = concat_scale_arr(scale_dict)
        self.scale_dict_par = scale_dict
        
        # 整体及品种特征值回归损失函数
        self.index_feature_loss = AdaptiveSingleFeatureLoss(loss_type='welsch', device=self.device)
        self.ins_feature_loss = AdaptiveSingleFeatureLoss(loss_type='cauchy', device=self.device)
        self.contrast_loss = WeightedSpearmanLoss()
        self.criterion = WeightedHuberLoss()

    def create_avg_trend_value(self,trend_value):
        """根据预测值生成趋势值"""
        
        sort_index = np.argsort(trend_value)
        # 根据未来多段趋势，取最大看多，最小的看空 ，其他看平
        if sort_index[0]==self.cut_len-1:
            return -1
        elif sort_index[-1]==self.cut_len-1:
            return 1
        return 0

    def create_trend_value(self,trend_value,past_target_trend,top_num=2):
        """根据预测值生成趋势值,中间模式"""
        
        # 通过绝对值取得接近0的索引，以和过去数值匹配
        sort_index = np.argsort(np.abs(trend_value))
        ind_index = sort_index[:top_num]
        pred_value = past_target_trend[ind_index].mean()
        
        return pred_value
                       
    def judge_topNum_from_trend(self,trend_value,top_num=1,trend_threhold=None):
        """根据趋势数值，生成top选取数量"""

        long_top_num = top_num
        short_top_num = top_num
        threhold_min = trend_threhold['min']
        threhold_short = trend_threhold['short']
        threhold_long = trend_threhold['long']
        threhold_max = trend_threhold['max']
        
        if trend_value<threhold_min:
            long_top_num = 0
            short_top_num = top_num*2
        elif trend_value>=threhold_min and trend_value<threhold_short:
            long_top_num = top_num - 1
            short_top_num = top_num + 1            
        elif trend_value>=threhold_short and trend_value<threhold_long:
            long_top_num = top_num
            short_top_num = top_num
        elif trend_value>=threhold_long and trend_value<threhold_max:
            long_top_num = top_num + 1
            short_top_num = top_num - 1
        else:
            long_top_num = top_num*2
            short_top_num = 0               
        
        return (long_top_num,short_top_num)    
    
    def reset_device(self):
        pass
        # for key in self.scale_dict.keys():
        #     scale_arr = self.scale_dict[key]
        #     scale_arr = [torch.Tensor(ins).to(self.device).long() for ins in scale_arr]
        #     self.scale_dict[key] = scale_arr        
    
    def compute_batch_with_time_section_loss(self,pred,target,future_week_info=None,top_num=3,mid_num=3):
        """批次内按照时间分片计算top损失"""
        
        dayofweek = future_week_info[:,0]
        week = future_week_info[:,1]
        loss = 0
        cnt = 0
        if all_elements_same(target) or all_elements_same(pred):
            loss = self.mse_loss(pred.unsqueeze(0), target.unsqueeze(0))
        else:
            loss = self.compute_top_loss(pred, target,top_num=top_num,mid_num=mid_num,need_mid=True)
        cnt += 1
        
        def _compute_item_loss(item_idx):
            if item_idx.shape[0]==1:
                return torch.abs(target[item_idx] - pred[item_idx]).mean()
            target_item = target[item_idx]
            pred_item = pred[item_idx]
            target_item = normalization_standard(target_item)
            pred_item = normalization_standard(pred_item)
            if all_elements_same(target_item) or all_elements_same(pred_item):
                loss = self.mse_loss(pred_item.unsqueeze(0), target_item.unsqueeze(0))  
            else:
                loss = self.ccc_loss_comp(pred_item, target_item)      
            return loss        
        
        # 分别按照周序号，以及星期内日期序号进行损失计算
        for dayofweek_no in dayofweek.unique():
            idx = torch.where(dayofweek==dayofweek_no)[0]
            loss += _compute_item_loss(idx)
            cnt += 1
        for week_no in week.unique():
            idx = torch.where(week==week_no)[0]
            loss +=  _compute_item_loss(idx)
            cnt += 1
            
        if cnt>0:
            loss = loss/cnt
        else:
            loss = self.mse_loss(pred.unsqueeze(0), target.unsqueeze(0))

        return loss   
                            
    def compute_top_loss(self,pred,target,top_num=3,mid_num=3,need_mid=False,return_index=False):
        """计算top损失"""

        top_pred, top_pred_index = torch.topk(pred, k=top_num, dim=0)
        top_pred_inverse, top_pred_inverse_index = torch.topk(pred, k=top_num, largest=False, dim=0)
        pred_index_mid = self.filter_middle_index(pred, mid_num=mid_num)      
        pred_mid = pred[pred_index_mid]  
        top_target = torch.gather(target, 0, top_pred_index)
        top_target_inverse = torch.gather(target, 0, top_pred_inverse_index)
        top_target_mid = torch.gather(target, 0, pred_index_mid)
        if need_mid:
            total_index = torch.stack([top_pred_index,top_pred_inverse_index,top_target_mid])
            top_pred_data = torch.cat([top_pred,top_pred_inverse,pred_mid])
            top_target_data = torch.cat([top_target,top_target_inverse,top_target_mid])
        else:
            total_index = torch.stack([top_pred_index,top_pred_inverse_index])
            top_pred_data = torch.cat([top_pred,top_pred_inverse])
            top_target_data = torch.cat([top_target,top_target_inverse])            
        
        if all_elements_same(top_target_data) or all_elements_same(top_pred_data):
            top_loss = self.mse_loss(top_pred_data.unsqueeze(0), top_target_data.unsqueeze(0)) 
        else:
            top_loss = self.ccc_loss_comp(top_pred_data, top_target_data)
        
        if not return_index:
            return top_loss
        return top_loss,total_index
    
    def get_scale_key(self,scale_def):
        return scale_def['p0'] + "_" + scale_def['p1']
    
    def compute_popu_top_loss(self,pred,target,key=None,ins_rel_index=None,top_num=1):
        """通用业务分支的TOP损失"""

        scale_arr = self.scale_dict[key]
        long_index_total = torch.Tensor([]).to(pred.device) 
        short_index_total = torch.Tensor([]).to(pred.device) 
        target_norm = []
        for i,instruments in enumerate(scale_arr):
            instruments = torch.Tensor(instruments).to(pred.device).long()
            pred_index_long,pred_index_short = self.filter_top_index_bidi(pred[instruments],top_num=top_num*2)
            pred_index_long = instruments[pred_index_long]
            pred_index_short = instruments[pred_index_short]
            long_index = pred_index_long[:top_num]
            short_index = pred_index_short[:top_num]    
            long_index_total = torch.cat([long_index_total,long_index])
            short_index_total = torch.cat([short_index_total,short_index])
            target_item_norm = target[instruments] # normalization_standard(target[instruments])
            target_norm.append(target_item_norm)
        
        target_norm = torch.cat(target_norm)
        combine_index = torch.cat([long_index_total,short_index_total])   
        combine_index = tensor_intersect(combine_index,ins_rel_index).long()
        top_loss = self._compute_top_loss(pred, target, combine_index)
        
        return top_loss        

    def compute_cate_trunk_loss(self,pred_ori,target,norm_in_batch=0,detail_trunk_loss=None):
        """按照类别趋势，分片比较"""
        
        
    def compute_multi_trunk_loss(self,pred_ori,target,key=None,norm_in_batch=0,detail_trunk_loss=None):
        """按照业务属性，分片比较"""

        scale_arr = self.scale_dict[key].values()
        loss_detail = {}
        loss = 0
        ins_all = np.concatenate([item['instruments'] for item in scale_arr])
        ins_all = torch.Tensor(ins_all).to(target.device).long()
        if norm_in_batch==1:
            target_item = normalization_standard(target[ins_all])
            pred = normalization_standard(pred_ori)
        else:
            target_item = target[ins_all]
            pred = pred_ori
        
        # top_num = 2
        # # 针对总体进行损失计算
        # if all_elements_same(target_item) or all_elements_same(pred):
        #     loss += self.mse_loss(pred.unsqueeze(0), target_item.unsqueeze(0))
        # else:
        #     loss += self.compute_top_loss(pred, target_item,top_num=top_num,mid_num=top_num,need_mid=True)   
        # loss_detail[key] = loss
        
        detail_top_num = 1
        cnt = 0
        # 针对每个小分类进行损失计算
        for item in scale_arr:
            ins = torch.Tensor(item['instruments']).to(pred.device).long() 
            ins_inner,real_ins_index,_ = torch_intersect_indices(ins_all,ins)
            if ins_inner.shape[0]<2:
                loss_detail.append(torch.tensor(0).to(pred.device))
                continue     
            if norm_in_batch==2:
                pred_norm = normalization_standard(pred[real_ins_index])
                target_norm_item = normalization_standard(target_item[real_ins_index])
            else:
                pred_norm = pred[real_ins_index]
                target_norm_item = target_item[real_ins_index]                
            if all_elements_same(target_norm_item) or all_elements_same(pred_norm):
                loss_item = self.mse_loss(pred_norm.unsqueeze(0), target_norm_item.unsqueeze(0))
            else:
                loss_item = self.compute_top_loss(pred_norm, target_norm_item,top_num=detail_top_num,mid_num=detail_top_num,need_mid=True)
            loss_key = item['p0_code'] + "_" + item['p1_code']
            if loss_key in detail_trunk_loss:
                detail_trunk_loss[loss_key] = torch.cat([detail_trunk_loss[loss_key],torch.Tensor([loss_item])])
            else:
                detail_trunk_loss[loss_key] = torch.Tensor([loss_item])
            cnt += 1
            loss = loss + loss_item
        loss = loss/cnt
            
        return loss,loss_detail

    def compute_multi_trend_loss(self,pred,target,key=None,ins_inner=None,ins_rel_index=None):

        ins_inner,inner_index,_ = torch_intersect_indices(ins_inner,ins_rel_index)
        ins_inner = ins_inner.long()
        
        if ins_inner.shape[0]<2:
            loss = 0
        else:    
            loss,pred_index = self.compute_top_loss(pred[inner_index], target[ins_inner], top_num=1,return_index=True)
            
        return loss  
        

    def compute_batch_trunk_loss(self,pred,target,sec_num=4):
        """批次内按照不同编号规则，分片比较"""

        loss = 0
        count = 0
        seq = torch.arange(pred.shape[0]).to(pred.device) 
        groups = seq.chunk(sec_num)
        # 按照顺序前后分组比较
        for seq_in_group in groups:
            pred_inner = pred[seq_in_group]
            target_inner = target[seq_in_group]
            if all_elements_same(pred_inner) or all_elements_same(target_inner):
                loss += self.mse_loss(pred_inner.unsqueeze(0), target_inner.unsqueeze(0))
            else:
                loss += self.ccc_loss_comp(pred_inner, target_inner)
            count += 1
        # 跳跃式分组
        remainder = seq % sec_num  
        groups = [seq[remainder == i] for i in range(sec_num)]
        for seq_in_group in groups:
            pred_inner = pred[seq_in_group]
            target_inner = target[seq_in_group]
            if all_elements_same(pred_inner) or all_elements_same(target_inner):
                loss += self.mse_loss(pred_inner.unsqueeze(0), target_inner.unsqueeze(0))
            else:
                loss += self.ccc_loss_comp(pred_inner, target_inner)
            count += 1   
                
        if count>0:
            loss = loss/count
        
        return loss  
              
    def compute_popu_weight_loss(self,pred,target,key=None,ins_rel_index=None,top_num=2,trend_threhold=None):
        """通用业务分支的WEIGHT损失"""

        scale_arr = self.scale_dict[key]
        loss = 0
        count = 0
        min_threhold = trend_threhold['min']
        max_threhold = trend_threhold['max']       
        short_threhold = -1.0 # trend_threhold['short']   
        long_threhold = 1.0 # trend_threhold['long']          
        for i,instruments in enumerate(scale_arr):
            count += 1
            instruments = torch.Tensor(instruments).to(pred.device).long()
            instruments = tensor_intersect(instruments,ins_rel_index).long()
            if instruments.shape[0]<top_num*2:
                loss += self.mse_loss(pred[instruments].unsqueeze(0),target[instruments].unsqueeze(0))
                continue
            pred_item = pred[instruments]
            target_item = target[instruments]
            pred_item_norm = pred_item # map_to_neg1_pos1_torch(pred_item)
            target_item_norm = target_item # normalization_standard(target_item)
            pred_weights = self.get_sample_weights(pred_item_norm, short_threhold, long_threhold,min_num=top_num)
            loss += self.compute_weight_top_loss(pred_item_norm, target_item_norm,pred_weights)
        
        if count>0:
            loss = loss/count
        
        return loss      
    
    def compute_weight_top_loss(self,pred,target,weights=None,real_weight_huber=False):
        
        top_index = torch.where(weights>1)
        pred_top = pred[top_index]
        target_top = target[top_index]
        if real_weight_huber:
            loss = self.criterion(pred,target,weights)
        else:
            if all_elements_same(target_top) or all_elements_same(pred_top):
                loss = self.mse_loss(pred_top.unsqueeze(0), target_top.unsqueeze(0))
            else:
                loss = self.ccc_loss_comp(pred_top, target_top)
        return loss
                    
    def  _compute_top_loss(self,pred,target,top_real_index):
        if top_real_index.shape[0]<2:
            return self.mse_loss(pred.unsqueeze(0),target.unsqueeze(0))
        if all_elements_same(pred[top_real_index]) or all_elements_same(target[top_real_index]):
            top_loss = self.mse_loss(pred[top_real_index].unsqueeze(0),target[top_real_index].unsqueeze(0))
        else:
            top_loss = self.ccc_loss_comp(pred[top_real_index],target[top_real_index])  
        return top_loss    

    def create_combine_index(self,pred,ins_rel_index=None,top_num=1):
        
        pred_index_long,pred_index_short = self.filter_top_index_bidi(pred,top_num=top_num)
        pred_index_mid = self.filter_middle_index(pred, mid_num=1)
        combine_index = torch.cat([pred_index_long,pred_index_mid,pred_index_short])
        
        combine_index = tensor_intersect(combine_index,ins_rel_index)
        
        return combine_index

    def compute_trend_loss(self,combine_trend_output,target,ins_rel_index=None):   
        """和过去的趋势数值做差分，进行损失比较"""
        
        # node_num = past_target.shape[1]
        # global_trend_feature = torch.abs(target[ins_rel_index].mean() - past_target[ins_rel_index].mean(0))
        # global_trend_output = combine_trend_output[:node_num]
        #
        # combine_index = self.create_combine_index(global_trend_output,ins_rel_index=ins_rel_index)
        # loss = self._compute_top_loss(global_trend_output, global_trend_feature,combine_index)
        
        loss = torch.zeros(2).to(target.device)
        key = list(self.scale_dict.keys())[0]
        scale_arr = self.scale_dict[key]
        for i,instruments in enumerate(scale_arr):
            instruments = tensor_intersect(instruments,ins_rel_index)
            if instruments.shape[0]<2:
                continue
            scale_output = combine_trend_output[key][instruments]
            scale_target = target[instruments]
            # scale_target_mean = normalization_axis(scale_target_mean)
            # loss += self.compute_top_loss(scale_output,scale_target_mean,top_num=1,mid_num=1,need_mid=False)
            loss[0] += self.mse_loss(scale_output, scale_target)/len(scale_arr)
            loss[1] += self.ccc_loss_comp(scale_output.mean(0), scale_target.mean(0))/len(scale_arr)
                
        return loss
                
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
        # 通过绝对值，取得接近0的索引作为中间值索引
        sort_index = pred_index[torch.argsort(torch.abs(pred[pred_index]))]
        pred_index_middle = sort_index[:mid_num]
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
    
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,trend_threhold=None):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,future_covs,target_class,future_round_targets,index_round_targets,price_targets,future_week_info,target_info) = target_ori
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
        indus_codes_mapping = FuturesMappingUtil.get_industry_codes_combine_index(sw_ins_mappings)
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        # 总体指标序号
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        main_index_abs = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        ins_index_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        predictions = None
        loop_size = self.opt_size
        target_len = -self.output_chunk_length + self.cut_len - 1
        detail_loss = None
        ins_in_scale = self.scale_dict_par['instruments'].values
        ins_in_scale = [torch.Tensor(item).to(target.device).long() for item in ins_in_scale]
        
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
                detail_trunk_loss = {}
                price_diff_rate = torch.zeros(target_class.shape[0]).to(target_class.device)
                trend_output = {}
                trend_target = {}
                ins_output_in_batch = torch.zeros([target_class.shape[0],ins_all.shape[0]]).to(target_class.device)
                ins_target_in_batch = torch.zeros([target_class.shape[0],ins_all.shape[0]]).to(target_class.device)
                for key in self.scale_dict:
                    trend_output[key] = {}
                    trend_target[key] = {}
                    for inner_key in self.scale_dict[key]:
                        item = self.scale_dict[key][inner_key]
                        trend_output[key][inner_key] = torch.zeros([target_class.shape[0]]).to(target_class.device)
                        trend_target[key][inner_key] = torch.zeros([target_class.shape[0]]).to(target_class.device)
                for j in range(target_class.shape[0]):
                    target_info_item = target_info[j][main_index_abs]
                    date = target_info_item['future_start_datetime']
                    # 如果存在缺失值，则忽略，不比较sw_index_data
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                    target_item = target[j,ins_all,target_len,0]
                    ins_diff = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info[j])])
                    ins_diff = torch.Tensor(ins_diff).to(target_class.device)                    
                    # ins_output_in_batch[j,ins_rel_index] = sw_index_data[0][j,ins_rel_index]
                    # ins_target_in_batch[j,ins_rel_index] = ins_diff[ins_rel_index]
                    if ins_rel_index.shape[0]<2:
                        continue
                    target_info_total.append(target_info[j])             
                    detail_trunk_loss_item = []
                    price_diff_range = price_targets[j,ins_rel_index]  
                    # 不同模式的损失计算                          
                    if target_mode in [2]:
                        # 按照类别趋势进行横向比较，兼顾大类和小类
                        pred_data = sw_index_data[0][j]
                        loss_item = 0
                        cnt = 0
                        p0_cate_target = torch.zeros(pred_data.shape[0]).to(target_class.device)
                        data_idx = 0
                        for scale_item in self.scale_arr:
                            p0 = scale_item['p0']
                            ins_arr = torch.Tensor(scale_item['instruments']).to(target_class.device).long()
                            ins_arr = tensor_intersect(ins_arr, ins_rel_index)
                            # 比较小类
                            ins_diff_item = ins_diff[ins_arr]
                            p1_cate_pred = []
                            p1_cate_target = []
                            for key in self.scale_dict[p0].keys():
                                p1_item = self.scale_dict[p0][key]
                                ins_p1 = torch.Tensor(p1_item['instruments']).to(target_class.device).long()
                                ins_p1 = tensor_intersect(ins_p1, ins_rel_index)
                                ins_diff_item = ins_diff[ins_p1]
                                if ins_p1.shape[0]==0:
                                    data_idx += 1  
                                    continue
                                p0_cate_target[data_idx] = ins_diff_item.mean()                      
                                p1_cate_target.append(ins_diff_item.mean())
                                p1_cate_pred.append(pred_data[data_idx])
                                data_idx += 1  
                                
                            # if len(p1_cate_target)>1:
                            #     p1_cate_target = torch.stack(p1_cate_target)    
                            #     p1_cate_pred = torch.stack(p1_cate_pred) 
                            #     if all_elements_same(p1_cate_target) or all_elements_same(p1_cate_pred):
                            #         loss_item += self.mse_lossreturn_index(p1_cate_pred.unsqueeze(0),p1_cate_target.unsqueeze(0))
                            #     else:
                            #         loss_item += self.ccc_loss_comp(p1_cate_pred,p1_cate_target)
                            #     cnt += 1                               
                        # 比较大类
                        if all_elements_same(p0_cate_target) or all_elements_same(pred_data):
                            loss_item += self.mse_loss(normalization_standard(pred_data).unsqueeze(0),normalization_standard(p0_cate_target).unsqueeze(0))
                        else:
                            loss_item += self.compute_top_loss(normalization_standard(pred_data),normalization_standard(p0_cate_target), top_num=1, mid_num=1, need_mid=True)
                        cnt += 1
                            
                        if cnt>0:
                            loss_item = loss_item/cnt
                            cls_loss[i] += loss_item
                            batch_size += 1                        
                    if target_mode in [5]:
                        # 根据网络输出，生成针对性业务分支输出,并依次计算损失
                        scale_output = sv[0]
                        # 业务分支top损失计算             
                        loss_item = 0
                        cnt = 0
                        for key in trend_output.keys():
                            ins_arr = self.scale_dict[key]
                            sv_out_item = scale_output[key][j]
                            loss,loss_detail = self.compute_multi_trunk_loss(sv_out_item,ins_diff,key=key,norm_in_batch=2,detail_trunk_loss=detail_trunk_loss)
                            loss_item += loss
                            cnt += 1
                        loss_item = loss_item/cnt
                        cls_loss[i] += loss_item
                        batch_size += 1                                   
                    elif target_mode==3:
                        for key in trend_output.keys():
                            ins_arr = self.scale_dict[key]
                            for k,inner_key in enumerate(ins_arr.keys()):
                                trend_output[key][inner_key][j] = sw_index_logits[0][key][inner_key][j]
                                ins_inner = torch.Tensor(ins_arr[inner_key]['instruments']).to(target_class.device).long()
                                ins_inner = tensor_intersect(ins_inner, ins_rel_index)
                                price_diff_range = price_targets[j,ins_inner]  
                                if ins_inner.shape[0]==0:
                                    continue
                                diff_rate = torch.sum(price_diff_range>0)/ins_inner.shape[0]
                                trend_target[key][inner_key][j] = diff_rate                        
                        batch_size += 1
                    elif target_mode==0:
                        for key in trend_output.keys():
                            ins_arr = self.scale_dict[key]
                            for k,inner_key in enumerate(ins_arr.keys()):
                                trend_output[key][inner_key][j] = sw_index_logits[0][key][inner_key][j]
                                inner_ins = ins_arr[inner_key]['instruments']
                                inner_ins = np.intersect1d(inner_ins, ins_rel_index.cpu().numpy())
                                if inner_ins.shape[0]<2:
                                    continue
                                ins_diff = np.array([t['open_diff'] for t in np.array(target_info[j])[inner_ins]])
                                target_mean = ins_diff.mean()
                                trend_target[key][inner_key][j] = target_mean               
                        batch_size += 1
                                        
                if target_mode in [1]:
                    ins_out = ins_output_in_batch # normalization_standard(ins_output_in_batch,dim=0)
                    ins_target = ins_target_in_batch # normalization_standard(ins_target_in_batch,dim=0)
                    cnt = 0
                    for k in range(ins_output_in_batch.shape[1]):
                        if (torch.sum(ins_target[:,k]==0)>ins_target.shape[0]-12) or (torch.sum(ins_out[:,k]==0)>ins_target.shape[0]-12):
                            continue
                        cls_loss[i] += self.compute_batch_with_time_section_loss(ins_out[:,k], ins_target[:,k],future_week_info,top_num=3,mid_num=3)
                        cnt += 1
                    if cnt>0:
                        cls_loss[i] = cls_loss[i]/cnt
                    else:
                        cls_loss[i] = self.mse_loss(ins_out,ins_target)          
                if target_mode in [2]:  
                    cls_loss[i] = cls_loss[i]/batch_size  
                    ce_loss[i] = ce_loss[i]/batch_size  
                if target_mode in [5]:  
                    cls_loss[i] = cls_loss[i]/batch_size  
                    # ce_loss[i] = ce_loss[i]/batch_size  
                    # detail_trunk_loss = torch.stack(detail_trunk_loss).to(target_class.device)
                    detail_loss = {}
                    for key in detail_trunk_loss.keys():
                        detail_loss[key] = detail_trunk_loss[key].mean()
                if target_mode in [0,3]:
                    loss = 0
                    cnt = 0
                    detail_loss = {}
                    for key in trend_output:
                        target_all = []
                        trend_all = []
                        for inner_key in trend_output[key]:
                            target_all.append(trend_target[key][inner_key])
                            trend_all.append(trend_output[key][inner_key])
                            target_norm = trend_target[key][inner_key] # normalization_standard(trend_target[key][inner_key])
                            trend_norm = trend_output[key][inner_key] # normalization_standard(trend_output[key][inner_key])
                            d_loss = self.compute_batch_with_time_section_loss(trend_norm, target_norm,future_week_info,top_num=3,mid_num=3)
                            log_key = key + "_" + inner_key
                            detail_loss[log_key] = d_loss
                            loss += d_loss
                            cnt += 1
                    cls_loss[i] = loss/cnt
                        
                                                
                loss_sum = loss_sum + cls_loss[i] + ce_loss[i] + fds_loss[i] + corr_loss[i]
                           
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss,detail_loss]    
    
    
    def get_sample_weights(self,y_true,min_threshold=-0.5,max_threshold=0.5, out_weight=10.0,min_num=2):
        
        weights = torch.ones_like(y_true)
        weights[y_true > max_threshold] = out_weight 
        weights[y_true < min_threshold] = out_weight 
        
        if torch.sum(weights>1)<min_num:
            _, top_index = torch.topk(y_true, k=min_num, dim=0)
            _, top_inverse_index = torch.topk(y_true, k=min_num, largest=False, dim=0)
            weights[top_index] = out_weight 
            weights[top_inverse_index] = out_weight 
        
        return weights        
    
    def build_batch_trend_data(self,sw_data,batch_size=32,mode=0):
        """生成批次内的趋势数据"""
        
        trend_output = {}
        for j in range(batch_size):
            self.build_scale_trend_output(sw_data,j,trend_output=trend_output,mode=mode)   
        # # 批次内归一化
        # for key in trend_output.keys():
        #     trend_output[key] = normalization_axis(trend_output[key], axis=0)
             
        return trend_output   
        
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
    
            
