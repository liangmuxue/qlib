import sys
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss,WeightedSpearmanLoss,HuberLoss,WeightedHuberLoss
from tft.class_define import get_simple_class
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import f1_score
# import torchsort

from cus_utils.common_compute import tensor_intersect,normalization_axis,scale_value,normalization_standard,all_elements_same,map_to_neg1_pos1_torch
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
        
        self.scale_dict = {}
        for key in scale_dict.keys():
            scale_arr = scale_dict[key]
            scale_arr = [torch.Tensor(ins).to(device).long() for ins in scale_arr]
            self.scale_dict[key] = scale_arr
        
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
        
    def compute_top_loss(self,pred,target,top_num=3,mid_num=3,need_mid=False):
        """计算top损失"""

        top_pred, top_pred_index = torch.topk(pred, k=top_num, dim=0)
        top_pred_inverse, top_pred_inverse_index = torch.topk(pred, k=top_num, largest=False, dim=0)
        pred_index_mid = self.filter_middle_index(pred, mid_num=mid_num)      
        pred_mid = pred[pred_index_mid]  
        top_target = torch.gather(target, 0, top_pred_index)
        top_target_inverse = torch.gather(target, 0, top_pred_inverse_index)
        top_target_mid = torch.gather(target, 0, pred_index_mid)
        if need_mid:
            top_pred_data = torch.cat([top_pred,top_pred_inverse,pred_mid])
            top_target_data = torch.cat([top_target,top_target_inverse,top_target_mid])
        else:
            top_pred_data = torch.cat([top_pred,top_pred_inverse])
            top_target_data = torch.cat([top_target,top_target_inverse])            
        
        if all_elements_same(top_target_data) or all_elements_same(top_pred_data):
            top_loss = self.mse_loss(top_pred_data.unsqueeze(0), top_target_data.unsqueeze(0)) 
        else:
            top_loss = self.ccc_loss_comp(top_pred_data, top_target_data)
        
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
            target_item_norm = normalization_standard(target[instruments])
            target_norm.append(target_item_norm)
        
        target_norm = torch.cat(target_norm)
        combine_index = torch.cat([long_index_total,short_index_total])   
        combine_index = tensor_intersect(combine_index,ins_rel_index).long()
        top_loss = self._compute_top_loss(pred, target_norm, combine_index)
        
        return top_loss        
        
    def compute_popu_weight_loss(self,pred,target,key=None,ins_rel_index=None,top_num=1,trend_threhold=None):
        """通用业务分支的WEIGHT损失"""

        scale_arr = self.scale_dict[key]
        loss = 0
        count = 0
        min_threhold = trend_threhold['min']
        max_threhold = trend_threhold['max']       
        short_threhold = -1.0 # trend_threhold['short']   
        long_threhold = 1.0 # trend_threhold['long']          
        for i,instruments in enumerate(scale_arr):
            instruments = torch.Tensor(instruments).to(pred.device).long()
            instruments = tensor_intersect(instruments,ins_rel_index).long()
            if instruments.shape[0]<3:
                continue
            pred_item = pred[instruments]
            target_item = target[instruments]
            pred_item_norm = pred_item # map_to_neg1_pos1_torch(pred_item)
            target_item_norm = normalization_standard(target_item)
            pred_weights = self.get_sample_weights(pred_item_norm, short_threhold, long_threhold,min_num=2)
            loss += self.compute_weight_top_loss(pred_item_norm, target_item_norm,pred_weights)
            count += 1
        
        if count>0:
            loss = loss/count
        
        return loss      
    
    def compute_weight_top_loss(self,pred,target,weights=None):
        
        top_index = torch.where(weights>1)
        pred_top = pred[top_index]
        target_top = target[top_index]
        # loss = self.criterion(pred,target,weights)
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
    
    def build_scale_output(self,sv,batch_no):
        
        scale_output = {}
        for key in self.scale_dict.keys():
            sv_out_item = sv[key][batch_no]
            scale_output[key] = sv_out_item
        return scale_output
    
    def build_scale_data_output(self,sv_data,batch_no,scale_data_output=None):  
        
        for key in self.scale_dict.keys():
            index_data = sv_data[key][batch_no]
            index_data = index_data.unsqueeze(0)
            if key not in scale_data_output:
                scale_data_output[key] = index_data
            else:
                scale_data_output[key] = torch.cat([scale_data_output[key],index_data],dim=0)   
                        
    def build_scale_data_target(self,target,scale_data_target=None,ins_rel_index=None):
        
        for key in self.scale_dict.keys():
            scale_arr = self.scale_dict[key]
            for i,instruments in enumerate(scale_arr):
                target_item = torch.zeros([target.shape[0]]).to(ins_rel_index.device)
                instruments = tensor_intersect(instruments,ins_rel_index)
                target_item[instruments] = target[instruments]
                target_item = target_item.unsqueeze(-1)
                if key not in scale_data_target:
                    scale_data_target[key] = target_item
                else:
                    scale_data_target[key] = torch.cat([scale_data_target[key],target_item],dim=-1)
     
    def build_scale_trend_output(self,sw_index,batch_no,mode=1,trend_output=None):

        # if "global_trend_feature" not in trend_output:
        #     trend_output["global_trend_feature"] = sw_index["global_trend_feature"][batch_no]
        # else:
        #     trend_output["global_trend_feature"] = torch.cat([trend_output["global_trend_feature"],sw_index["global_trend_feature"][batch_no]])
        combine_trend_output = {}
        for key in self.scale_dict.keys():
            index_data = sw_index[key][batch_no]
            combine_trend_output[key] = index_data
            if mode==1:
                index_data = index_data.unsqueeze(0)
            else:
                index_data = np.expand_dims(index_data,0)
            if key not in trend_output:
                trend_output[key] = index_data
            else:
                trend_output[key] = torch.cat([trend_output[key],index_data],dim=0)        
                             
        return combine_trend_output        

    def build_scale_trend_target(self,target,target_info=None,ins_rel_index=None,trend_target=None):

        for key in self.scale_dict.keys():
            scale_arr = self.scale_dict[key]
            scale_target = []
            scale_value = torch.Tensor(target_info['scale_arr'][key]).to(ins_rel_index.device)
            # scale_diff_sec_value = torch.Tensor(target_info['scale_arr'][key+'_sec']).to(ins_rel_index.device) 
            for i,instruments in enumerate(scale_arr):
                # scale_diff_sec = scale_diff_sec_value[i:i+1,:self.input_chunk_length]
                instruments = tensor_intersect(instruments,ins_rel_index)
                if instruments.shape[0]==0:
                    scale_target.append(torch.tensor(0).to(ins_rel_index.device))
                else:
                    if scale_value.shape[0]<i+1:
                        scale_target_mean = torch.tensor(0).to(ins_rel_index.device)
                    else:
                        scale_target_mean = target[instruments].mean() # scale_value[i]
                    scale_target.append(scale_target_mean)                    
            scale_target = torch.stack(scale_target).unsqueeze(0)
            if key not in trend_target:
                trend_target[key] = scale_target
                # trend_target[key+'_sec'] = scale_diff_sec
            else:
                trend_target[key] = torch.cat([trend_target[key],scale_target],dim=0)
                # trend_target[key+'_sec'] = torch.cat([trend_target[key+'_sec'],scale_diff_sec],dim=0)
        return trend_target        
           
    
    def forward(self, output_ori,target_ori,sw_ins_mappings=None,optimizers_idx=0,top_num=5,scale_class_weights=None,trend_threhold=None):
        """Multiple Loss Combine"""

        (output,vr_class,_) = output_ori
        (target,future_covs,target_class,future_round_targets,index_round_targets,price_targets,past_target,target_info) = target_ori
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
                scale_data_output = {}
                scale_data_target = []
                trend_ref = None
                for j in range(target_class.shape[0]):
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    index_target_item = future_index_round_target[j,indus_rel_index,:,i]
                    indus_index = tensor_intersect(keep_index,indus_data_index).to(keep_index.device)
                    inner_class_item = target_class_item[indus_data_index]                            
                    ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                    target_item = target[j,ins_all,target_len,0]
                    target_seq_item = target[j,ins_all,:,0]
                    target_info_item = target_info[j][main_index_abs]
                    date = target_info_item['future_start_datetime']
                    # 收集数据用于批次比较
                    self.build_scale_data_output(sv[0],j,scale_data_output=scale_data_output)   
                    target_in_batch = torch.zeros([target_item.shape[0]]).to(ins_rel_index.device)
                    target_in_batch[ins_rel_index] = target_item[ins_rel_index]
                    scale_data_target.append(target_in_batch)
                    # 收集趋势业务数据
                    self.build_scale_trend_output(sw_index_logits[0],j,trend_output=trend_output)
                    self.build_scale_trend_target(target_item,target_info=target_info_item,trend_target=trend_target,ins_rel_index=ins_rel_index)   
                                     
                    if ins_rel_index.shape[0]<3:
                        continue
                    target_info_total.append(target_info[j])             
                    
                    # 不同模式的损失计算                          
                    if target_mode==2:
                        # 比较全部品种，辅助整体指数比较
                        ref_indicator = 0 
                        # 使用价格指标作为主要指标
                        price_diff_range = price_targets[j,ins_rel_index]  
                        price_diff_range_all = price_targets[j,ins_all]  
                        round_targets_item = future_round_targets[j,ins_all,target_len,0]
                        node_num = ins_all.shape[0]
                        # 根据网络输出，生成针对性业务分支输出,并依次计算损失
                        scale_output = self.build_scale_output(sv[0],j)
                        # 业务分支top损失计算                      
                        for sidx, key in enumerate(scale_output.keys()):
                            sv_out_item = scale_output[key]
                            # 参考趋势输出，作为top选取参数
                            if sidx==0:
                                cls_loss[i] += self.compute_popu_top_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index)
                                # cls_loss[i] += self.compute_popu_weight_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index,trend_threhold=trend_threhold)
                            if sidx==1:
                                ce_loss[i] += self.compute_popu_top_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index)
                                # ce_loss[i] += self.compute_popu_weight_loss(sv_out_item,target_item,key=key,ins_rel_index=ins_rel_index,trend_threhold=trend_threhold)
                                
                        batch_size += 1          
                                  
                    elif target_mode==3:
                        batch_size += 1
                        
                if target_mode in [0]:
                    loss_sum = loss_sum + ce_loss[i]    
                if target_mode in [2]:  
                    cls_loss[i] = cls_loss[i]/batch_size  
                    ce_loss[i] = ce_loss[i]/batch_size
                    fds_loss[i] = fds_loss[i]/batch_size
                if target_mode in [3]:
                    min_threhold = trend_threhold['min']
                    max_threhold = trend_threhold['max']       
                    short_threhold = trend_threhold['short']   
                    long_threhold = trend_threhold['long']                
                    scale_out_total = []
                    scale_target_total = []
                    batch_size_inner = 0
                    for key_idx,key in enumerate(trend_output.keys()):
                        scale_output = trend_output[key]
                        scale_target = trend_target[key]
                        for h in range(2):
                            target = scale_target[:,h]
                            target_norm = map_to_neg1_pos1_torch(target) # scale_value(target,target.min(),target.max(),min_threhold,max_threhold)
                            if torch.sum(target==0)/target.shape[0]>0.5:
                                continue 
                            scale_output_norm = scale_output[:,h] # scale_value(scale_output[:,h],scale_output[:,h].min(),scale_output[:,h].max(),min_threhold,max_threhold)
                            pred_weights = self.get_sample_weights(scale_output_norm, short_threhold, long_threhold)
                            if key_idx==0:
                                cls_loss[i] += self.compute_weight_top_loss(scale_output_norm, target_norm,pred_weights)
                            else:
                                ce_loss[i] += self.compute_weight_top_loss(scale_output_norm, target_norm,pred_weights)
                            # target_weights = self.get_sample_weights(target_norm, short_threhold, long_threhold)
                            # ce_loss[i] += self.criterion(scale_output_norm, target_norm,target_weights)
                            # scale_target_total.append(target)
                            # scale_out_total.append(scale_output[:,h])
                            # ce_loss[i] += HuberLoss()(scale_output_norm[-4:], target[-4:])
                            # pred_label = torch.max(scale_output[:,h],1)[1]
                            # pred_label_total.append(pred_label)
                            # scale_target_class_total.append(scale_target_class)
                        batch_size_inner += 1
                    # scale_out_total = torch.stack(scale_out_total).transpose(1,0)
                    # scale_target_total = torch.stack(scale_target_total).transpose(1,0)
                    # cls_loss[i] = HuberLoss()(scale_out_total,scale_target_total)
                    
                    cls_loss[i] = cls_loss[i]/batch_size_inner
                    ce_loss[i] = ce_loss[i]/batch_size_inner
                    # if optimizers_idx==-1:
                    #     scale_target_class_total = torch.cat(scale_target_class_total)
                    #     pred_label_total = torch.cat(pred_label_total)
                    #     f1 = f1_score(scale_target_class_total.cpu().numpy(), pred_label_total.cpu().numpy(), average='macro')
                    #     print(f"F1 score: {f1:.4f}")
                        
                    # 业务分支top整合损失计算   
                    # scale_data_target = torch.stack(scale_data_target,0)                   
                    # for sidx, key in enumerate(scale_data_output.keys()):
                    #     sv_out_item = scale_data_output[key]
                    #     # 参考趋势输出，作为top选取参数
                    #     if sidx==0:
                    #         cls_loss[i] += self.ccc_loss_comp(sv_out_item,scale_data_target)
                        # if sidx==1:
                        #     ce_loss[i] += self.ccc_loss_comp(sv_out_item,scale_data_target)
                                                
                loss_sum = loss_sum + cls_loss[i] + ce_loss[i] + fds_loss[i] + corr_loss[i]
                           
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss,predictions]    
    
    
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
    
            
