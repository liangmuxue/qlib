import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from losses.mtl_loss import UncertaintyLoss
from cus_utils.common_compute import batch_cov,batch_cov_comp,eps_rebuild,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from cus_utils.common_compute import tensor_intersect,normalization_axis,pairwise_compare
from .feature_loss import AdaptiveSingleFeatureLoss
from .triplet_loss import AdaptiveSemiHardTripletLoss,ContinuousSemiHardTripletLoss
from .triplet_miner import ContinuousTripletLossWithMemory,ContinuousTripletConfig
from .contrastive_regression_loss import TripletContrastiveRegressionLoss,ContrastiveRegressionLoss,PairwiseContrastiveRegressionLoss
from .arc_loss import RobustArcFaceRegression
from .rank_loss import GlobalLambdaRankLoss,GlobalListwiseRankingLoss,LambdaRankLoss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from audioop import minmax

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

class FuturesIndustryLoss(UncertaintyLoss):
    """整合不同行业板块，并基于策略选取的损失"""

    def __init__(self,ref_model=None,device=None,target_mode=None,embedding_size=16,lock_epoch_num=0,num_mixtures=5,output_chunk_length=2,cut_len=2,loss_weights=None):
        
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
        
        # 整体及品种特征值回归损失函数
        self.index_feature_loss = AdaptiveSingleFeatureLoss(loss_type='welsch', device=self.device)
        self.ins_feature_loss = AdaptiveSingleFeatureLoss(loss_type='cauchy', device=self.device)
        
        # 基于有监督回归的比对损失函数，用于衡量整体指数
        # self.contrast_loss = SupervisedContrastiveRegressionLoss(contrast_weight=0.5,base_loss='mse',temperature=0.5,device=self.device)
        
        # self.contrast_loss = TripletContrastiveRegressionLoss(distance_func=self.ccc_distance,margin=1.0,device=self.device)
        # self.contrast_loss = ContinuousSemiHardTripletLoss(pairwise_distance=self.ccc_distance,device=self.device)
        self.contrast_loss = RobustArcFaceRegression(embedding_size,out_dim=embedding_size,num_proxies=embedding_size,device=self.device)
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
        self.rank_loss = LambdaRankLoss()
        
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
        
        for i in range(len(output)):
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
                features_out_main_total = []
                
                for j in range(target_class.shape[0]):
                    target_info_inbatch = target_info[j]
                    # 如果存在缺失值，则忽略，不比较
                    target_class_item = target_class[j]
                    keep_index = torch.where(target_class_item>=0)[0]
                    index_target_item = future_index_round_target[j,indus_rel_index,:,i]
                    indus_index = tensor_intersect(keep_index,indus_data_index).to(keep_index.device)
                    inner_class_item = target_class_item[indus_data_index]                            
                    sv_out_item = sv[0][j]
                    ins_rel_index = torch.where(target_class_item[ins_all]>=0)[0].long()
                    if ins_rel_index.shape[0]<3:
                        continue
                    round_targets_item = future_round_targets_factor[j,ins_rel_index]  
                    # 样本太少则忽略
                    if round_targets_item.shape[0]<=3:
                        continue                        
                    future_covs_ins = future_covs[i][j,ins_rel_index,:,:]
                    future_covs_main = future_covs[i][j,main_index_abs,-1,:]
                    # 记录主指数的多个指标特征，后续计算对比损失
                    future_covs_main_total.append(future_covs_main)   
                    target_item = target[j,ins_rel_index,:,i]
                    features_out_main = sw_index_data[j,:-1]
                    output_main = sw_index_data[j,-1]
                    sv_out_item = sv_out_item[ins_rel_index]
                    price_diff_range = price_targets[j,ins_rel_index]  
                    price_diff_range_total = long_diff_seq_targets[j,0]     
                    price_index_total.append(price_diff_range_total)   
                    features_out_main_total.append(features_out_main)
                    dec_out_item = dec_out[j,ins_rel_index,i]
                    
                    # 不同模式的损失计算                          
                    if target_mode==0:
                        if indus_index.shape[0]<2:
                            continue
                        # 板块整体损失计算
                        ce_loss[i] += self.ccc_loss_comp(sw_index_data[j],index_target_item[:,-1])/10
                        # if target_mode==1: 
                        #     ce_loss[i] += self.mse_loss(sw_index_data[j].unsqueeze(-1),time_diff_targets.unsqueeze(-1))  
                    elif target_mode==2:
                        # 比较全部品种，辅助整体指数比较
                        target_info_item = np.array(target_info[j])[ins_rel_index.cpu().numpy()]
                        price_diff_range_ori = [(item['open_array'][-1] - item['open_array'][-self.output_chunk_length])/item['open_array'][-self.output_chunk_length]*100 for item in target_info_item]
                        # ce_loss[i] += nn.HuberLoss(delta=1.0)(dec_out_item.mean(), price_diff_range_total)  
                        ce_loss[i] += self.ccc_loss_comp(dec_out_item, round_targets_item)  
                        # ce_loss[i] += self.mse_loss(dec_out_item, target_item)  
                        # 使用价格指标作为主要指标
                        cls_loss[i] += self.ccc_loss_comp(sv_out_item,price_diff_range)  
                        # 衡量整体走势，借用1号目标
                        ref_indicator = 0
                        index_target_total.append(future_index_round_target[j,main_index,-1,ref_indicator])
                        sw_index_total.append(sw_index_data[j,0])
                    elif target_mode==3:
                        # 整体指数比较，辅助品种比较 
                        index_target = future_round_targets[j,main_index_abs,i]
                        index_target_total.append(index_target)
                        sw_index_total.append(output_main)
                        # 每个品种的输出，和协变量未来数据作为特征，进行比较
                        ce_loss[i] += self.ccc_loss_comp(dec_out_item,round_targets_item)
                        # ce_loss[i] += self.contrast_loss(future_covs_ins,dec_out_item,round_targets_item.unsqueeze(-1))
                        # 计算标量特征损失
                        # cls_loss[i] += self.index_feature_loss(sw_index_data.squeeze(-1)[j],index_target)
                        # ce_loss[i] += self.ins_feature_loss(dec_out_item.squeeze(-1),round_targets_item)
                        # ce_loss[i] += self.ccc_loss_comp(sv_out_item,price_diff_range)  
                        # index_target_total.append(future_round_targets[j,main_index_abs,i])
                        # sw_index_total.append(sw_index_data[j,0])    
                batch_size = len(sw_index_total)
                
                if target_mode in [0]:
                    loss_sum = loss_sum + ce_loss[i]           
                if target_mode in [2]:
                    cls_loss[i] = cls_loss[i]/target_class.shape[0]
                    ce_loss[i] = ce_loss[i]/target_class.shape[0]
                    # 板块整体损失计算,批次内样本比较
                    index_target_total = torch.stack(index_target_total)
                    # 对目标值在批次内进行归一化
                    index_target_total = normalization_axis(index_target_total)
                    dec_out_item = dec_out[j]
                    features_out_main_total = torch.stack(features_out_main_total)
                    predictions, arc_loss, loss_dict = self.contrast_loss(features_out_main_total,index_target_total)
                    fds_loss[i] += arc_loss/batch_size         
                    # fds_loss[i] += self.ccc_loss_comp(torch.stack(sw_index_total),index_target_total)    
                    loss_sum = loss_sum + cls_loss[i] + ce_loss[i]           
                if target_mode in [3]:
                    # 板块整体损失计算,批次内样本比较
                    sw_index_total = torch.stack(sw_index_total)
                    main_price_index_total = torch.stack(price_index_total)  
                    index_target_total = torch.stack(index_target_total)
                    future_covs_main_total = torch.stack(future_covs_main_total)  
                    future_covs_main_total = normalization_axis(future_covs_main_total)
                    features_out_main_total = torch.stack(features_out_main_total).squeeze(-1)
                    # 对目标值在批次内进行归一化
                    index_target_total = normalization_axis(index_target_total)
                    # 辅助任务为对比损失
                    if batch_size<3:
                        # 如果批次内数据过少，则使用标准损失
                        cls_loss[i] += self.ccc_loss_comp(sw_index_total,index_target_total)
                    else:
                        predictions, arc_loss, loss_dict = self.contrast_loss(features_out_main_total,index_target_total)
                        # eva_loss = self.ccc_loss_comp(predictions, index_target_total)
                        cls_loss[i] += arc_loss/batch_size 
                    ce_loss[i] = ce_loss[i]/batch_size
                    # ce_loss[i] += self.ccc_loss_comp(features_out_main_total,future_covs_main_total)
                    # cls_loss[i] += self.contrast_loss(features_out_main_total,index_target_total)
                    # 主任务为排序学习损失
                    # sw_index_total = normalization_axis(sw_index_total)
                    # cls_loss[i] += self.rank_loss.forward(sw_index_total.unsqueeze(0),index_target_total.unsqueeze(0))
                    # ce_loss[i] = ce_loss[i]/batch_size
                    loss_sum = loss_sum + cls_loss[i] + ce_loss[i]
                if target_mode in [6]:
                    # 衡量目标值与前面各段已知结果比较的相对位置，作为优化目标，整体指标模式
                    diff_target = long_diff_seq_targets[:,main_index,-1,i]
                    ce_loss[i] = torch.abs(sw_index_data.squeeze()-diff_target).mean()        
                    loss_sum = loss_sum + ce_loss[i]      
                           
        return loss_sum,[corr_loss,ce_loss,fds_loss,cls_loss,predictions]    

    def compute_batch_last_distance(self,features,targets):
        """计算批次内最后一条数据与前面数据的特征距离和实际目标距离的匹配度"""
        
        fea_distance_vec = pairwise_compare(features[-1:],features[:-1],self.ccc_distance).squeeze(0)
        tar_distance_vec = torch.abs(targets[-1] - targets[:-1]) 
        distance = self.ccc_loss_comp(fea_distance_vec, tar_distance_vec)
        return distance
        
            