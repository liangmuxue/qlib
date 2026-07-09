import os

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tsaug
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from pytorch_lightning.trainer.states import RunningStage
from sklearn.preprocessing import StandardScaler
from cus_utils.wise_corrcef import analyze_model_complexity
from cus_utils.process import create_from_cls_and_kwargs
from .mlp_module import MlpModule
import cus_utils.global_var as global_var
from cus_utils.visualization import plot_feature_heatmap,plot_sample_lines,plot_grouped_bar
from darts_pro.act_model.union_transformer import UnionTransCombine
from darts_pro.act_model.fur_industry_ts import FurIndustryMixer, FurStrategy
from losses.mixer_loss import FuturesIndustryLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from darts_pro.act_model.union_transformer import TimeFeatureEncoder
from .multiTask_optimizer import MultiTaskOptimizer,analyze_similarity
from cus_utils.common_compute import linear_map, pairwise_compare, min_max_norm,scale_value, normalization_axis
from tft.class_define import CLASS_SIMPLE_VALUES, get_simple_class
from trader.utils.data_stats import DataStats, RESULT_FILE_PATH, RESULT_FILE_VIEW, INTER_RS_FILEPATH
from darts_pro.tft_futures_dataset import get_scale_conf,concat_scale_arr,emb_scale_arr,get_scale_cate_ins
import matplotlib.pyplot as plt

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# TRACK_DATE = [20250728,20250715,20250731]
TRACK_DATE = [20250812, 20250811, 20250825, 20250728, 20250715, 20250731]
TRACK_DATE = [item for item in range(20250403,20250404)]
# TRACK_DATE = [item for item in range(20250328,20250329)]
# TRACK_DATE = [item for item in range(20241231,20250105)]
# TRACK_DATE = [20250403]
STAT_DATE = [20240428, 20260505]
# TRACK_DATE = [date for date in range(STAT_DATE[0],STAT_DATE[1]+1)]
INDEX_ITEM = 0
DRAW_SEQ = [0]
DRAW_SEQ_ITEM = [0]
DRAW_SEQ_DETAIL = [0]

class FeatureExtractorHook:
    def __init__(self, feature_dict, name,nodes_num=0):
        self.features = feature_dict
        self.name = name
        self.nodes_num = nodes_num

    def __call__(self, module, input_ori, output):
        """执行网络输入输出的变量记录"""
        
        name = self.name
        S = self.nodes_num
        B = input_ori[0].shape[0]
        if B>100 or B==S:
            B = B//S
        if B<3:
            return
        input = input_ori[0]
        if isinstance(output,tuple):
            output = output[0]
        if module.training:
            name = "train_" + name
            input = input.detach()
            output = output.detach()
        else:
            name = "val_" + name
        # 训练阶段和验证阶段关注不同的内容
        self.features[name + "_input"] = input.reshape(B,S,-1)
        self.features[name + "_output"] = output.reshape(B,S,-1)    
          
def build_scale_arr(sw_ins_mappings):
    """对品种按照不同业务范围进行分组"""
    
    indus_data_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
    # 按照行业分组
    scale_conf,_ = get_scale_conf()
    indus_code = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
    threhold_bin = [['ZS_CDIFI','ZS_HSJS'],['ZS_ABPFI','ZS_YZYL','ZS_NFFI']]
    # threhold_bin = [['ZS_CDIFI'],['ZS_NFFI','ZS_HSJS'],['ZS_ABPFI','ZS_YZYL']]
    indus_scale_arr = [None for _ in range(len(threhold_bin))]
    for i in range(len(indus_code)):
        for j in range(len(threhold_bin)):
            if indus_code[i] in threhold_bin[j]:
                indus_scale_arr[j] = indus_data_index[i] if indus_scale_arr[j] is None else np.concatenate([indus_scale_arr[j],indus_data_index[i]])
    magin_radio = FuturesMappingUtil.get_magin_radio_flags(sw_ins_mappings).astype(int)
    create_year = FuturesMappingUtil.get_create_year_flags(sw_ins_mappings).astype(int)
    # 交易保证金比例分组
    threhold_bin = [[0,18],[18,100]]
    mr_scale_arr = [np.where((magin_radio>=threhold[0])&(magin_radio<threhold[1]))[0] for threhold in threhold_bin]
    # 按照是否包含夜盘来分组
    night_flag = FuturesMappingUtil.get_night_flag_ids(sw_ins_mappings)
    nt_scale_arr =  [np.where(night_flag==i)[0] for i in range(2)]
    # 创建年份分组
    threhold_bin = [[0,2013],[2013,2030]]
    # threhold_bin = [[0,2012],[2012,2018],[2018,2030]]
    cy_scale_arr = [np.where((create_year>threhold[0])&(create_year<=threhold[1]))[0] for threhold in threhold_bin]
    
    # 统合分组
    combine_scale_arr = [nt_scale_arr[0],[],[]]
    for instrument_idx in nt_scale_arr[1]:
        # 对包含夜盘的品种，再按照创建年份来分
        if create_year[instrument_idx]<=2012:
            combine_scale_arr[1].append(instrument_idx)
        else:
            combine_scale_arr[2].append(instrument_idx)
                
    # scale_arr = (cy_scale_arr,nt_scale_arr,mr_scale_arr,indus_scale_arr)
    scale_arr_total = {'indus_scale':indus_scale_arr,'cy_scale':cy_scale_arr,'nt_scale':nt_scale_arr,'mr_scale':mr_scale_arr}
    scale_arr = {key:scale_arr_total[key] for key in scale_conf.keys()}
    # scale_arr = {'indus_scale':indus_scale_arr,'cy_scale':cy_scale_arr}
    
    return scale_arr

def build_mul_scale_arr(sw_ins_mappings,mode=0,dataset=None):
    """对品种按照不同业务范围进行分组"""
    
    indus_data_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
    ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
    scale_data = []
    if mode==0:
        # 按照是否包含夜盘以及交易所分组
        exchange_ids = FuturesMappingUtil.get_exchange_ids(sw_ins_mappings)
        exc_u_id = np.unique(exchange_ids)
        night_flag = FuturesMappingUtil.get_night_flag_ids(sw_ins_mappings)
        nt_scale_arr = [np.where(night_flag==i)[0] for i in range(2)]    
        def check_in_exchange(ins_arr,exchange_ids):
            rtn = {'exc_{}'.format(i):[] for i in range(exc_u_id.shape[0])}
            for ins in ins_arr:
                for j in range(exc_u_id.shape[0]):
                    if exchange_ids[ins]==exc_u_id[j]:
                        rtn['exc_{}'.format(j)].append(ins)
            return rtn
        
        for i,instruments in enumerate(nt_scale_arr):
            p0 = 'nt_{}'.format(i)
            if i==0:
                item = {'p0':p0,'p1':p0,'instruments':instruments}
                scale_data.append(item)      
                continue          
            items = check_in_exchange(instruments,exchange_ids)
            for key in items.keys():
                if len(items[key])<=2:
                    continue
                item = {'p0':p0,'p1':key,'instruments':np.array(items[key])}
                scale_data.append(item)
        scale_data = pd.DataFrame(scale_data)   
    if mode==1:
        # 按照行业类别分组
        indus_code = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
        threhold_bin = [['ZS_CDIFI','ZS_HSJS'],['ZS_ABPFI','ZS_YZYL','ZS_NFFI']]
        # threhold_bin = [['ZS_CDIFI'],['ZS_NFFI','ZS_HSJS'],['ZS_ABPFI','ZS_YZYL']]
        for i in range(len(indus_code)):
            if indus_code[i] in threhold_bin[0]:
                seq = 0
            else:
                seq = 1
            key = "indus_{}".format(seq)
            if len(indus_data_index[i])<=2:
                continue            
            item = {'p0':key,'p1':indus_code[i],'instruments':indus_data_index[i]}
            scale_data.append(item)
        scale_data = pd.DataFrame(scale_data)           
    if mode==2:
        # 按照交易所以及行业类别分组
        indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
        threhold_bin = [['ZS_CDIFI','ZS_HSJS'],['ZS_ABPFI','ZS_YZYL','ZS_NFFI']]        
        exchange_ids = FuturesMappingUtil.get_exchange_ids(sw_ins_mappings)
        u_exc_ids = np.unique(exchange_ids)
        # def check_in_indus(ins_arr):
        #     rtn = {indus_codes[i]:[] for i in range(indus_data_index.shape[0])}
        #     for ins in ins_arr:
        #         for j in range(indus_data_index.shape[0]):
        #             if ins in indus_data_index[j]:
        #                 rtn[indus_codes[j]].append(ins)
        #     return rtn
        
        def check_in_indus(ins_arr):
            rtn = {'indus_{}'.format(i):[] for i in range(indus_data_index.shape[0])}
            for ins in ins_arr:
                for j in range(indus_data_index.shape[0]):
                    if ins in indus_data_index[j]:
                        rtn['indus_{}'.format(j)].append(ins)
            return rtn
                
        for i,exchange_id in enumerate(u_exc_ids):
            p0 = 'ex_{}'.format(exchange_id)
            p0_code,p0_name = dataset.get_exchange(exchange_id)
            instruments = np.where(exchange_id==exchange_ids)[0]    
            items = check_in_indus(instruments)
            for key in items.keys():
                if len(items[key])<=2:
                    continue
                key_no = int(key.split("_")[1])
                indus_code = indus_codes[key_no][3:].lower()
                p1_code,p1_name = dataset.get_industry_by_code(indus_code)
                item = {'p0':p0,'p0_code':p0_code,'p0_name':p0_name,'p1':key,'p1_name':p1_name,'p1_code':p1_code,'instruments':np.array(items[key])}
                scale_data.append(item)
        scale_data = pd.DataFrame(scale_data)     
    if mode==3:
        # 按照是否包含夜盘以及行业类别分组
        night_flag = FuturesMappingUtil.get_night_flag_ids(sw_ins_mappings)
        nt_scale_arr = [np.where(night_flag==i)[0] for i in range(2)]    
        def check_in_indus(ins_arr):
            rtn = {'indus_{}'.format(i):[] for i in range(indus_data_index.shape[0])}
            for ins in ins_arr:
                for j in range(indus_data_index.shape[0]):
                    if ins in indus_data_index[j]:
                        rtn['indus_{}'.format(j)].append(ins)
            return rtn
        
        for i,instruments in enumerate(nt_scale_arr):
            p0 = 'nt_{}'.format(i)
            if i==0:
                item = {'p0':p0,'p1':p0,'instruments':instruments}
                scale_data.append(item)      
                continue          
            items = check_in_indus(instruments)
            for key in items.keys():
                if len(items[key])<=2:
                    continue
                item = {'p0':p0,'p1':key,'instruments':np.array(items[key])}
                scale_data.append(item)
        scale_data = pd.DataFrame(scale_data)    
    if mode==5:
        # 按照行业类别分组
        indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
        for i,indus_code in enumerate(indus_codes):
            ins = FuturesMappingUtil.get_instrument_rel_index_within_industry(sw_ins_mappings,i)
            p0 = indus_code
            p1 = indus_code
            _,indus_name = dataset.get_industry_by_code(indus_code[3:].lower())
            item = {'p0':indus_code,'p0_code':indus_code,'p0_name':indus_name,'p1':indus_code,'p1_name':indus_name,'p1_code':indus_code,'instruments':ins}
            scale_data.append(item)
        scale_data = pd.DataFrame(scale_data)   
    if mode==6:
        # 按照行业类别以及交易所分组
        indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
        exchange_ids = FuturesMappingUtil.get_exchange_ids(sw_ins_mappings)
        u_exc_ids = np.unique(exchange_ids)
        
        def check_in_exchange(ins_arr):
            rtn = {uid:[] for uid in u_exc_ids}
            for ins_idx in ins_arr:
                rtn[exchange_ids[ins_idx]].append(ins_idx)
            return rtn
        
        ins_arr = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        for i,indus_code in enumerate(indus_codes):
            ins = ins_arr[i]
            p0 = indus_code[3:].lower()
            _,indus_name = dataset.get_industry_by_code(p0)
            items = check_in_exchange(ins)
            for key in items.keys():
                if len(items[key])<=2:
                    continue
                p1_code,p1_name = dataset.get_exchange(key)            
                item = {'p0':p0,'p0_code':p0,'p0_name':indus_name,'p1':p1_code,'p1_name':p1_name,'p1_code':p1_code,'instruments':np.array(items[key])}
                scale_data.append(item)
        scale_data = pd.DataFrame(scale_data) 
                                         
    return scale_data

class FuturesTransformerModule(MlpModule):
    """期货基于Transformer的双向判断的模型"""              

    def __init__(
        self,
        output_dim: Tuple[int, int],
        variables_meta_array: Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        cut_len=2,
        train_step_mode=1,
        use_weighted_loss_func=False,
        past_split=None,
        target_mode=None,
        scale_mode=None,
        batch_file_path=None,
        device="cpu",
        train_sw_ins_mappings=None,
        valid_sw_ins_mappings=None,
        pred_top_num=3,
        task_weights=None,
        grad_limits=None,
        opt_size=1,
        main_task_seq=None,
        **kwargs,
    ):
        self.mode = None
        self.train_sw_ins_mappings = train_sw_ins_mappings
        self.valid_sw_ins_mappings = valid_sw_ins_mappings
        self.scale_arr = build_mul_scale_arr(train_sw_ins_mappings,mode=6,dataset=global_var.get_value("dataset"))
        self.target_mode = target_mode
        self.scale_mode = scale_mode
        self.cut_len = cut_len
        self.pred_top_num = pred_top_num
        self.opt_size = opt_size
        # 阶段模式，0--表示全阶段， 1--表示第一阶段，先进行整体和行业预测 2--表示第二阶段，进行品种预测
        self.train_step_mode = train_step_mode
        # 任务初始权重
        self.task_weights = task_weights  
        self.grad_limits = torch.tensor(grad_limits)  
        self.pred_weights = [1.0, 0.0]
        self.main_task_seq = main_task_seq
        self.time_encoder = None
        self.nhead = 4
        self.cate_mode = 'cateMain'
        self.cate_mode = 'cateTotal'        
        # 趋势数值量级区间
        self.trend_threhold = None

        super().__init__(output_dim, variables_meta_array, num_static_components, hidden_size, lstm_layers, num_attention_heads,
                                    full_attention, feed_forward, hidden_continuous_size,
                                    categorical_embedding_sizes, dropout, add_relative_index, norm_type, past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func, batch_file_path=batch_file_path,
                                    device=device, **kwargs)  
        
        self.result_view_file_path = os.path.join(RESULT_FILE_PATH, RESULT_FILE_VIEW)
        self.coll_record_file_path = os.path.join(RESULT_FILE_PATH, "coll_record.csv")
        self.rate_file_path = os.path.join(RESULT_FILE_PATH, "rate.csv")
        # For pred step1 result
        self.inter_rs_filepath = os.path.join(RESULT_FILE_PATH, INTER_RS_FILEPATH)
        self.result_columns = ["date", "indus_index", "trend_flag", "price_inf", "ce_inf"]
        self.outer_call = False
        
    def set_outer_params(self, params):
        for name in params:
            setattr(self, name, params[name])       
    
    def create_real_model(self,
        output_dim: Tuple[int, int],
        variables_meta: Dict[str, Dict[str, List[str]]],
        num_static_components: int,
        hidden_size: Union[int, List[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: Dict[str, Tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        model_type="tft",
        device="cpu",
        seq=0,
        **kwargs):
        
            (
                past_target,
                past_covariate,
                historic_future_covariate,
                future_covariate,
                static_covariates,
                _,
                future_target,
                _,
                _,
                _,
                _,
                _,
                _
            ) = self.train_sample
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariate[..., past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            
            # 过去协变量维度计算,不使用时间协变量
            input_dim = (
                past_target_shape
                +past_covariates_shape
            )
            
            # 加入短期指标
            pred_len = self.output_chunk_length    
            combine_nodes = FuturesMappingUtil.get_all_instrument(self.train_sw_ins_mappings)
            combine_nodes_num = combine_nodes.shape[0]
            
            dataset = global_var.get_value("dataset") 
            # 记录时间字段
            self.embed_cols = dataset.get_future_columns()
            
            target_feat_dim = 1
            time_embed_dim = len(self.embed_cols)
            # 使用混合时间序列模型,TFT底座
            model = UnionTransCombine(
                target_feat_dim=target_feat_dim,
                static_num=static_covariates.shape[-1]-1,
                obs_dim=input_dim,
                fut_dim=future_covariate.shape[-1],
                time_embed_dim=time_embed_dim,
                d_model=64,
                hidden_dim=64,
                hidden_size=16,
                nhead=8,
                num_layers=4,
                dropout=dropout,
                seq_len=self.input_chunk_length,
                pred_len=pred_len,
                sample_dim=combine_nodes_num,
                # sample_heads=4,
                static_emb_dim=4,
                static_cate_emb=dataset.get_cate_dict(),
                scales_arr=self.scale_arr,
                device=self.device,
                target_mode=self.target_mode[seq]
            )         
            self.time_embed_dim = time_embed_dim      
            self.embedding_size = input_dim
            
            ################# 植入钩子进行中间变量输出调试 #################
            if seq==0:
                self.reg_hook(model,combine_nodes_num)
            
            return model
    
    def reg_hook(self,model,nodes_num):
        self.features = {}
        # self.inout_compare_names = ['pool_layer','encoder','decoder','attention_net','score_head']
        self.inout_compare_names = ['indus_scale','nt_scale','cy_scale','score_head0','score_head1','score_head2','score_head3']
        self.inout_compare_names = ['indus_scale','score_head0','score_head1']
        
        # # 输入部分的全局池化
        # model.trans_model.pool_layer.register_forward_hook(FeatureExtractorHook(self.features, 'pool_layer',nodes_num=nodes_num))
        # # 查看编码器层的输入输出
        # model.trans_model.transformer_encoder.register_forward_hook(FeatureExtractorHook(self.features, 'encoder',nodes_num=nodes_num))
        # # 查看主模型中解码层输出的前后数值
        # model.trans_model.tar_decoder.register_forward_hook(FeatureExtractorHook(self.features, 'decoder',nodes_num=nodes_num))
        # 查看后置模型中基于注意力的特征输出的前后数值
        # model.top_selector[0].top_att_layer.score_head.register_forward_hook(FeatureExtractorHook(self.features, 'score_head',nodes_num=nodes_num))
        
        # # 查看关注层的前后数值
        # model.top_selector[0].score_head[0].register_forward_hook(FeatureExtractorHook(self.features, 'score_head0',nodes_num=nodes_num))
        # for i,key in enumerate(self.scale_arr.keys()):
        #     model.top_selector[0].scales_layer[key].register_forward_hook(FeatureExtractorHook(self.features, key,nodes_num=nodes_num))   
        #     model.top_selector[0].score_head[i+1].register_forward_hook(FeatureExtractorHook(self.features, 'score_head{}'.format(i+1),nodes_num=nodes_num))
                          
    def create_loss(self, model, device="cpu"):
        combine_nodes = FuturesMappingUtil.get_all_instrument(self.train_sw_ins_mappings)
        return FuturesIndustryLoss(device=device, ref_model=model, lock_epoch_num=self.lock_epoch_num,input_chunk_length=self.input_chunk_length,output_chunk_length=self.output_chunk_length,
                                   opt_size=self.opt_size,embedding_size=self.embedding_size, target_mode=self.target_mode, trend_threhold=self.trend_threhold,
                                   cut_len=self.cut_len, loss_weights=self.task_weights,combine_nodes=combine_nodes,scale_dict=self.scale_arr)       

    def _construct_classify_layer(self, input_dim, output_dim, device=None):
        """新增策略选择模型"""
        
        self.lock_epoch_num = 180
        
        self.top_num = 5  # 选取目标数量
        self.select_num = 10  # 一次筛选的数量
        self.past_tar_dim = 3  # 参考过去数值的时间段
        
        strategy_model = FurStrategy(target_num=len(self.past_split), select_num=self.select_num, trend_threhold=self.trend_threhold)        
        return strategy_model
    
    def configure_optimizers(self):
        """定制优化器"""

        optimizers = []
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}  
        
        # 使用自定义优化器，用于调整多任务损失函数权重和梯度策略
        for i in range(self.opt_size):
            task_weights = self.task_weights[i]
            use_gradient_surgery_flag = (len(task_weights) > 1 and task_weights[1] > 0)
            mt_optimizer = MultiTaskOptimizer(nn.ModuleList(self.sub_models)[i].parameters(), optimizer_kws,
                            model=self.sub_models[i], task_weights=task_weights, main_task_seq=self.main_task_seq[i],grad_limits=self.grad_limits,
                            use_gradient_surgery=use_gradient_surgery_flag,
                            use_adaptive_clip=False, use_pcgrad=self.use_pcgrad,parent=self)  
            optimizers.append(mt_optimizer)
        # 对应优化器，生成多个学习率
        lr_schedulers = []
        for i in range(self.opt_size):
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizers[i]
            lr_monitor = lr_sched_kws.pop("monitor", None)
            lr_scheduler_config = self.create_lr_scheduler(lr_sched_kws,lr_monitor=lr_monitor)
            lr_schedulers.append(lr_scheduler_config)  
        lr_schedulers.append(lr_scheduler_config) 
        return optimizers, lr_schedulers     
    
    def create_lr_scheduler(self,lr_sched_kws,lr_monitor="val_loss"):
        # 余弦退火
        lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        # Linear Ls
        # lr_scheduler_cls = torch.optim.lr_scheduler.LinearLR
        lr_scheduler = create_from_cls_and_kwargs(
            lr_scheduler_cls, lr_sched_kws
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": self.lr_freq["interval"],
            "frequency": self.lr_freq["frequency"],
            "monitor": lr_monitor if lr_monitor is not None else "val_loss",
        } 
        return  lr_scheduler_config       

    def get_scale_match_key(self):
        # rel_scale_key = list(self.scale_arr.keys())[0]
        rel_scale_key = "nt_scale"
        return rel_scale_key
            
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """在原来基础上，添加策略选择模式"""
        
        out_total = []
        out_class_total = []
        batch_size = x_in[2].shape[0]
        
        sub_model_length = len(self.sub_models) 
        vr_class = (torch.ones([batch_size, self.select_num]).to(self.device),
                    torch.ones([batch_size]).long().to(self.device),
                    torch.ones([batch_size, self.select_num]).long().to(self.device))
        past_index_targets = x_in[-1]
        instruments = torch.Tensor(FuturesMappingUtil.get_all_instrument(self.train_sw_ins_mappings)).to(self.device).long()
        target_len = -self.output_chunk_length + self.cut_len - 1
        # 分别单独运行模型
        for i, m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]  
            # 使用指标整体数据作为输入部分  
            past_index_round_targets = past_index_targets[..., i]
            past_round_targets = x_in[6][..., i]
            futures_convs = x_in[3]
            his_future_covs = x_in[2]
            static_covs = x_in[4]
            # 暂存价格数据，用于可视化
            self.cur_price_targets = x_in[5]
            target_class = x_in[8][:,instruments]
            # 根据优化器编号匹配计算,当编号超出模型数量时，也需要全部进行向前传播，此时没有梯度回传，主要用于生成二次模型输入数据
            if optimizer_idx == i or optimizer_idx >= sub_model_length or optimizer_idx == -1:
                x_in_item = (past_convs_item, his_future_covs, futures_convs, past_round_targets, past_index_round_targets)
                past_convs_item = past_convs_item[:,instruments,:,:]
                his_future_covs = his_future_covs[:,instruments,:,:]
                past_round_targets = past_round_targets.permute(0,2,1)[...,instruments].unsqueeze(2)
                past_targets = past_convs_item[:,:,:,:1]
                futures_convs = futures_convs[:,instruments,:,:]
                static_covs = static_covs[:,instruments,:]
                # 批次内计算时间嵌入特征,按照品类分别计算
                def transform_emb(convs):
                    convs_emb = []
                    ref_emb = {}
                    for k in range(convs.shape[1]):
                        batch_data = {}
                        # 如果当前品种没有数据，则取其他品种的日期数据进行对照，避免嵌入异常
                        nodata_idx = torch.where(target_class[:,k]<0)[0]
                        if nodata_idx.shape[0]>0:
                            # 直接使用1号参考数值
                            convs[nodata_idx,k,:,:] = convs[nodata_idx,0,:,:]
                        for k_idx,key in enumerate(self.embed_cols):
                            # Ignore week property
                            if key=='week':
                                continue
                            batch_data[key] = convs[:,k,:,k_idx].flatten().cpu()   
                        emb_data = m.transform_inner(batch_data)
                        emb_data = emb_data.reshape(his_future_covs.shape[0],convs.shape[2],-1)
                        convs_emb.append(emb_data)
                    convs_emb = torch.stack(convs_emb).permute(1,0,2,3)
                    return convs_emb
                # his_future_emb = transform_emb(his_future_covs).to(self.device)   
                # future_emb = transform_emb(futures_convs).to(self.device) 
                # if his_future_emb.dtype==torch.float64:  
                #     future_emb = future_emb.double()    
                # future_single_emb =  future_emb[:,:,target_len,:]   
                future_single_conv =  his_future_covs[:,:,target_len,:]   
                input_final = (static_covs,past_convs_item, his_future_covs,future_single_conv)
                # input_final = (static_covs,past_convs_item, his_future_covs)
                if not hasattr(self, 'cur_epoch'):
                    self.cur_epoch = 0
                out = m(*input_final,current_epoch=self.cur_epoch,max_epochs=self.max_epochs)                
                out_class = torch.ones([batch_size, self.output_chunk_length, 1]).to(self.device)
            else:
                # 模拟数据
                out = (None,
                       torch.ones([batch_size, past_convs_item.shape[1], sub_model_length]).to(self.device),
                       None
                       )
                out_class = torch.ones([batch_size, 1]).to(self.device)
            out_total.append(out)    
            out_class_total.append(out_class)
        
        return out_total, input_final, out_class_total  
    
    def on_train_start(self): 
        super().on_train_start()
        sw_ins_mappings = self.train_sw_ins_mappings
        self.ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        self.criterion.reset_device()
            
    def on_validation_start(self): 
        sw_ins_mappings = self.valid_sw_ins_mappings
        self.ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        self.output_result = []
        
        self.criterion.reset_device()
        
        
    def on_validation_end(self): 
        pass
        
    def on_validation_epoch_start(self): 
        self.import_price_result = None
        self.total_imp_cnt = 0
        self.loss_result = {}
    
    def get_optimizer_size(self):
        return self.opt_size
       
    def training_step_real(self, train_batch, batch_idx): 
        """重载父类方法，重点关注多优化器配合"""
        
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            future_target,
            target_class,
            price_targets,
            past_future_round_targets,
            index_round_targets,
            future_week_info,
            target_info
        ) = train_batch
                
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class,target_info)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp,mode='train')
        future_covs = input_batch[1]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
            self.sub_models[0].current_epoch = self.epochs_trained  
            
        total_loss = torch.tensor(0.0).to(self.device)
        for i in range(self.get_optimizer_size()):
            (output, vr_class, tar_class) = self(input_batch, optimizer_idx=i)
            loss, detail_loss = self._compute_loss((output, vr_class, tar_class),
                            (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, future_week_info,target_info), optimizers_idx=i)
            (corr_loss, ce_loss, fds_loss, cls_loss, cls_detail) = detail_loss 
            if cls_loss[i] != 0:
                self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
                if cls_detail is not None:
                    for key in cls_detail.keys():
                        self.log("train_trunk_detail_{}".format(key), cls_detail[key], batch_size=train_batch[0].shape[0], prog_bar=False,sync_dist=True)                
            if ce_loss[i] != 0:
                self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            if fds_loss[i] != 0:
                self.log("train_fds_loss_{}".format(i), fds_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)       
            if corr_loss[i] != 0:
                self.log("train_corr_loss_{}".format(i), corr_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)                        
            self.loss_data.append((corr_loss.detach(), ce_loss.detach(), fds_loss.detach(), cls_loss.detach()))
            # 手动更新参数，使用自定义具备梯度校正功能的优化器
            opt = self.trainer.optimizers[i]
            task_weights = self.task_weights[i]
            if len(task_weights) == 3:
                update_info = opt.step_with_auto_weights([cls_loss[i], ce_loss[i], fds_loss[i]])
            elif len(task_weights) == 2:
                update_info = opt.step_with_auto_weights([cls_loss[i], ce_loss[i]])
            elif len(task_weights) == 4:
                update_info = opt.step_with_auto_weights([cls_loss[i], ce_loss[i], fds_loss[i],corr_loss[i]])                
            else:
                # 对于三元组损失，有可能没有样例，会返回0，需要忽略
                if cls_loss[i] == 0:
                    print("cls loss 0")
                    continue
                update_info = opt.step([cls_loss[i]])
            # 记录梯度信息，后续统计可视化使用
            if 'total_gradients' in update_info:
                self.features['total_gradients'] = update_info['total_gradients']
                self.features['gradient_components_ori'] = update_info['gradient_components_ori']
            # update_info = opt.step_with_batch([cls_loss[i],ce_loss[i]],batch_idx=batch_idx,total_batch_number=self.trainer.num_training_batches)
            task_weights = self.task_weights[i]
            if len(task_weights) > 1 and update_info is not None:
                # total_loss = total_loss + update_info["total_loss"]
                # 当前总梯度和分量梯度
                if "conflict_analysis" in update_info:
                    self.log("task_grad_norm_cls", update_info["task_grad_norms"][0], batch_size=train_batch[0].shape[0], prog_bar=False)
                    self.log("task_grad_norm_ce", update_info["task_grad_norms"][1], batch_size=train_batch[0].shape[0], prog_bar=False)
                    if len(task_weights) > 2:
                        self.log("task_grad_norm_fds", update_info["task_grad_norms"][2], batch_size=train_batch[0].shape[0], prog_bar=False)
                    if len(task_weights) > 3:
                        self.log("task_grad_norm_corr", update_info["task_grad_norms"][3], batch_size=train_batch[0].shape[0], prog_bar=False)                        
                    # self.log("conflict_cnt", update_info["conflict_analysis"]["conflict_count"], batch_size=train_batch[0].shape[0], prog_bar=False)
                    # self.log("similarity", update_info["conflict_analysis"]["similarity"], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("total_norm_encoder", update_info["total_norm_encoder"], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("total_norm_decoder", update_info["total_norm_decoder"], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("total_norm_ins_layer", update_info["total_norm_ins_layer"], prog_bar=True) 
            else:
                self.log("total_grad_norm", update_info["total_grad_norm"], batch_size=train_batch[0].shape[0], prog_bar=True)           
                self.log("total_norm_encoder", update_info["total_norm_encoder"], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("total_norm_decoder", update_info["total_norm_decoder"], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("total_norm_ins_layer", update_info["total_norm_ins_layer"], prog_bar=False)                 
                                       
        # self.log("train_loss", total_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)  
        # self.log("lr_last",self.trainer.optimizers[-2].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=False)  
        
        # 手动维护global_step变量  
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()

        # 可视化中间结果与实际目标的比较情况 
        # self.viz_in_out_data(mode="train") 
                
        # 可视化中间输出和结果的比对
        # if 'gradient_components_ori' in  self.features:
        #     gradient_components_ori = self.features['gradient_components_ori']
        #     similarity = analyze_similarity(gradient_components_ori,task_grads_size=len(self.task_weights[0]))     
        #     for i in range(len(self.task_weights[0])-1):
        #         self.log("grad_similarity_{}".format(i), similarity[i].item(),batch_size=train_batch[0].shape[0], prog_bar=False) 
        
        return total_loss, detail_loss, output 

    def on_train_epoch_start(self):
        self.loss_data = []
        self.cur_epoch = self.current_epoch
        
    def on_train_epoch_end(self):
        
        # 统一在每个epoch结束时再进行学习率更新
        for i in range(self.get_optimizer_size()):
            self.lr_schedulers()[i].step() 
        # 可视化权重和梯度
        for name,params in self.sub_models[0].named_parameters():
            global_step = self.global_step
            if not "top_selector.0" in name:
                continue
            if name.endswith("bias"):
                continue
            ind_flag = False
            for ind_name in self.inout_compare_names:
                if ind_name in name:
                    ind_flag = True
                    break
            if not ind_flag:
                continue               
            if params is None or params.shape[0]==0:
                continue
            self.logger.experiment.add_histogram('weights/' + name,params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram('grad/' + name,params.grad,global_step)
        # 可视化中间特征输出      
        for name,feat in self.features.items():
            if name=='total_gradients' or name=='gradient_components_ori':
                continue
            # 可视化重点层的输入输出数据
            self.logger.experiment.add_histogram(f'Features/{name}', feat.flatten(), self.current_epoch)  
            
        
        # 可视化梯度
        if 'total_gradients' in self.features:
            total_gradients = self.features['total_gradients']
            name_matches = self.inout_compare_names
            for grad_name in total_gradients.keys():
                for item in name_matches:
                    if item in (grad_name):
                        grad = total_gradients[grad_name]
                        self.logger.experiment.add_histogram('grad/' + grad_name,grad,global_step)

        self.cur_epoch = self.current_epoch
                
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        loss, detail_loss, output = self.validation_step_real(val_batch, batch_idx)
        (corr_loss_combine, ce_loss, fds_loss, cls_loss, cls_detail) = detail_loss

        batch_data = np.ones([1])
        self.dump_val_data(val_batch, output, batch_data)
        real_batch_size = output[-1].shape[0]
        batch_size = self.trainer.val_dataloaders[0].batch_size        
        self.dump_loss_data(detail_loss[-1],batch_size=batch_size,real_batch_size=real_batch_size)
        
        return loss
        
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        # 全部转换为2维模式进行网络计算
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            future_target,
            target_class,
            price_targets,
            past_future_round_targets,
            index_round_targets,
            future_week_info,
            target_info
        ) = val_batch
              
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class,target_info) 
        input_batch = self._process_input_batch(inp,mode='valid')
        future_covs = input_batch[1]
        (output, vr_class, vr_class_list) = self(input_batch, optimizer_idx=-1)
        scale_info = output[0][-1]
        # print("fur_scale:{},trans_scale:{}".format(scale_info[0],scale_info[1]))
        # 全部损失
        loss, detail_loss = self._compute_loss((output, vr_class, vr_class_list),
                    (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, future_week_info,target_info), optimizers_idx=-1)
        (corr_loss, ce_loss, fds_loss, cls_loss, cls_detail) = detail_loss
        
        if not self.outer_call:
            self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True, sync_dist=True)
            for i in range(self.opt_size):
                task_weights = self.task_weights[i]
                if ce_loss[i] != 0 and len(task_weights) > 1:
                    self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True,sync_dist=True)
                if cls_loss[i] != 0:
                    self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True,sync_dist=True)
                    if cls_detail is not None:
                        for key in cls_detail.keys():
                            if key=='total':
                                self.log("val_trunk_detail_{}".format(key), cls_detail[key], batch_size=val_batch[0].shape[0], prog_bar=True,sync_dist=True)  
                            else:
                                self.log("val_trunk_detail_{}".format(key), cls_detail[key], batch_size=val_batch[0].shape[0], prog_bar=False,sync_dist=True)  
                if fds_loss[i] != 0 and len(task_weights) > 2:
                    self.log("val_fds_loss_{}".format(i), fds_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True,sync_dist=True)                
                if corr_loss[i] != 0 and len(task_weights) > 3:
                    self.log("val_corr_loss_{}".format(i), corr_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True,sync_dist=True)   

        output_combine = (output, vr_class, price_targets, past_future_round_targets)
        
                
        return loss, detail_loss, output_combine       

    def _process_input_batch(
        self, input_batch,mode='train'
    ):
        """重载方法，以适应数据结构变化"""
        (
            past_target,
            future_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            price_targets,
            past_future_round_targets,
            index_round_targets,
            target_class,
            target_info
        ) = input_batch
        dim_variable = -1

        def rebuild_covariates(covariates, targets):
            x_cov_array = []
            for i, p_index in enumerate(self.past_split):
                conv_index = self.past_split[i]
                covariates_item = covariates[..., conv_index[0]:conv_index[1]]
                # 修改协变量生成模式，只取自相关目标作为协变量，不使用时间协变量（时间协变量不进行归一化，只用于EMB嵌入）
                conv_defs = [
                            targets[..., i:i + 1],
                            covariates_item,
                    ]            
                x_past = torch.cat(
                    [
                        tensor
                        for tensor in conv_defs if tensor is not None
                    ],
                    dim=dim_variable,
                )
                x_cov_array.append(x_past)
            return x_cov_array
        
        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = rebuild_covariates(past_covariates, past_target)
        # 生成未来协变量，用于特征比对模式
        x_future_array = rebuild_covariates(past_future_covariates, future_target)
          
        # 忽略静态协变量第一列(索引列),后边的都是经过归一化的
        static_covariates = static_covariates[..., 1:]
        # 切分出过去整体round数值,规则为全部过去数值-冗余值(预测长度)-1l
        past_index_targets = index_round_targets[:,:,:self.input_chunk_length,:]
        sw_ins_mappings = self.train_sw_ins_mappings if mode=='train' else self.valid_sw_ins_mappings
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        past_index_targets = past_index_targets[:, indus_rel_index,:,:]
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:self.input_chunk_length,:]
        
        price_targets_real = price_targets # np.array([t['open_diff'] for t in target_info])
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, x_future_array, historic_future_covariates, future_covariates, 
                static_covariates, price_targets_real, past_round_targets, past_index_targets,target_class)
    
    def _compute_loss(self, output, target, optimizers_idx=0):
        """重载父类方法"""

        (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, past_target, target_info) = target 
        # 只保留最后一天的数值，作为损失目标
        future_round_targets = past_future_round_targets[:,:,-self.output_chunk_length:,:]  
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        tft_dataset = global_var.get_value("dataset") 
        return self.criterion(output, (future_target, future_covs, target_class, future_round_targets, index_round_targets, price_targets, past_target, target_info),
                    sw_ins_mappings=sw_ins_mappings, optimizers_idx=optimizers_idx, top_num=self.top_num, trend_threhold=self.trend_threhold)        

    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        rate_total, coll_result,trend_result,cate_result = self.combine_result_data(self.output_result, pred_top_num=self.pred_top_num)
        if rate_total is None:
            return
        self.rate_total = rate_total
        date_total_num = float(coll_result['date'].unique().shape[0])
        
        # 生成epoch内损失均值数据
        loss_result = pd.DataFrame(self.loss_result)
        real_batch_size_sum = 0
        batch_size_sum = 0
        loss_result_combine = {key:0 for key in loss_result.columns}
        # 综合每个批次size与标准批次size，计算实际loss数据
        for i,row in loss_result.iterrows():
            real_batch_size = int(row['real_batch_size'])
            real_batch_size_sum += real_batch_size
            batch_size_sum += int(row['batch_size'])
            for key in row.keys():
                item = float(row[key])
                loss_result_combine[key] += item * real_batch_size
        loss_result_combine = {key:loss_result_combine[key]/real_batch_size_sum for key in loss_result_combine.keys()}
        del loss_result_combine['batch_size']
        del loss_result_combine['real_batch_size']
        exclude_cols = ['total','main']
        ins_total_loss = 0
        sum_num = 0
        loss_result_combine_new = {}
        for key in loss_result_combine.keys():
            new_key = key
            if key not in exclude_cols:
                ins_total_loss += loss_result_combine[key]
                sum_num += 1
                new_key = 'ins/' + key
            else:
                new_key = 'cate_' + key
            loss_result_combine_new[new_key] = loss_result_combine[key]
        if ins_total_loss>0:
            loss_result_combine_new['ins_total'] = ins_total_loss/sum_num
        self.loss_result = loss_result_combine_new
        
        # 打印相关指标
        if rate_total is not None and rate_total.shape[0] > 0:
            dur_num = self.cut_len - 1
            anno_yield = rate_total['yield_rate'].values[0] * (240 / date_total_num) / (2 * self.pred_top_num * dur_num) 
            self.anno_yield = anno_yield
            if self.outer_call:
                return                 
                     
            for col in rate_total.columns:
                if col != "total_cnt":
                    self.log(col, rate_total[col].values[0], prog_bar=True,sync_dist=True)  
            self.log("anno_yield", anno_yield, prog_bar=True,sync_dist=True) 
            self.log("trend_eva_diff", rate_total["trend_eva_diff"].values[0], prog_bar=True,sync_dist=True) 
            # self.log("cate_corr_rate", rate_total["cate_corr_rate"].values[0], prog_bar=True,sync_dist=True)
            # self.log("cate_yield", rate_total["cate_yield"].values[0], prog_bar=True,sync_dist=True)  
                
        output_3d, past_target_3d, future_target_3d, target_class_3d, price_targets_total, \
            past_future_round_targets_total, future_week_info_total, index_round_targets_3d, target_info_3d = self.combine_output_total(self.output_result)
        viz_total_size = 0
        
        if self.mode is None or not self.mode.startswith("pred_"):
            # 验证模式，进行board的可视化
            # self.viz_data_board()
            dig_info = self.dig_result_info(coll_result)
            self.viz_dig_info(dig_info)       
            return
        
        # 测试模式，在此进行结果的可视化
        print("all date:", coll_result['date'].unique())
        coll_result.to_csv(self.coll_record_file_path, index=False)
        pred_data_path = os.path.join(RESULT_FILE_PATH, self.pred_index_data_path)
        self.log("date_total_num", date_total_num, prog_bar=True,sync_dist=True) 
        # 生成进一步的结果指标
        coll_result_output = coll_result.rename(columns={'trend_value':'pred_trend'})
        # stats = DataStats(work_dir=RESULT_FILE_PATH,backtest_dir="/home/qdata/workflow/fur_backtest_flow/trader_data/05") 
        # stat_result = stats.compute_val_result(coll_result.rename(columns={'trend_value':'pred_trend'}))
        col_data_types = {"top_index":int, "instrument":str, "yield_rate":float, "result":int, "pred_trend":int, "date":int}               
        if os.path.exists(self.result_view_file_path):
            import_price_result_total = pd.read_csv(self.result_view_file_path, dtype=col_data_types)  
            # 去重
            date_min = coll_result_output["date"].min()
            date_max = coll_result_output["date"].max()
            import_price_result_total = import_price_result_total[(import_price_result_total['date'] < date_min) | (import_price_result_total['date'] > date_max)]
            import_price_result_total = pd.concat([import_price_result_total, coll_result_output])
            import_price_result_total.to_csv(self.result_view_file_path) 
        else: 
            coll_result_output.to_csv(self.result_view_file_path)                 
        
        viz_result = global_var.get_value("viz_result")
        viz_result_ext = global_var.get_value("viz_result_ext")
        viz_result_detail = global_var.get_value("viz_result_detail")
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        main_index_rel = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_names_all = FuturesMappingUtil.get_industry_names(sw_ins_mappings)
        indus_names = indus_names_all[indus_rel_index]
        indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
        industry_instrument_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        trend_data  = output_3d[3]
        features = output_3d[2]
        trend_logits = output_3d[4]  
        batch_trend_data = output_3d[5]
        predictions = batch_trend_data
        node_num = ins_all.shape[0]
        match_key = self.get_scale_match_key()
        
        first_date = None
        for index in range(target_class_3d.shape[0]):
            viz_total_size += 1
            target_class_item = target_class_3d[index]
            keep_index = np.where(target_class_item >= 0)[0]
            round_targets = past_future_round_targets_total[index]
            future_target = future_target_3d[index]
            ts_arr = target_info_3d[index]
            date = int(ts_arr[keep_index][0]["future_start_datetime"])
            if index==0:
                first_date = date
            if not date in TRACK_DATE:
                continue    
            instruments, _, _ = np.intersect1d(ins_all, keep_index, return_indices=True)
            ts_arr_ins = ts_arr[instruments]
            price_diff_range_ins = np.array([(item['open_array'][-self.output_chunk_length + self.cut_len - 1] - item['open_array'][-self.output_chunk_length]) / item['open_array'][-self.output_chunk_length] * 100 for item in ts_arr_ins])
            inner_class_item = target_class_item[ins_all]
            inner_index = np.where(inner_class_item >= 0)[0]           
            # dec_output_item = dec_output[index,inner_index,-1,j] 
            price_targets = price_targets_total[index, instruments]
            # 品种比对图
            scale_arr = emb_scale_arr(self.scale_arr)
            fur_target_total = []
            ins_in_scale_total = []
            ins_output_total = []
            for k,p0 in enumerate(scale_arr):
                scale_item_p0 = scale_arr[p0]
                for j,key_p1 in enumerate(scale_item_p0.keys()):
                    scale_item = scale_item_p0[key_p1]
                    ins = scale_item['instruments']
                    # print("p0_{},p1_{} ins shape:{}".format(p0,key_p1,ins.shape[0]))
                    # 取得相对位置，用于匹配子趋势
                    rel_ins = scale_item['rel_ins']
                    p1 = scale_item['p1']   
                    p0_name = scale_item['p0_name']      
                    p1_name = scale_item['p1_name']     
                    ins_output_outer = features[p0][index]          
                    ins_output_item = ins_output_outer[rel_ins]
                    ins_output_total.append(ins_output_item)
                    ins_in_scale = np.intersect1d(ins,instruments)
                    ins_in_scale_total.append(ins_in_scale)
                    price_array_range = np.array([self.criterion.compute_diff_range_class(item)[0] for item in ts_arr[ins_in_scale]])
                    price_array_range = price_array_range / 10                
                    # fur_round_target = round_targets[ins_in_scale, -self.output_chunk_length+self.cut_len-1, 0]
                    fur_target = future_target[ins_in_scale, -self.output_chunk_length+self.cut_len-1, 0]
                    fur_target_total.append(fur_target)
                    coll_item = coll_result[(coll_result['date']==date)&(coll_result['p1']==key_p1)]
                    trend_result_item = trend_result[(trend_result['date']==date)&(trend_result['p0']==p0)&(trend_result['p1']==key_p1)]
                    cate_result_item = cate_result[(cate_result['date']==date)]
                    self.draw_ins_visdom(ins_in_scale, ins_output_item, fur_target, ts_arr,coll_item=coll_item,scale_item=scale_item,cate_result=cate_result_item,
                                trend_result=trend_result_item, date=date, iter_num='{}_{}_{}'.format(k,j,date), key=(p0_name+'/'+p1_name))
            
            fur_target_total = np.concatenate(fur_target_total)     
            ins_in_scale_total = np.concatenate(ins_in_scale_total)
            ins_output_total = np.concatenate(ins_output_total)
            trend_result_item = trend_result[(trend_result['date']==date)&(trend_result['p0']=='total')]
            fur_target = future_target[instruments, -self.output_chunk_length+self.cut_len-1, 0]
            # self.draw_ins_visdom(ins_in_scale_total, ins_output_total, fur_target_total, ts_arr,coll_item=coll_item,
            #             trend_result=trend_result_item, date=date, iter_num='{}_all'.format(date), key='total')                    
            # 分类趋势比较可视化
            # self.draw_section_visdom(trend_result, date)
            self.draw_cate_visdom(cate_result, date)
        # 整体趋势可视化      
        self.draw_trend_visdom(trend_result,first_date)
                 
        # df_plot = coll_result_output.groupby(['date', 'scale_idx'])[['pred_trend_value','real_trend_values','real_trend_ref_values','trend_match_flag']].mean().unstack(level=1)
        # custom_labels = df_plot['trend_match_flag'].astype(int).values.tolist()
        # del df_plot['trend_match_flag']
        # # 直接画分组柱状图（多列自动并列）
        # df_plot.plot(kind='bar', figsize=(9, 5), width=0.7)
        # # ===================== 美化 =====================
        # name = "scale_trend"
        # # plt.figure(name,figsize=(12,9))
        # plt.subplots_adjust(bottom=0.2)
        # plt.xticks(rotation=45, ha='right', fontsize=8,ticks=range(len(custom_labels)),labels=custom_labels)  # 旋转+缩小字体
        # plt.subplots_adjust(bottom=0.3)                 # 底部留足空间
        # plt.tight_layout()                             # 自动适配
        # plt.gcf().set_size_inches(16, 6)               # 拉宽图       
        # plt.title('scale trend', fontsize=14)
        # plt.xlabel('date', fontsize=12)
        # plt.ylabel('trend value', fontsize=12)
        # plt.xticks(rotation=0)  # X轴文字不旋转
        # plt.legend(title='Date-ScareType', bbox_to_anchor=(1, 1))
        # plt.grid(axis='y', alpha=0.3)
        # plt.tight_layout()
        # plt.savefig("{}/{}".format(RESULT_FILE_PATH,name), dpi=300, bbox_inches='tight')
        # plt.show()     
    
    def draw_ins_visdom(self,instruments,output=None,fur_target=None,ts_arr=None,coll_item=None,scale_item=None,
                        cate_result=None,trend_result=None,date=None,iter_num='',key=None):
        
        name_arr = []
        p0 = scale_item['p0']
        p1 = scale_item['p1']
        for inner_index, item in enumerate(ts_arr[instruments]):
            match_item = coll_item[coll_item['instrument'] == item['instrument']]
            if match_item.shape[0] > 0:
                trend = match_item['trend_value'].values[0]
                name_arr.append(item["instrument"] + "_match_" + str(trend))
            else:
                name_arr.append(item["instrument"])
        price_array_range = np.array([self.criterion.compute_diff_range_class(item)[0] for item in ts_arr[instruments]])
        price_array_range = price_array_range / 10      
        view_data = np.stack([output, fur_target, price_array_range]).transpose(1, 0)
        # view_data = np.stack([ins_output,dec_output_item,fur_round_target,price_array_range]).transpose(1,0)
        win = "detail_target_{}".format(iter_num)
        # pred_trend_value = trend_result['pred_trend_value'].values[0]
        # real_trend_values = trend_result['real_trend_values'].values[0]
        match_flag = ""
        # if np.any(trend_result['is_can']==1):
        #     trend_flag = trend_result['can_trend_flag'].values[0]
        #     trend_flag_str = "long" if trend_flag==1 else "short"
        #     match_flag = trend_flag_str + "_" + str(trend_result['can_ins_num'].values[0])
        if self.cate_mode=='cateTotal':
            cate_item = cate_result[(cate_result['p0']==p0)&(cate_result['p1']==p1)].iloc[0]
        else:
            cate_item = cate_result[(cate_result['p0']==p0)].iloc[0]
        if cate_item['top_flag']==1:
            target_title = "{}_{} long".format(date,key,match_flag)  
        elif cate_item['top_flag']==-1:
            target_title = "{}_{} short".format(date,key,match_flag)   
        else:
            target_title = "{}_{}".format(date,key)             
        viz_result_detail = global_var.get_value("viz_result_detail")
        viz_result_detail.viz_bar_compare(view_data, win=win, title=target_title, rownames=name_arr, legends=["pred_cls", "target", "price"])        

    def draw_trend_visdom(self,trend_result,date):
        """整体趋势可视化"""
        
        viz_result_ext = global_var.get_value("viz_result_ext")
        names = ['pred','target','pred_scale','target_scale']  
        xticklabels = trend_result['date'].unique().tolist()
        xtickvals = np.arange(0,len(xticklabels),5).tolist()
        xticklabels = [xticklabels[i] for i in xtickvals]
        # # 小类趋势
        # for i,row in self.scale_arr.iterrows():
        #     p0 = row['p0']
        #     p1 = row['p1']
        #     win = "batch_trend_line_{}_{}".format(date,i)
        #     title = "bt_{}/{}".format(row['p0_name'],row['p1_name'])  
        #     item = trend_result[(trend_result['p0']==p0)&(trend_result['p1']==p1)]
        #     view_data = item[['pred_trend_value','real_trend_values','pred_trend_value_scale','real_trend_values_scale']].values
        #     # x_range = item['date'].values
        #     viz_result_ext.viz_matrix_var(view_data,win=win,title=title,xtickvals=xtickvals,xticklabels=xticklabels,names=names)
        # # 大类趋势
        # names = ['pred_scale','target_scale'] 
        # for p0 in self.scale_arr['p0'].unique():
        #     p0_name = self.scale_arr[self.scale_arr['p0']==p0]['p0_name'].values[0]
        #     item = trend_result[(trend_result['p0']==p0)]
        #     win = "batch_trend_p0_{}_{}".format(date,p0)
        #     title = "bt_p0:{}".format(p0_name)              
        #     view_data = item.groupby('date')[['pred_trend_value_scale','real_trend_values_scale']].mean().values
        #     viz_result_ext.viz_matrix_var(view_data,win=win,title=title,xtickvals=xtickvals,xticklabels=xticklabels,names=names)
        # 整体趋势
        win = "batch_trend_all_{}".format(date)
        title = "bt_all:{}".format(date)   
        view_data = trend_result[(trend_result['p0']=='total')&(trend_result['p1']=='total')][['pred_trend_value','real_trend_values','pred_trend_value_scale','real_trend_values_scale']].values
        viz_result_ext.viz_matrix_var(view_data,win=win,title=title,xtickvals=xtickvals,xticklabels=xticklabels,names=names)          
        
    def draw_section_visdom(self,trend_result,date):
        """分类趋势比较可视化"""
        
        trend_result_item = trend_result[(trend_result['date']==date)]
        viz_result = global_var.get_value("viz_result")
        view_data = []
        name_arr = []
        for i,row in self.scale_arr.iterrows():
            p0 = row['p0']
            p1 = row['p1']
            if self.cate_mode=='cateTotal':
                item = trend_result_item[(trend_result_item['p0']==p0)&(trend_result_item['p1']==p1)]
            else:
                item = trend_result_item[(trend_result_item['p0']==p0)]  
            view_data.append([item['pred_trend_value_scale'].values[0],item['real_trend_values_scale'].values[0]])
            name = row['p0_name'][:2]+'/'+row['p1_name'][:2]
            if np.any(item['is_can']==1):
                trend_flag = item['can_trend_flag'].values[0]
                trend_flag_str = "L" if trend_flag==1 else "S"
                match_flag = trend_flag_str + "_" + str(trend_result['can_ins_num'].values[0])
                name = name + "/" + match_flag
            name_arr.append(name)
            
        view_data = np.array(view_data)
        win = "sec_comp_{}".format(date)
        target_title = "sec_comp__{}".format(date)  
        viz_result.viz_bar_compare(view_data, win=win, title=target_title, rownames=name_arr, legends=["pred", "target"])     

    def draw_cate_visdom(self,cate_result,date):
        """分类趋势比较可视化"""
        
        cate_result_item = cate_result[(cate_result['date']==date)]
        viz_result = global_var.get_value("viz_result")
        name_arr = []
        for i,cate_item in cate_result_item.iterrows():
            p0 = cate_item['p0']
            p1 = cate_item['p1']
            if self.cate_mode=='cateTotal':
                scale_item = self.scale_arr[(self.scale_arr['p0']==p0)&(self.scale_arr['p1']==p1)]
            else:
                scale_item = self.scale_arr[(self.scale_arr['p0']==p0)]                 
            name = "{}/{}".format(scale_item['p0_name'].values[0][:2],scale_item['p1_name'].values[0][:2])
            if cate_item['top_flag']==1:
                name = name + "/T"
            if cate_item['top_flag']==-1:
                name = name + "/V"    
            name_arr.append(name)               
        view_data = cate_result_item[['value','real_value']].values
        win = "cate_{}".format(date)
        target_title = "cate_{}".format(date)  
        viz_result.viz_bar_compare(view_data, win=win, title=target_title, rownames=name_arr, legends=["pred", "target"])    
                               
    def viz_data_board(self):
        """可视化验证集数据流"""
        
        for name,feat in self.features.items():
            if name=='total_gradients' or name=='gradient_components_ori':
                continue
            feat = feat.squeeze(-1)
            # 针对验证结果，以品种为单位计算均值，并可视化
            if len(feat.shape)==2:
                ins_feat = feat
            elif len(feat.shape)==3:
                # ins_feat = weighted_signed_score_3d(feat)
                ins_feat = feat.mean(-1)
            ins_feat = ins_feat.cpu().numpy()
            # self.logger.experiment.add_figure('{}_heatmap_{}/'.format(section,name), plot_feature_heatmap(ins_feat), global_step=self.global_step)
            range_num = 8 if ins_feat.shape[0]>8 else ins_feat.shape[0]
            fig = plot_sample_lines(ins_feat, sample_indices=range(range_num), title='Batch Sample Features')
            self.logger.experiment.add_figure('lines_{}/'.format(name), fig, global_step=self.global_step)    
            
    def viz_in_out_data(self,mode="train"): 
        """可视化中间结果与实际目标的比较情况"""

        data_flow = {'long_yield':{},'short_yield':{},'win_rate':{}}
        for name,feat in self.features.items():
            if name=='total_gradients':
                continue
            name_prefix = name.split("_")[0]
            if name_prefix!=mode:
                continue
            feat = feat.squeeze(-1)
            # 以品种为单位计算L2
            if len(feat.shape)==2:
                ins_feat = feat
            elif len(feat.shape)==3:
                ins_feat = feat.mean(-1)
            for ind_name in self.inout_compare_names:
                if (ind_name) in name and name.endswith("output"):
                    # 取得排名数值
                    price_targets = self.cur_price_targets[:,self.ins_all]
                    # 计算收益率和胜率
                    top_pred, top_pred_index = torch.topk(ins_feat, k=self.top_num, dim=-1)
                    top_pred_inverse, top_pred_inverse_index = torch.topk(ins_feat, k=self.top_num, largest=False, dim=-1)
                    long_target_data = torch.gather(price_targets, -1, top_pred_index) 
                    short_target_data = torch.gather(price_targets, -1, top_pred_inverse_index)           
                    long_yield = long_target_data.mean()
                    long_win = torch.sum(long_target_data>0)/long_target_data.flatten().shape[0]
                    short_yield = -short_target_data.mean()
                    short_win = torch.sum(short_target_data<0)/short_target_data.flatten().shape[0]
                    data_flow['long_yield'][ind_name] = long_yield
                    data_flow['short_yield'][ind_name] = short_yield
                    data_flow['win_rate'][ind_name] = (long_win+short_win)/2
                    # data_flow['short_win'][ind_name] = short_win
                    break
            
        metrics = np.zeros([len(self.inout_compare_names),3])
        for i,name in enumerate(self.inout_compare_names):
            metrics[i,0] = data_flow['long_yield'][name]
            metrics[i,1] = data_flow['short_yield'][name]
            metrics[i,2] = data_flow['win_rate'][name]
        category_labels = ['long_yield','short_yield','win_rate']
        group_labels = self.inout_compare_names
        colors = ['#CDF022', '#541AC4', '#F02241']
        if mode=="train":
            fig = plot_grouped_bar(metrics, group_labels, category_labels, ylabel='Value', title='yield in Train',colors=colors)
        else:
            fig = plot_grouped_bar(metrics, group_labels, category_labels, ylabel='Value',title='yield in Valid',colors=colors)
        self.logger.experiment.add_figure('inoutRes_{}/'.format(mode), fig, global_step=self.global_step)  
    
    def viz_dig_info(self,dig_info,mode="val"): 
        
        indus_info,cy_info,win_rate_info,yield_rate_info = dig_info
        
        colors = ['#CDF022', '#541AC4']
        category_labels = ["suc_yield","fail_yield"]
        group_labels = indus_info['industry'].values
        fig = plot_grouped_bar(indus_info[category_labels].values, group_labels, category_labels, ylabel='Number',title='industry dig',colors=colors)
        self.logger.experiment.add_figure('indusInfo_{}/'.format(mode), fig, global_step=self.global_step)  
        
        group_labels = cy_info['create_year'].values
        fig = plot_grouped_bar(cy_info[category_labels].values, group_labels, category_labels, ylabel='Number',title='createYear dig',colors=colors)
        self.logger.experiment.add_figure('cyInfo_{}/'.format(mode), fig, global_step=self.global_step)    
        
        category_labels = ["suc_cnt","fail_cnt","fail_trend_cnt"]      
        colors = ['#CDF022', '#541AC4', '#F02241']
        group_labels = win_rate_info['mode'].values
        fig = plot_grouped_bar(win_rate_info[category_labels].values.astype(int), group_labels, category_labels, ylabel='Number',title='win rate dig',colors=colors)
        self.logger.experiment.add_figure('winRate_{}/'.format(mode), fig, global_step=self.global_step)  
        
        category_labels = ["suc_yield","fail_yield","fail_trend_yield"]     
        group_labels = yield_rate_info['mode'].values
        fig = plot_grouped_bar(yield_rate_info[category_labels].values.astype(float), group_labels, category_labels, ylabel='Number',title='yield rate dig',colors=colors)
        self.logger.experiment.add_figure('yieldRate_{}/'.format(mode), fig, global_step=self.global_step)  
                        
    def dig_result_info(self,coll_results):
        """对预测评判信息进一步挖掘，后续用于可视化"""
        
        tft_dataset = global_var.get_value("dataset") 
        result_data = coll_results.merge(tft_dataset.base_info, on='instrument', how='left')
        result_data['industry'] = result_data['industry'].astype(int)
        result_data['create_year'] = result_data['create_year'].astype(int)
        result_data['trend_value'] = result_data['trend_value'].astype(int)
        
        # 分别统计成功失败的情况
        fail_result = result_data[result_data['diff_range']<0]
        suc_result = result_data[result_data['diff_range']>=0]    
        # suc_fail_info = np.array([[0,suc_result.shape[0],fail_result.shape[0]]])
        suc_fail_info = np.array([[0,suc_result['diff_range'].sum(),-fail_result['diff_range'].sum()]])
        
        # 按照行业统计收益率情况
        indus_info = pd.DataFrame(suc_fail_info,columns=['industry',"suc_yield","fail_yield"])
        suc_res = suc_result.groupby("industry")['diff_range'].apply(lambda x: x.sum()).to_frame(name='suc_yield')
        fail_res = -fail_result.groupby("industry")['diff_range'].apply(lambda x: x.sum()).to_frame(name='fail_yield')
        total_res = suc_res.merge(fail_res, on='industry', how='left').reset_index().fillna(0)
        indus_info = pd.concat([indus_info,total_res])
        indus_info['industry'] = indus_info['industry'].astype(int)    
        
        # 按照创建年份，统计收益率情况   
        cy_info = pd.DataFrame(suc_fail_info,columns=['create_year',"suc_yield","fail_yield"])
        suc_res = suc_result.groupby("create_year")['diff_range'].apply(lambda x: x.sum()).to_frame(name='suc_yield')
        fail_res = -fail_result.groupby("create_year")['diff_range'].apply(lambda x: x.sum()).to_frame(name='fail_yield')
        total_res = suc_res.merge(fail_res, on='create_year', how='left').reset_index().fillna(0)
        cy_info = pd.concat([cy_info,total_res])   
        cy_info['create_year'] = cy_info['create_year'].astype(int)    
        
        # 按照多空判断，统计收益率情况   
        long_fail = result_data[(result_data['diff_range']<0)&(result_data['trend_value']==1)]
        long_fail_withtrend = long_fail[(long_fail['real_trend_values']<0)]
        long_suc = result_data[(result_data['diff_range']>0)&(result_data['trend_value']==1)]
        short_fail = result_data[(result_data['diff_range']<0)&(result_data['trend_value']==0)]
        short_fail_withtrend = short_fail[(short_fail['real_trend_values']>0)]
        short_suc = result_data[(result_data['diff_range']>0)&(result_data['trend_value']==0)]   
        total_fail_withtrend = pd.concat([long_fail_withtrend,short_fail_withtrend])
        # Win Rate
        data = [['total',suc_result.shape[0],fail_result.shape[0],total_fail_withtrend.shape[0]],
                ['short',short_suc.shape[0],short_fail.shape[0],short_fail_withtrend.shape[0]],
                ['long',long_suc.shape[0],long_fail.shape[0],long_fail_withtrend.shape[0]],]
        win_rate_info = pd.DataFrame(np.array(data),columns=['mode',"suc_cnt","fail_cnt","fail_trend_cnt"])    
        # Yield Rate
        data = [['total',suc_result['diff_range'].sum(),-fail_result['diff_range'].sum(),-total_fail_withtrend['diff_range'].sum()],
                ['short',short_suc['diff_range'].sum(),-short_fail['diff_range'].sum(),-short_fail_withtrend['diff_range'].sum()],
                ['long',long_suc['diff_range'].sum(),-long_fail['diff_range'].sum(),-long_fail_withtrend['diff_range'].sum()]]
        yield_rate_info = pd.DataFrame(np.array(data),columns=['mode',"suc_yield","fail_yield","fail_trend_yield"])            
        return (indus_info,cy_info,win_rate_info,yield_rate_info)        

    def dump_loss_data(self, loss_detail,batch_size=32,real_batch_size=32):
        """计算当前epoch的验证损失均值"""
        
        for key in loss_detail.keys():
            item = loss_detail[key].item()
            if key in self.loss_result:
                self.loss_result[key] = np.concatenate([self.loss_result[key],np.array([item])])
            else:
                self.loss_result[key] = np.array([item])
        if 'batch_size' in self.loss_result:
            self.loss_result['batch_size'] = np.concatenate([self.loss_result['batch_size'],np.array([batch_size])])
        else:
            self.loss_result['batch_size'] = np.array([batch_size])
        if 'real_batch_size' in self.loss_result:
            self.loss_result['real_batch_size'] = np.concatenate([self.loss_result['real_batch_size'],np.array([real_batch_size])])
        else:
            self.loss_result['real_batch_size'] = np.array([real_batch_size])
                                                                                                               
    def dump_val_data(self, val_batch, outputs, batch_data):
    
        output, vr_class, price_outputs, past_future_round_targets = outputs
        # choice_out, trend_value, combine_index = vr_class
        (past_target, past_covariates, historic_future_covariates, future_covariates,
            static_covariates, past_future_covariates, future_target, target_class, price_targets, _, index_round_targets, future_week_info, target_info) = val_batch
        # 记录批次内价格涨跌幅，用于整体指数批次归一化数据的回溯
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        #
        # for index,ts in enumerate(target_info):
        #     ts[main_index]["price_round_data"] = price_round_data
        #     ts[main_index]["price_round_index"] = index
        #     ts[main_index]["target_round_data"] = index_round_targets.cpu().numpy()[:,-1,-1,-1]
        #     ts[main_index]["pred_round_data"] = output[-1][2].cpu().numpy().squeeze(-1)
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output, batch_data, past_target.cpu().numpy(),
                      future_target.cpu().numpy(), target_class.cpu().numpy(),
                      price_targets.cpu().numpy(), past_future_round_targets.cpu().numpy(), whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(), future_week_info.cpu().numpy(), target_info)
        self.output_result.append(output_res)

    def combine_output_total(self, output_result):
        """重载父类方法，以适应整合数据"""
        
        target_class_total = []
        target_info_total = []
        past_target_total = []   
        future_target_total = []  
        price_targets_total = []    
        past_future_round_targets_total = []
        whole_index_round_targets_total = []
        x_bar_total = []
        sv_total = [[] for _ in range(len(self.past_split))]
        cls_total = {}
        comm_index_total = []
        trend_list_main_total = []
        trend_logits_total = {}
        trend_total = []
        index_round_targets_total = []
        future_week_info_total = []
        for item in output_result:
            (output, batch_data, past_target, future_target, target_class, price_targets, past_future_round_targets, whole_index_round_targets, index_round_targets, future_week_info, target_info) = item
            x_bar_inner = []
            dec_inner = []
            # 合并列表中的品种和整体趋势
            if self.opt_size>1:
                _,sv_indus, comm_index = output[0]
                trend_logits,_,_  = output[1]
            else:
                trend_logits,sv_indus, comm_index,trend_list_main,_ = output[0]
            sv_indus = sv_indus[0]
            comm_index = comm_index[0]
            trend_logits = trend_logits[0]       
            trend_list_main = trend_list_main[0]
            
            for key in sv_indus.keys():
                cur_data = sv_indus[key].cpu().numpy()
                if key not in cls_total:
                    cls_total[key] = cur_data
                else:
                    cls_total[key] = np.concatenate([cls_total[key],cur_data],0)
            comm_index_total.append(comm_index.cpu().numpy())
            for key in trend_logits.keys():
                if key not in trend_logits_total:
                    trend_logits_total[key] = {}
                for inner_key in trend_logits[key]:
                    cur_data = trend_logits[key][inner_key].cpu().numpy()
                    if inner_key not in trend_logits_total[key]:
                        trend_logits_total[key][inner_key] = cur_data
                    else:
                        trend_logits_total[key][inner_key] = np.concatenate([trend_logits_total[key][inner_key],cur_data],0)      
                
            # ce_index_inner = np.stack(ce_index_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
            trend_total.append(batch_data)
            
            trend_list_main_total.append(trend_list_main.cpu().numpy())  
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            price_targets_total.append(price_targets)
            past_future_round_targets_total.append(past_future_round_targets)
            whole_index_round_targets_total.append(whole_index_round_targets)
            index_round_targets_total.append(index_round_targets)
            future_week_info_total.append(future_week_info)
        
        x_bar_total = np.concatenate(x_bar_total)
        trend_total = np.concatenate(trend_total)
        comm_index_total = np.concatenate(comm_index_total)
        trend_list_main_total = np.concatenate(trend_list_main_total)
        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        price_targets_total = np.concatenate(price_targets_total)
        past_future_round_targets_total = np.concatenate(past_future_round_targets_total)
        whole_index_round_targets_total = np.concatenate(whole_index_round_targets_total)
        index_round_targets_total = np.concatenate(index_round_targets_total)
        target_info_total = np.concatenate(target_info_total)
        future_week_info_total = np.concatenate(future_week_info_total)
                    
        return (trend_list_main_total, sv_total, cls_total, comm_index_total, trend_logits_total, trend_total), \
                    past_target_total, future_target_total, target_class_total, price_targets_total, past_future_round_targets_total, \
                    future_week_info_total, index_round_targets_total, target_info_total        
                           
    def combine_result_data(self, output_result=None, predict_mode=False, pred_top_num=2):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # return None,None,None,None
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d, past_target_3d, future_target_3d, target_class_3d, price_targets_3d, batch_trend_data, future_week_info_3d, \
            index_round_targets_3d, target_info_3d = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d == 3)[0].shape[0]
        rate_total = []
        result_date_list = None
        
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        industry_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        # 按照时间索引暂存预测数据，用于全局化共享使用
        glo_match_data = []
        ref_output_index = 1
        target_len = self.cut_len-1

        pred_data_path = os.path.join(RESULT_FILE_PATH, self.pred_index_data_path)
        # 根据配置，从指数预测结果中加载数据
        if len(self.pred_index_data_path)>3 and self.load_index_data:
            trend_result = pd.read_csv(pred_data_path)
        else:
            trend_result = None
        trend_target = self.scale_branch_trend_target(target_info_3d)
        pred_cate_data_path = os.path.join(RESULT_FILE_PATH, self.pred_cate_data_path)
        if len(self.pred_cate_data_path)>3 and self.load_cate_data:
            cate_result = pd.read_csv(pred_cate_data_path)
        else:
            cate_result = None
                    
        for i in range(target_class_3d.shape[0]):
            past_target = past_target_3d[i]
            target_class_list = target_class_3d[i]
            target_info_list = target_info_3d[i]
            keep_index = np.where(target_class_list >= 0)[0]
            keep_index = np.intersect1d(keep_index, ins_all)              
            date = target_info_list[0]['future_start_datetime']
        
        result_total_list = None
        trend_result_total = None
        cate_result_total = None
        
        # 遍历按日期进行评估
        for i in range(target_class_3d.shape[0]):
            future_target = future_target_3d[i]
            past_target = past_target_3d[i]
            whole_target = np.concatenate([past_target, future_target], axis=1)
            target_info_list = target_info_3d[i]
            target_class_list = target_class_3d[i]
            # 有一些空值，找出对应索引后续做忽略处理
            keep_index = np.where(target_class_list >= 0)[0]
            # 去除指数整体及行业
            keep_index = np.intersect1d(keep_index, ins_all)  
            trend_list_main = output_3d[0]
            trend_data = output_3d[3]
            features = output_3d[2]
            trend_logits = output_3d[4]
            # trend_logits_item = {key:trend_logits[key][i] for key cate_compare_resultin trend_logits}
            output_list = [features, trend_data,trend_logits]
            price_target_list = price_targets_3d[i]
            date = int(target_info_list[np.where(target_class_list >= 0)[0][0]]["future_start_datetime"])
            index_round_targets = index_round_targets_3d[i]
            if not (date >= STAT_DATE[0] and date <= STAT_DATE[1]):
                continue              
            (features, cate_data,trend_logits_item) = output_list
            # 类别比较结果
            if cate_result is None:
                if self.cate_mode=='cateMain':
                    cate_compare_result = self.compute_cateMain_compare(trend_list_main,batch_no=i,date=date)   
                else:
                    cate_compare_result = self.compute_cate_compare(cate_data,batch_no=i,date=date)  
            else:
                cate_compare_result =  cate_result[cate_result['date']==date]               
            if trend_result is None:
                trend_result_item = self.compute_branch_trend(trend_logits_item, date=date,batch_no=i)
                # 对候选类别进行筛选
                # trend_result_item = self.sumup_trend(trend_result_item)   
            else:
                trend_result_item = trend_result[trend_result['date']==date]
            if trend_result_item.shape[0]==0:
                continue     
            # 对候选类别进行筛选            
            if self.cate_mode=='cateMain':
                trend_result_item = self.sumup_trend_with_cateMain(trend_result_item,cate_compare_result)   
            else:
                trend_result_item = self.sumup_trend_with_cate(trend_result_item,cate_compare_result)  
            # 生成目标索引
            result_list = self.build_import_index(output_data=output_list, trend_result=trend_result_item,
                                                  cate_result=cate_compare_result,pred_top_num=pred_top_num,date=date,batch_no=i)
            import_index = result_list['top_index'].values
            # 使用有效数据（当日有交易的品种）
            if import_index is not None and import_index.shape[0] > 0:
                import_index = np.intersect1d(keep_index, import_index)  
                result_list = result_list[result_list['top_index'].isin(import_index)]
              
            # 如果是预测模式，则只输出结果,不验证
            if predict_mode:
                result_date_list = result_list
                coll_results = []
                for index, row in result_list.iterrows():
                    imp_idx = row['top_index']
                    overroll_trend = row['top_flag']
                    ts = target_info_list[imp_idx]
                    coll_results.append([imp_idx, ts["instrument"], overroll_trend])                    
                continue

            # 验证趋势预测准确度
            trend_result_item = self.eva_total_trend(trend_result_item,trend_target,batch_no=i)
            # 对分类比较预测结果进行评估
            cate_compare_result = self.eva_cate_compare(cate_compare_result,target_info_list,batch_no=i)
            # 验证品种预测准确性
            coll_results = self.collect_result_compindex(date=date, target_info=target_info_list, 
                                    result_list=result_list,trend_target=trend_target, keep_index=keep_index)  
  
            # 把趋势数据同步到品种选择结果中
            coll_results['trend_eva_diff'] = np.abs(coll_results['pred_trend_value'].values - coll_results['real_trend_values'].values)
            coll_results_new = []
            for i,row in coll_results.iterrows():
                p0 = row['p0']
                p1 = row['p1']
                if self.cate_mode=='cateTotal':
                    row_data = trend_result_item[(trend_result_item['p0']==p0)&(trend_result_item['p1']==p1)][['real_trend_flag','trend_match_flag']].values   
                else:
                    row_data = trend_result_item[(trend_result_item['p0']==p0)][['real_trend_flag','trend_match_flag']].values   
                row['real_trend_flag'] = row_data[0][0]
                row['trend_match_flag'] = row_data[0][1]    
                coll_results_new.append(row)
                              
            coll_results = pd.DataFrame(coll_results_new) 
            coll_results["trend_match_rate"] = coll_results.groupby(['p0','p1'])["trend_match_flag"].transform("mean")
            # 把结果数据整合到预测记录中
            if result_total_list is None:
                result_total_list = coll_results
                trend_result_total = trend_result_item
                cate_result_total = cate_compare_result
            else:
                result_total_list = pd.concat([result_total_list, coll_results])      
                trend_result_total = pd.concat([trend_result_total, trend_result_item])     
                cate_result_total = pd.concat([cate_result_total, cate_compare_result])  
         
        # Save Trend Result           
        if len(self.pred_index_data_path)>3 and not self.load_index_data:    
            trend_result_total.to_csv(pred_data_path, index=False)        
        if len(self.pred_cate_data_path)>3 and not self.load_cate_data:    
            cate_result_total.to_csv(pred_cate_data_path, index=False)    
                        
        if predict_mode:
            return result_date_list      
        
        trend_total = trend_result_total[trend_result_total['p0']=='total']
        trend_total['trend_eva_diff'] = np.abs(trend_total['real_trend_values'] - trend_total['pred_trend_value'])
        rate_columns = ["total_cnt", "yield_rate", "win_rate","trend_eva_diff","trend_match_rate","trend_recall","cate_corr_rate","cate_yield"]    
        ls_num = np.sum(trend_total['real_trend_flag']!=0)
        match_ls_num = np.sum((trend_total['real_trend_flag']!=0)&(trend_total['trend_match_flag']))
        recall_rate = match_ls_num/ls_num
        trend_match_rate = np.sum(trend_total['trend_match_flag'])/trend_total.shape[0]
        cate_top_res = cate_result_total[cate_result_total['top_flag']==1]
        cate_top_inverse_res = cate_result_total[cate_result_total['top_flag']==-1]
        cate_corr_rate = (np.sum(cate_top_res['real_value']>0)/cate_top_res.shape[0] + np.sum(cate_top_inverse_res['real_value']<0)/cate_top_inverse_res.shape[0])/2
        cate_yield = (np.sum(cate_top_res['real_value'])/cate_top_res.shape[0] + np.sum(-cate_top_inverse_res['real_value'])/cate_top_inverse_res.shape[0])
        
        rate_total = [result_total_list.shape[0],
                      round(result_total_list['diff_range'].sum(), 3),
                      round(np.sum(result_total_list['diff_range'] > 0) / result_total_list.shape[0], 3),
                      trend_total['trend_eva_diff'].mean(),
                      trend_match_rate,recall_rate,
                      cate_corr_rate,cate_yield
                      ]
        rate_total = pd.DataFrame(np.array([rate_total]), columns=rate_columns)
        for i in range(4):
            distribute = round(np.sum(result_total_list['target_class'] == i) / result_total_list.shape[0], 3)
            rate_total['dist_{}'.format(i)] = round(distribute, 3)
            
        if rate_total.shape[0] == 0:
            return None, None,None, None
        
        return rate_total, result_total_list,trend_result_total,cate_result_total
    
    
    def build_import_index(self, date=None, pred_top_num=2, output_data=None, trend_result=None, cate_result=None,batch_no=0): 
        """生成涨幅达标的预测数据下标"""
        
        (features, trend_data,trend_logits_item) = output_data

        import_index_list = self.strategy_top_bidi(features, trend_logits_item,pred_top_num=pred_top_num, trend_result=trend_result,
                                            cate_result=cate_result,batch_no=batch_no,date=date)
        # import_index_list = self.strategy_top_bidi(features, trend_logits_item,pred_top_num=pred_top_num, trend_result=trend_result,
        #                                     cate_result=cate_result,batch_no=batch_no,date=date)
 
        # 构建结果集
        result_list = import_index_list
        result_list['date'] = date     
        result_list = result_list.astype({'top_index':int,'top_flag':int,'date':int,'pred_trend_value':float})
        
        return result_list

    def compute_branch_trend(self,trend_data,date=None,batch_no=0):
        
        result = []
        columns = ['p0','p1','date','pred_trend_value','pred_trend_value_scale','pred_trend_flag']
        scale_arr = emb_scale_arr(self.scale_arr)
        
        def proc_item(key,inner_key):
            trend_batch = trend_data[key][inner_key]
            trend_batch_norm = StandardScaler().fit_transform(np.expand_dims(trend_batch,-1)).squeeze(-1)
            # pred_trend_value = trend_batch_norm[batch_no]
            trend_item = trend_batch[batch_no]
            pred_trend_value = scale_value(trend_item,trend_batch.min(),trend_batch.max(),self.trend_threhold['min'],self.trend_threhold['max'])
            pred_trend_flag = self.get_trend_flag_from_value(pred_trend_value)
            result.append([key,inner_key,date,trend_item,pred_trend_value,pred_trend_flag])            
            
        for key in scale_arr: 
            for inner_key in scale_arr[key]:
                proc_item(key,inner_key)
        # 汇总所有类别数据
        result = pd.DataFrame(result,columns=columns)
        total_pred_value = result[['pred_trend_value','pred_trend_value_scale']].mean().values
        pred_trend_flag = self.get_trend_flag_from_value(total_pred_value[1])
        total_row = {'p0':'total','p1':'total','date':result['date'].values[0],
                     'pred_trend_value':total_pred_value[0],'pred_trend_value_scale':total_pred_value[1],'pred_trend_flag':pred_trend_flag}
        result = pd.concat([result,pd.DataFrame(total_row, index=[0])])
        
        return result
            
    def strategy_top_bidi(self, features, trend_data,pred_top_num=2,trend_result=None,cate_result=None,batch_no=0,date=None):
        """筛选品种明细,使用双向模式"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        top_num = pred_top_num
        cancidate_list = []
        mode = 'single'
        
        # 同时从正反2个方向选取品种
        trend_result_item = trend_result[(trend_result['date']==date)]
        trend_result_can = trend_result_item[(trend_result_item['is_can']==1)]
        if self.cate_mode=='cateTotal':
            cancidate_list = self.compute_arg_sort_by_trend(features,trend_result=trend_result_can,cate_result=cate_result,batch_no=batch_no,date=date)
        else:   
            cancidate_list = self.compute_arg_sort_by_main_trend(features,trend_result=trend_result,batch_no=batch_no,date=date)     
        
        return cancidate_list
    
    def sumup_trend_with_cate(self,trend_result,cate_result):
        """参考类别比较结果，对类别趋势结果进行筛选，获取类别候选"""
        
        trend_result['is_can'] = 0
        trend_result['can_trend_flag'] = 0
        trend_result['can_ins_num'] = 0
        total_trend_flag = trend_result[trend_result['p0']=='total']['pred_trend_flag'].values[0]
        trend_result = trend_result.sort_values(by='pred_trend_value_scale').reset_index()
        cate_result = cate_result.sort_values(by='value').reset_index()
        long_cate_top = cate_result.iloc[-1]
        long_cate_top_dict = {'p0':long_cate_top['p0'],'p1':long_cate_top['p1']}
        short_cate_top = cate_result.iloc[0]
        short_cate_top_dict = {'p0':short_cate_top['p0'],'p1':short_cate_top['p1']}
        long_condi = (trend_result['p0']==long_cate_top_dict['p0'])&(trend_result['p1']==long_cate_top_dict['p1'])
        short_condi = (trend_result['p0']==short_cate_top_dict['p0'])&(trend_result['p1']==short_cate_top_dict['p1'])        
        # 31模式：如果总体趋势为多方，则取得1个多方小类（内部品种3）。空方则反过来。趋势为平，则类别1+1，品种2+2
        if total_trend_flag==1:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 3
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 1            
        elif total_trend_flag==-1:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 1
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 3    
        else:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 2
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 2        
        
        trend_result = trend_result.astype({'is_can':int,'can_trend_flag':int,'can_ins_num':int})  
        return trend_result

    def sumup_trend_with_cateMain(self,trend_result,cate_result):
        """参考类别比较结果，对类别趋势结果进行筛选，获取类别候选"""
        
        trend_result = trend_result.groupby('p0').first().reset_index()
        trend_result['is_can'] = 0
        trend_result['can_trend_flag'] = 0
        trend_result['can_ins_num'] = 0
        total_trend_flag = trend_result[trend_result['p0']=='total']['pred_trend_flag'].values[0]
        total_trend_flag = 0
        trend_result = trend_result.sort_values(by='pred_trend_value_scale').reset_index()
        cate_result = cate_result.sort_values(by='value').reset_index()
        long_cate_top = cate_result.iloc[-1]
        long_cate_top_dict = {'p0':long_cate_top['p0']}
        short_cate_top = cate_result.iloc[0]
        short_cate_top_dict = {'p0':short_cate_top['p0']}
        long_condi = (trend_result['p0']==long_cate_top_dict['p0'])
        short_condi = (trend_result['p0']==short_cate_top_dict['p0'])    
        # 31模式：如果总体趋势为多方，则取得1个多方大类（内部品种3个,空方大类内部1个。空方则反过来。趋势为平，则品种2+2
        if total_trend_flag==1:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 3
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 1            
        elif total_trend_flag==-1:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 1
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 3    
        else:
            trend_result.loc[long_condi,'is_can'] = 1
            trend_result.loc[long_condi,'can_trend_flag'] = 1
            trend_result.loc[long_condi,'can_ins_num'] = 2
            trend_result.loc[short_condi,'is_can'] = 1
            trend_result.loc[short_condi,'can_trend_flag'] = 0
            trend_result.loc[short_condi,'can_ins_num'] = 2        
        
        trend_result = trend_result.astype({'is_can':int,'can_trend_flag':int,'can_ins_num':int})  
        return trend_result        
       
    def sumup_trend(self,trend_result):
        """对类别趋势结果进行筛选，获取类别候选"""
        
        trend_result['is_can'] = 0
        trend_result['can_trend_flag'] = 0
        trend_result['can_ins_num'] = 0
        total_trend_flag = trend_result[trend_result['p0']=='total']['pred_trend_flag'].values[0]
        trend_result = trend_result[~(trend_result['p0']=='total')]
        trend_result = trend_result.sort_values(by='pred_trend_value_scale').reset_index()
        # 211模式：如果总体趋势为多方，则取得2个多方小类（内部品种2+1），1个空方小类（内部品种1）。空方则反过来。趋势为平，则类别2+2，品种1+1+1+1
        if total_trend_flag==1:
            trend_result.loc[trend_result.index[-2:],'is_can'] = 1
            trend_result.loc[trend_result.index[-2:],'can_trend_flag'] = 1
            trend_result.loc[trend_result.index[-1],'can_ins_num'] = 2
            trend_result.loc[trend_result.index[-2],'can_ins_num'] = 1
            trend_result.loc[trend_result.index[0],'is_can'] = 1
            trend_result.loc[trend_result.index[0],'can_trend_flag'] = 0
            trend_result.loc[trend_result.index[0],'can_ins_num'] = 1
        elif total_trend_flag==-1:
            trend_result.loc[trend_result.index[:2],'is_can'] = 1
            trend_result.loc[trend_result.index[:2],'can_trend_flag'] = 0
            trend_result.loc[trend_result.index[0],'can_ins_num'] = 2
            trend_result.loc[trend_result.index[1],'can_ins_num'] = 1
            trend_result.loc[trend_result.index[-1],'is_can'] = 1        
            trend_result.loc[trend_result.index[-1],'can_trend_flag'] = 1  
            trend_result.loc[trend_result.index[-1],'can_ins_num'] = 1   
        else:
            trend_result.loc[trend_result.index[0],'is_can'] = 1
            trend_result.loc[trend_result.index[0],'can_trend_flag'] = 0
            trend_result.loc[trend_result.index[0],'can_ins_num'] = 2
            trend_result.loc[trend_result.index[-1],'is_can'] = 1   
            trend_result.loc[trend_result.index[-1],'can_trend_flag'] = 1
            trend_result.loc[trend_result.index[-1],'can_ins_num'] = 2         
                
        
        trend_result = trend_result.astype({'is_can':int,'can_trend_flag':int,'can_ins_num':int})  
        return trend_result
    
    def compute_cate_compare(self,cate_data,batch_no=0,date=None):
        
        result = []
        cate_data_item = cate_data[batch_no]
        top_idx = cate_data_item.argmax()
        top_inverse_idx = cate_data_item.argmin()
        columns = ['date','p0','p1','p0_name','p1_name','top_flag','value']
        for i,item in self.scale_arr.iterrows(): 
            p0 = item['p0']
            p0_name = item['p0_name']
            p1 = item['p1']
            p1_name = item['p1_name']
            if i==top_idx:
                top_flag = 1
            elif i==top_inverse_idx:
                top_flag = -1
            else:
                top_flag = 0
            result.append([date,p0,p1,p0_name,p1_name,top_flag,cate_data_item[i]])
        result = pd.DataFrame(result,columns=columns)
        result['date'] = result['date'].astype(int)
        result['top_flag'] = result['top_flag'].astype(int)
        result['value'] = result['value'].astype(float)
        
        return result
    
    def compute_cateMain_compare(self,cate_data,batch_no=0,date=None):
        
        result = []
        cate_data_item = cate_data[batch_no]
        top_idx = cate_data_item.argmax()
        top_inverse_idx = cate_data_item.argmin()
        columns = ['date','p0','p1','p0_name','p1_name','top_flag','value']
        scale_arr = emb_scale_arr(self.scale_arr)
        for i,key in enumerate(scale_arr.keys()):
            item = scale_arr[key] 
            p0 = key
            p0_name = self.scale_arr[self.scale_arr['p0_code']==key]['p0_name'].values[0]
            p1 = 'any'
            p1_name = 'any'
            if i==top_idx:
                top_flag = 1
            elif i==top_inverse_idx:
                top_flag = -1
            else:
                top_flag = 0
            result.append([date,p0,p1,p0_name,p1_name,top_flag,cate_data_item[i]])
        result = pd.DataFrame(result,columns=columns)
        result['date'] = result['date'].astype(int)
        result['top_flag'] = result['top_flag'].astype(int)
        result['value'] = result['value'].astype(float)
        
        return result
                        
    def compute_arg_sort_by_trend(self, features,cate_result=None,trend_result=None,date=None,batch_no=0):
        """根据输出进行排序,参照整体及分支输出趋势"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        
        pre_index_total = []
        
        scale_arr = emb_scale_arr(self.scale_arr)
        # 根据趋势增减top数量
        for i,row in trend_result.iterrows():
            p0 = row['p0']
            p1 = row['p1']
            can_trend_flag = row['can_trend_flag']
            can_ins_num = row['can_ins_num']     
            pred_trend_value = row['pred_trend_value'] 
            pred_trend_value_scale = row['pred_trend_value_scale']
            rel_ins = scale_arr[p0][p1]['rel_ins']
            features_item = features[p0][batch_no][rel_ins]
            if can_trend_flag==1:
                pre_index = np.argsort(-features_item)[:can_ins_num]
            else:
                pre_index = np.argsort(features_item)[:can_ins_num]            
            for index in pre_index:
                real_index = scale_arr[p0][p1]['instruments'][index]
                pre_index_total.append([real_index,can_trend_flag,pred_trend_value,pred_trend_value_scale,can_trend_flag,p0,p1])
        pre_index_total = np.array(pre_index_total)
        pre_index_total = pd.DataFrame(pre_index_total,columns=['top_index','top_flag','pred_trend_value','pred_trend_value_scale','pred_trend_flag','p0','p1'])
        pre_index_total['top_index'] = pre_index_total['top_index'].astype(int)
        pre_index_total['top_flag'] = pre_index_total['top_flag'].astype(int)
        
        return pre_index_total
        
    def compute_arg_sort_by_main_trend(self, features,trend_result=None,date=None,batch_no=0,top_num=2):
        """根据输出进行排序,参照大类别输出趋势"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        
        pre_index_total = []
        
        scale_arr = emb_scale_arr(self.scale_arr)
        # 根据趋势增减top数量
        for i,row in trend_result.iterrows():
            p0 = row['p0']
            can_trend_flag = row['can_trend_flag']
            can_ins_num = row['can_ins_num']     
            pred_trend_value = row['pred_trend_value'] 
            pred_trend_value_scale = row['pred_trend_value_scale']
            if p0=='total':
                continue
            rel_ins = get_scale_cate_ins(self.scale_arr,p0)
            features_item = features[p0][batch_no]
            if can_trend_flag==1:
                pre_index = np.argsort(-features_item)[:can_ins_num]
            else:
                pre_index = np.argsort(features_item)[:can_ins_num]            
            for index in pre_index:
                real_index = rel_ins[index]
                p1 = 'any'
                pre_index_total.append([real_index,can_trend_flag,pred_trend_value,pred_trend_value_scale,can_trend_flag,p0,p1])
        pre_index_total = np.array(pre_index_total)
        pre_index_total = pd.DataFrame(pre_index_total,columns=['top_index','top_flag','pred_trend_value','pred_trend_value_scale','pred_trend_flag','p0','p1'])
        pre_index_total['top_index'] = pre_index_total['top_index'].astype(int)
        pre_index_total['top_flag'] = pre_index_total['top_flag'].astype(int)
        
        return pre_index_total
    
    def scale_branch_trend_target(self,target_info_list):
        
        scale_data = []
        total_mean_data = []
        for i,row in self.scale_arr.iterrows():
            p0 = row['p0']
            p1 = row['p1']
            ins = row['instruments']
            scale_data_item = []
            for target_info_arr in target_info_list:
                data_mean = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info_arr)[ins]]).mean()
                scale_data_item.append(data_mean)
            scale_data_item = np.array(scale_data_item)
            scale_data.append([p0,p1,scale_data_item])
        for target_info_arr in target_info_list:
            total_mean = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info_arr)]).mean()
            total_mean_data.append(total_mean)            
        scale_data.append(['total','total',np.array(total_mean_data)])
        scale_data = pd.DataFrame(scale_data,columns=['p0','p1','value'])
        
        return scale_data
    
    def collect_result_compindex(self, date=None, target_info=None, result_list=None,trend_target=None, keep_index=None):
 
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        open_diff = np.array([t['open_diff'] if t is not None else 0 for t in target_info])
        
        # dataset = global_var.get_value("dataset") 
        # scale_dict = dataset.scale_dict
        coll_results = []
        scale_arr = emb_scale_arr(self.scale_arr)
        # 对于预测数据，生成对应涨跌幅类别
        for row in result_list.itertuples():
            imp_idx = row.top_index
            overroll_trend = row.top_flag
            p1 = row.p1
            p0 = row.p0
            if self.cate_mode=='cateTotal':
                ins = scale_arr[p0][p1]['instruments']
            else:
                ins = get_scale_cate_ins(self.scale_arr,p0)
            ins = np.intersect1d(ins,keep_index)
            # real_trend_values = np.sum(open_diff[ins]>0)/ins.shape[0]
            real_trend_values = open_diff[ins].mean()
            # Do Target Scale
            # trend_target_item = trend_target[(trend_target['p0']==p0)&(trend_target['p1']==p1)]['value'].values[0]
            # real_trend_values_scale = scale_value(real_trend_values,trend_target_item.min(),trend_target_item.max(),
            #                     self.trend_threhold['min'],self.trend_threhold['max'])
            real_trend_values_scale = 0.0
            ts = target_info[imp_idx]
            diff_range, p_taraget_class, _ = self.criterion.compute_diff_range_class(ts)
            # 根据多空判断取得实际对应的类别
            if overroll_trend == 0:
                diff_range_with_trend = -diff_range
                p_taraget_class = np.array([3, 2, 1, 0])[p_taraget_class]
            else:
                diff_range_with_trend = diff_range
            coll_results.append([imp_idx,p0, p1,ts["instrument"], diff_range_with_trend, p_taraget_class, overroll_trend,real_trend_values,real_trend_values_scale])    
        
        coll_results = np.array(coll_results)
        coll_results = pd.DataFrame(coll_results, columns=['top_index', 'p0', 'p1','instrument',
                                     'diff_range', 'target_class', 'trend_value','real_trend_values','real_trend_values_scale'])
        coll_results['diff_range'] = coll_results['diff_range'].astype(float)
        coll_results['target_class'] = coll_results['target_class'].astype(int)
        coll_results['real_trend_values'] = coll_results['real_trend_values'].astype(float)
        coll_results['real_trend_values_scale'] = coll_results['real_trend_values_scale'].astype(float)
        coll_results['date'] = date
        coll_results['pred_trend_value'] = result_list['pred_trend_value'].values
        coll_results['pred_trend_value_scale'] = result_list['pred_trend_value_scale'].values.astype(float)
        coll_results['pred_trend_flag'] = result_list['pred_trend_flag'].values.astype(int)
        
        return coll_results          
    
    def get_trend_flag_from_value(self,trend_value):
        
        long_num,short_num = self.criterion.judge_topNum_from_trend(trend_value,trend_threhold=self.trend_threhold)
        if long_num>short_num:
            trend_flag = 1
        elif long_num==short_num:
            trend_flag = 0   
        else:
            trend_flag = -1   
                   
        # trend_flag = trend_value -1 
                   
        return trend_flag
    
    def eva_cate_compare(self,cate_result, target_info_list,batch_no=0):
        """对分类比较预测结果进行评估"""
        
        cate_result_new = []
        index = 0
        for _,row in cate_result.iterrows():
            p0 = row['p0']
            p1 = row['p1']             
            if self.cate_mode=='cateTotal':
                scale_item = self.scale_arr[(self.scale_arr['p0']==p0)&(self.scale_arr['p1']==p1)]
                ins = scale_item['instruments'].values[0]
            else:
                ins = get_scale_cate_ins(self.scale_arr,p0)
            cate_mean = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info_list)[ins]]).mean()
            row['real_value'] = cate_mean
            cate_result_new.append(row)
            index += 1
        cate_result_new = pd.DataFrame(cate_result_new)
        cate_result_new['cate_item_yield'] = cate_result_new['real_value'] * cate_result_new['top_flag']
        # max_index = cate_result_new['value'].idxmax()
        # min_index = cate_result_new['value'].idxmin()
        # cate_result_new['top_flag'] = 0
        # cate_result_new.loc[max_index,'top_flag'] = 1
        # cate_result_new.loc[min_index,'top_flag'] = -1
        
        return cate_result_new
        
    def eva_total_trend(self,trend_result, trend_target,batch_no=0):
        """对整体趋势预测结果进行评估"""

        scale_arr = self.scale_arr
        trend_result_new = []
        for i,row in trend_result.iterrows():
            p0 = row['p0']
            p1 = row['p1']       
            trend_target_item = trend_target[(trend_target['p0']==p0)&(trend_target['p1']==p1)]['value'].values[0]
            trend_target_norm = StandardScaler().fit_transform(np.expand_dims(trend_target_item,-1)).squeeze(-1)
            # real_trend_values_scale = trend_target_norm[batch_no]
            # real_trend_values = np.sum(open_diff[ins]>0trend_result_new = pd.DataFrame(trend_result_new))/ins.shape[0]
            real_trend_values = trend_target_item[batch_no]
            row['real_trend_values'] = real_trend_values
            # Do Target Scale
            real_trend_values_scale = scale_value(real_trend_values,trend_target_item.min(),trend_target_item.max(),
                                self.trend_threhold['min'],self.trend_threhold['max'])
            row['real_trend_values_scale'] = real_trend_values_scale
            real_trend_flag = self.get_trend_flag_from_value(real_trend_values_scale)
            row['real_trend_flag'] = real_trend_flag
            row['trend_match_flag'] = (row['pred_trend_flag']==row['real_trend_flag'])
            trend_result_new.append(row)
        trend_result_new = pd.DataFrame(trend_result_new)
        return trend_result_new
    
    def compute_feature_target_trend_corr(self, main_index_feature, main_targets):
        """计算特征数据距离与实际目标值距离的相关性"""
        
        # 取得最后一条指数特征值，并分别与前值进行距离匹配，取得最相近的几条记录，并取这几条记录对应的实际价格的平均值
        distances = self.compute_feature_distance(main_index_feature)
        dis_target = main_targets[-1] - main_targets[:-1]
        corr_dis = self.criterion.ccc_distance(distances, dis_target.cpu())
        return corr_dis
    
    def compute_feature_distance(self, main_index_feature):
        
        output_last = main_index_feature[-1:]
        output_other = main_index_feature[:-1]
        distances = pairwise_compare(torch.Tensor(output_last), torch.Tensor(output_other), distance_func=self.criterion.mse_dis).squeeze(0)
        return distances
            
            
    def on_test_start(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print('test_step')
        
        
    ##############################  Predict Part ################################

    def on_predict_start(self): 
        if self.train_step_mode == 2:
            # 第二阶段，首先加载预存结果
            with open(self.inter_rs_filepath, "rb") as fin:
                self.result_data = pickle.load(fin)   
                
    def on_predict_epoch_start(self): 
        self.output_result = []
        
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int]=None
    ):
        """预测流程，生成输出结果"""
 
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            future_target,
            target_class,
            price_targets,
            past_future_round_targets,
            index_round_targets,
            future_week_info,
            target_info
        ) = batch
               
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class,target_info)     
        input_batch = self._process_input_batch(inp)
        (output, vr_class, vr_class_list) = self(input_batch, optimizer_idx=-1)
        choice_out, trend_value, combine_index = vr_class
        
        # 只获取整体指数的价格数据
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output, choice_out.cpu().numpy(), trend_value.cpu().numpy(), combine_index.cpu().numpy(), past_target.cpu().numpy(),
                      future_target.cpu().numpy(), target_class.cpu().numpy(),
                      price_targets.cpu().numpy(), past_future_round_targets.cpu().numpy(), whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(), future_week_info.cpu().numpy(), target_info)
        self.output_result.append(output_res)        
         
    def on_predict_epoch_end(self, args): 
        """汇总预测数据，生成实际业务预测结果"""
        
        sw_ins_mappings = self.valid_sw_ins_mappings
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        result_date_list = self.combine_result_data(self.output_result, predict_mode=True, pred_top_num=self.pred_top_num)  
        result_target = {}  
        # 根据原始数组，生成实际品种信息
        if result_date_list is None:
            self.result_target = None
            return                
        dates = result_date_list['date'].unique()
        for date in dates:
            res_arr = result_date_list[result_date_list['date'] == date].sort_values(by=['top_index'])
            res_index = res_arr['top_index']
            target = combine_content[np.isin(combine_content[:, 0], res_index.values)]
            target = target[np.argsort(target[:, 0])]
            res_arr['instrument'] = target[:, -1]
            result_target[date] = res_arr.copy()
        self.result_target = result_target
        
        return result_target
                         
