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
from sklearn.preprocessing import MinMaxScaler
from cus_utils.wise_corrcef import analyze_model_complexity
from cus_utils.process import create_from_cls_and_kwargs
from .mlp_module import MlpModule
import cus_utils.global_var as global_var
from cus_utils.visualization import plot_feature_heatmap,plot_sample_lines
from darts_pro.act_model.union_transformer import UnionTransCombine
from darts_pro.act_model.fur_industry_ts import FurIndustryMixer, FurStrategy
from losses.mixer_loss import FuturesIndustryLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from darts_pro.act_model.union_transformer import TimeFeatureEncoder
from .multiTask_optimizer import MultiTaskOptimizer
from cus_utils.common_compute import linear_map, pairwise_compare, min_max_norm,min_max_norm_reverse, normalization_axis
from tft.class_define import CLASS_SIMPLE_VALUES, get_simple_class
from trader.utils.data_stats import DataStats, RESULT_FILE_PATH, RESULT_FILE_VIEW, INTER_RS_FILEPATH

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# TRACK_DATE = [20250728,20250715,20250731]
TRACK_DATE = [20250812, 20250811, 20250825, 20250728, 20250715, 20250731]
# TRACK_DATE = [item for item in range(20250825,20250905)]
TRACK_DATE = [item for item in range(20250425,20250515)]
# TRACK_DATE = [20250312, 20250328, 20250322]
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

    def __call__(self, module, input, output):
        """执行网络输入输出的变量记录"""
        
        name = self.name
        S = self.nodes_num
        B = input[0].shape[0]
        if B>100:
            B = int(B/S)
        
        input = input[0]
        if module.training:
            input = input.detach()
            output = output.detach()
        # 训练阶段和验证阶段关注不同的内容
        self.features[name + "_input"] = input.reshape(B,S,-1)
        self.features[name + "_output"] = output.reshape(B,S,-1)                

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
            
            combine_nodes = FuturesMappingUtil.get_industry_instrument(self.train_sw_ins_mappings, without_main=True)
            combine_nodes_num = np.array([ins.shape[0] for ins in combine_nodes])
            combine_nodes_num = torch.Tensor(combine_nodes_num).int().to(self.device)
            # 加入短期指标
            pred_len = self.output_chunk_length    
            combine_nodes = FuturesMappingUtil.get_all_instrument(self.train_sw_ins_mappings)
            combine_nodes_num = combine_nodes.shape[0]
            # 初始化时间编码器
            dataset = global_var.get_value("dataset") 
            if self.time_encoder is None:
                time_encoder = TimeFeatureEncoder('datetime',device=device)   
                # range_data = {'year':df['datetime'].dt.year.unique().tolist(),
                #               'month':df['datetime'].dt.month.unique().tolist(),
                #               'day':df['datetime'].dt.day.unique().tolist(),
                #               'dayofweek':df['datetime'].dt.dayofweek.unique().tolist(),
                #             }
                range_data = {'year':[i for i in range(2012,2026)],
                              'month':[i for i in range(0,12)],
                              'day':[i for i in range(1,32)],
                              'dayofweek':[i for i in range(0,5)],
                            }            
                time_encoder.fit_static(range_data) 
                self.time_embed_dim = time_encoder.transform(dataset.df_all.iloc[:1], device).shape[-1]    
                # 记录时间字段
                self.embed_cols = dataset.get_future_columns()
                self.time_encoder = time_encoder
            # 使用混合时间序列模型,TFT底座
            model = UnionTransCombine(
                target_feat_dim=1,
                static_num=static_covariates.shape[-1]-1,
                obs_dim=input_dim,
                fut_dim=future_covariate.shape[-1],
                time_embed_dim=self.time_embed_dim,
                hidden_dim=64,
                hidden_size=16,
                nhead=8,
                num_layers=3,
                dropout=0.1,
                pred_len=pred_len,
                sample_dim=combine_nodes_num,
                sample_heads=4,
                static_emb_dim=4,
                static_cate_emb=dataset.get_cate_dict(),
                device=device,
            ).to(device)             
            self.embedding_size = input_dim
            
            ################# 植入钩子进行中间变量输出调试 #################
            if seq==0:
                self.reg_hook(model,combine_nodes_num)
            
            return model
    
    def reg_hook(self,model,nodes_num):
        self.features = {}
        # 查看输入数据，以及经过变量选择层的输出
        model.trans_model.transformer_encoder.register_forward_hook(FeatureExtractorHook(self.features, 'transformer_encoder',nodes_num=nodes_num))
        # 查看主模型中解码层输出的前后数值
        model.trans_model.tar_decoder.register_forward_hook(FeatureExtractorHook(self.features, 'decoder',nodes_num=nodes_num))
        # 查看后置模型中基于注意力的特征输出的前后数值
        model.top_selector[0].top_att_layer.score_head.register_forward_hook(FeatureExtractorHook(self.features, 'score_head',nodes_num=nodes_num))
        # 查看注意力层的前后数值
        model.top_selector[0].top_att_layer.att_layer.attention_net.register_forward_hook(FeatureExtractorHook(self.features, 'attention_net',nodes_num=nodes_num))   
                  
    def create_loss(self, model, device="cpu"):
        return FuturesIndustryLoss(device=device, ref_model=model, lock_epoch_num=self.lock_epoch_num,output_chunk_length=self.output_chunk_length,
                                   opt_size=self.opt_size,embedding_size=self.embedding_size, target_mode=self.target_mode, 
                                   cut_len=self.cut_len, loss_weights=self.task_weights)       

    def _construct_classify_layer(self, input_dim, output_dim, device=None):
        """新增策略选择模型"""
        
        self.lock_epoch_num = 180
        
        self.top_num = 5  # 选取目标数量
        self.select_num = 10  # 一次筛选的数量
        self.trend_threhold = 0.55  # 整体趋势阈值
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
                            use_adaptive_clip=False, use_pcgrad=self.use_pcgrad)  
            optimizers.append(mt_optimizer)
        # 对应优化器，生成多个学习率
        lr_schedulers = []
        for i in range(self.opt_size):
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizers[i]
            lr_monitor = lr_sched_kws.pop("monitor", None)
            lr_scheduler = create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": self.lr_freq["interval"],
                "frequency": self.lr_freq["frequency"],
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            } 
            lr_schedulers.append(lr_scheduler_config)  
        lr_schedulers.append(lr_scheduler_config) 
        return optimizers, lr_schedulers    
    
    def create_lr_scheduler(self,lr_sched_kws,lr_monitor="val_loss"):
        # 先预热5轮，再余弦退火
        lr_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
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
                            batch_data[key] = convs[:,k,:,k_idx].flatten().cpu()   
                        emb_data = self.time_encoder.transform_inner(batch_data, device=self.device)
                        emb_data = emb_data.reshape(his_future_covs.shape[0],convs.shape[2],-1)
                        convs_emb.append(emb_data)
                    convs_emb = torch.stack(convs_emb).permute(1,0,2,3)
                    return convs_emb
                his_future_emb = transform_emb(his_future_covs)         
                future_emb = transform_emb(futures_convs).double()    
                future_single_emb =  future_emb[:,:,target_len,:]   
                out = m(static_covs,past_convs_item, his_future_emb,future_emb,future_single_emb)                
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
        
        return out_total, vr_class, out_class_total  

    def on_validation_start(self): 
        self.output_result = []
        if self.train_step_mode == 2:
            # 第二阶段，首先加载预存结果
            with open(self.result_file_path, "rb") as fin:
                self.result_data = pickle.load(fin)    
        
    def on_validation_end(self): 
        pass
        
    def on_validation_epoch_start(self): 
        self.import_price_result = None
        self.total_imp_cnt = 0
    
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
            long_diff_index_targets,
            target_info
        ) = train_batch
                
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        future_covs = input_batch[1]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        for i in range(self.get_optimizer_size()):
            (output, vr_class, tar_class) = self(input_batch, optimizer_idx=i)
            loss, detail_loss = self._compute_loss((output, vr_class, tar_class),
                            (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, long_diff_index_targets, target_info), optimizers_idx=i)
            (corr_loss, ce_loss, fds_loss, cls_loss, _) = detail_loss 
            if cls_loss[i] != 0:
                self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
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
            self.features['total_gradients'] = update_info['total_gradients']
            # update_info = opt.step_with_batch([cls_loss[i],ce_loss[i]],batch_idx=batch_idx,total_batch_number=self.trainer.num_training_batches)
            self.lr_schedulers()[i].step() 
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
                self.log("total_norm_tcn", update_info["total_norm_tcn"], batch_size=train_batch[0].shape[0], prog_bar=True)
                self.log("total_norm_ins_layer", update_info["total_norm_ins_layer"], prog_bar=True) 
            else:
                self.log("total_grad_norm", update_info["total_grad_norm"], batch_size=train_batch[0].shape[0], prog_bar=True)           
                                       
        self.log("train_loss", total_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)  
        # self.log("lr_last",self.trainer.optimizers[-2].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=False)  
        
        # 手动维护global_step变量  
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()
        return total_loss, detail_loss, output 

    def on_train_epoch_end(self):
        
        # 可视化权重和梯度
        for name,params in self.sub_models[0].named_parameters():
            global_step = self.global_step
            if not "top_selector" in name:
                continue
            if name.endswith("bias"):
                continue
            if params is not None:
                self.logger.experiment.add_histogram('weights/' + name,params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram('grad/' + name,params.grad,global_step)
        # 可视化中间特征输出      
        for name,feat in self.features.items():
            if name=='total_gradients':
                continue
            # 可视化重点层的输入输出数据
            self.logger.experiment.add_histogram(f'Features/{name}', feat.flatten(), self.current_epoch)  
        # 可视化梯度
        total_gradients = self.features['total_gradients']
        name_matches = ['trans_model.tar_decoder','top_att_layer.att_layer.attention_net','top_att_layer.score_head']
        for grad_name in total_gradients.keys():
            for item in name_matches:
                if item in (grad_name):
                    grad = total_gradients[grad_name]
                    self.logger.experiment.add_histogram('grad/' + grad_name,grad,global_step)
                    
        # 可视化品种间的比较
        self.viz_data_board(section="train")                                
        
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        loss, detail_loss, output = self.validation_step_real(val_batch, batch_idx)
        (corr_loss_combine, ce_loss, fds_loss, cls_loss, predictions) = detail_loss
        # 补充计算批次内指数数据评估
        sw_ins_mappings = self.valid_sw_ins_mappings
        indicator_idx = 0
        sw_index_data = output[0][indicator_idx][2].cpu().numpy()
        main_idx = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        main_index_feature = np.stack([data[:-1] for data in sw_index_data])
        main_targets = val_batch[-4][:, main_idx, -1, indicator_idx]
        # 计算指标数据中最后一条特征数据与前面特征数据的距离，并与实际目标值距离数据比较相关性
        # corr_dis = self.compute_feature_target_trend_corr(main_index_feature,main_targets)
        # self.log("corr_dis", corr_dis, batch_size=val_batch[0].shape[0], prog_bar=True)
        # batch_data = predictions.cpu().numpy()
        batch_data = np.ones([1])
        
        self.dump_val_data(val_batch, output, batch_data)
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
            long_diff_index_targets,
            target_info
        ) = val_batch
              
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class) 
        input_batch = self._process_input_batch(inp)
        future_covs = input_batch[1]
        (output, vr_class, vr_class_list) = self(input_batch, optimizer_idx=-1)
        
        # 全部损失
        loss, detail_loss = self._compute_loss((output, vr_class, vr_class_list),
                    (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, long_diff_index_targets, target_info), optimizers_idx=-1)
        (corr_loss, ce_loss, fds_loss, cls_loss, predictions) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True, sync_dist=True)
        preds_combine = []
        for i in range(self.opt_size):
            task_weights = self.task_weights[i]
            if ce_loss[i] != 0 and len(task_weights) > 1:
                self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            if cls_loss[i] != 0:
                self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            if fds_loss[i] != 0 and len(task_weights) > 2:
                self.log("val_fds_loss_{}".format(i), fds_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)                
            if corr_loss[i] != 0 and len(task_weights) > 3:
                self.log("val_corr_loss_{}".format(i), corr_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)   

        output_combine = (output, vr_class, price_targets, past_future_round_targets)
        return loss, detail_loss, output_combine       

    def _process_input_batch(
        self, input_batch
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
            target_class
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
        # 去掉正泰zsall部分
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        past_index_targets = past_index_targets[:, indus_rel_index,:,:]
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:self.input_chunk_length,:]
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, x_future_array, historic_future_covariates, future_covariates, 
                static_covariates, price_targets, past_round_targets, past_index_targets,target_class)
    
    def _compute_loss(self, output, target, optimizers_idx=0):
        """重载父类方法"""

        (future_target, future_covs, target_class, past_future_round_targets, index_round_targets, price_targets, long_diff_index_targets, target_info) = target 
        # 只保留最后一天的数值，作为损失目标
        future_round_targets = past_future_round_targets[:,:,-self.output_chunk_length:,:]  
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output, (future_target, future_covs, target_class, future_round_targets, index_round_targets, price_targets, long_diff_index_targets, target_info),
                    sw_ins_mappings=sw_ins_mappings, optimizers_idx=optimizers_idx, top_num=self.top_num, epoch_num=self.current_epoch)        

    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        rate_total, coll_result = self.combine_result_data(self.output_result, pred_top_num=self.pred_top_num)
        date_total_num = float(coll_result['date'].unique().shape[0])
        
        # 打印相关指标
        if rate_total is not None and rate_total.shape[0] > 0:
            for col in rate_total.columns:
                if col != "total_cnt":
                    self.log(col, rate_total[col].values[0], prog_bar=True)  
            
            dur_num = self.cut_len - 1
            anno_yield = rate_total['yield_rate'].values[0] * (240 / date_total_num) / (2 * self.pred_top_num * dur_num) 
            self.log("anno_yield", anno_yield, prog_bar=True) 
        
        output_3d, past_target_3d, future_target_3d, target_class_3d, price_targets_total, \
            past_future_round_targets_total, long_diff_index_targets_total, index_round_targets_3d, target_info_3d = self.combine_output_total(self.output_result)
        viz_total_size = 0
        
        if self.mode is None or not self.mode.startswith("pred_"):
            # 验证模式，进行board的可视化
            self.viz_data_board(section="valid")
            return
        
        # 测试模式，在此进行结果的可视化
        print("all date:", coll_result['date'].unique())
        coll_result.to_csv(self.coll_record_file_path, index=False)
        self.log("date_total_num", date_total_num, prog_bar=True) 
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
        dec_output = output_3d[-3]
        cls_output = output_3d[2]     
        sw_index = output_3d[3][0]       
        batch_trend_data = output_3d[5]
        predictions = batch_trend_data
        price_targets_main = long_diff_index_targets_total.squeeze(-1)
        
        for index in range(target_class_3d.shape[0]):
        
            sw_index_cur = sw_index[index,0]
            viz_total_size += 1
            target_class_item = target_class_3d[index]
            keep_index = np.where(target_class_item >= 0)[0]
            round_targets = past_future_round_targets_total[index]
            index_output = output_3d[3][0][index, -1]
            ts_arr = target_info_3d[index]
            date = int(ts_arr[keep_index][0]["future_start_datetime"])
            if not date in TRACK_DATE:
                continue    
            coll_item = coll_result[coll_result['date'] == date]
            instruments, k_idx, _ = np.intersect1d(ins_all, keep_index, return_indices=True)
            ts_arr_ins = ts_arr[instruments]
            price_diff_range_ins = np.array([(item['open_array'][-self.output_chunk_length + self.cut_len - 1] - item['open_array'][-self.output_chunk_length]) / item['open_array'][-self.output_chunk_length] * 100 for item in ts_arr_ins])
            # index_price = long_diff_index_targets_total[index,0]
            index_price = price_diff_range_ins.mean()        
            price_diff = self.criterion.compute_diff_range_class(ts_arr[main_index], target_info_arr=ts_arr[instruments], is_main=True)[2]    
            
            for j in range(1):
                inner_class_item = target_class_item[ins_all]
                inner_index = np.where(inner_class_item >= 0)[0]           
                ins_output = cls_output[j][index,:][:ins_all.shape[0]]
                ins_output = ins_output[inner_index]
                ins_output_mean = ins_output.mean()
                ins_output_scale = MinMaxScaler().fit_transform(np.expand_dims(ins_output, -1)).squeeze(-1)
                dec_output_item = dec_output[index,:,:, j]
                # dec_output_item = dec_output[index,inner_index,-1,j] 
                fur_round_target = round_targets[instruments, -self.output_chunk_length+self.cut_len-1, j]
                price_targets = price_targets_total[index, instruments]
                # 品种比对图
                if j in DRAW_SEQ_DETAIL:
                    price_array_range = np.array([self.criterion.compute_diff_range_class(item)[0] for item in ts_arr[instruments]])
                    price_array_range = price_array_range / 10
                    name_arr = []
                    for inner_index, item in enumerate(ts_arr[instruments]):
                        match_item = coll_item[coll_item['instrument'] == item['instrument']]
                        if match_item.shape[0] > 0:
                            trend = match_item['trend_value'].values[0]
                            name_arr.append(item["instrument"] + "_match_" + str(trend))
                        else:
                            name_arr.append(item["instrument"])
                    # view_data = np.stack([ins_output_scale,price_targets,price_array_range]).transpose(1,0)
                    view_data = np.stack([ins_output, fur_round_target, price_array_range]).transpose(1, 0)
                    # view_data = np.stack([ins_output,dec_output_item,fur_round_target,price_array_range]).transpose(1,0)
                    win = "detail_target_{}_{}=".format(j, viz_total_size)
                    index_target = round_targets[main_index, -1, j]
                    index_output = dec_output_item[-1, -1]
                    trend_value = ts_arr[main_index]['trend_value']
                    target_title = "Detail_{}_{}_{},date:{}".format(round(index_price, 3), round(index_target, 3), round(trend_value, 3), date)  
                    # target_title = "Detail_{}_{}_{},date:{}".format(round(index_mean,3),round(index_target,3),round(dec_output_mean,3),date)  
                    viz_result_detail.viz_bar_compare(view_data, win=win, title=target_title, rownames=name_arr, legends=["pred_cls", "target", "price"])
                # 品种走势图,所有候选的目标走势和价格走势
                # if j in DRAW_SEQ_ITEM:
                #     price_diff = np.pad(price_diff,(1,0),'constant',constant_values=(0,0))
                #     win = "trend_line_{}_{}_{}".format(j,viz_total_size,date)
                #     target_title = "trend_line_{},date_{}".format(round(index_price,3),date)
                #     # output = sw_index[index]
                #     output = dec_output_item.squeeze(-1)
                #     output_value = np.pad(output,(self.input_chunk_length,0),'constant',constant_values=(0,0))
                #     past_target_values = past_target_3d[index,main_index,:,j]
                #     futures_round_targets = round_targets[main_index,:,j]
                #     target_values = np.concatenate([past_target_values,futures_round_targets[-self.output_chunk_length:]])
                #     view_data = np.stack([output_value,target_values,price_diff]).transpose(1,0)
                #     names = ['pred','target','price_diff']                              
                #     viz_result_ext.viz_matrix_var(view_data,win=win,title=target_title,names=names)                             
    
    def viz_data_board(self,section="valid"):
        """可视化验证集数据流"""
        
        for name,feat in self.features.items():
            if name=='total_gradients':
                continue
            feat = feat.squeeze(-1)
            # 针对验证结果，以品种为单位计算均值，并可视化
            if len(feat.shape)==2:
                ins_feat = feat
            elif len(feat.shape)==3:
                ins_feat = feat.mean(-1)    
            ins_feat = ins_feat.cpu().numpy()
            self.logger.experiment.add_figure('{}_heatmap_{}/'.format(section,name), plot_feature_heatmap(ins_feat), global_step=self.global_step)
            fig = plot_sample_lines(ins_feat, sample_indices=range(8), title='Batch Sample Features')
            self.logger.experiment.add_figure('{}_lines_{}/'.format(section,name), fig, global_step=self.global_step)     
                                                 
    def dump_val_data(self, val_batch, outputs, batch_data):
    
        output, vr_class, price_outputs, past_future_round_targets = outputs
        choice_out, trend_value, combine_index = vr_class
        (past_target, past_covariates, historic_future_covariates, future_covariates,
            static_covariates, past_future_covariates, future_target, target_class, price_targets, _, index_round_targets, long_diff_index_targets, target_info) = val_batch
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
        output_res = (output, choice_out.cpu().numpy(), batch_data, combine_index.cpu().numpy(), past_target.cpu().numpy(),
                      future_target.cpu().numpy(), target_class.cpu().numpy(),
                      price_targets.cpu().numpy(), past_future_round_targets.cpu().numpy(), whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(), long_diff_index_targets.cpu().numpy(), target_info)
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
        cls_total = [None for _ in range(len(self.past_split))]
        dec_total = []
        comm_index_total = [None for _ in range(len(self.past_split))]
        choice_total = []
        trend_total = []
        combine_index_total = []
        index_round_targets_total = []
        long_diff_index_targets_total = []
        for item in output_result:
            (output, choice, batch_data, combine_index, past_target, future_target, target_class, price_targets, past_future_round_targets, whole_index_round_targets, index_round_targets, long_diff_index_targets, target_info) = item
            x_bar_inner = []
            dec_inner = []
            for i in range(len(self.past_split)):
                output_item = output[i]
                dec_out, sv_indus, comm_index = output_item 
                dec_inner.append(dec_out.cpu().numpy())
                # 合并列表中的品种维度部分
                sv_indus = torch.cat(sv_indus, dim=1).squeeze(-1)
                if cls_total[i] is None:
                    cls_total[i] = sv_indus.cpu().numpy()
                else:
                    cls_total[i] = np.concatenate([cls_total[i], sv_indus.cpu().numpy()], axis=0)                
                if comm_index_total[i] is None:
                    comm_index_total[i] = comm_index.cpu().numpy()
                else:
                    comm_index_total[i] = np.concatenate([comm_index_total[i], comm_index.cpu().numpy()], axis=0)
                
            dec_inner = np.stack(dec_inner).transpose(1, 2, 3, 4,0)
            dec_total.append(dec_inner)
            # ce_index_inner = np.stack(ce_index_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
            choice_total.append(choice)
            trend_total.append(batch_data)
            combine_index_total.append(combine_index)
            
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            price_targets_total.append(price_targets)
            past_future_round_targets_total.append(past_future_round_targets)
            whole_index_round_targets_total.append(whole_index_round_targets)
            index_round_targets_total.append(index_round_targets)
            long_diff_index_targets_total.append(long_diff_index_targets)
        
        dec_total = np.concatenate(dec_total)
        x_bar_total = np.concatenate(x_bar_total)
        choice_total = np.concatenate(choice_total)
        trend_total = np.concatenate(trend_total)
        combine_index_total = np.concatenate(combine_index_total)
        
        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        price_targets_total = np.concatenate(price_targets_total)
        past_future_round_targets_total = np.concatenate(past_future_round_targets_total)
        whole_index_round_targets_total = np.concatenate(whole_index_round_targets_total)
        index_round_targets_total = np.concatenate(index_round_targets_total)
        target_info_total = np.concatenate(target_info_total)
        long_diff_index_targets_total = np.concatenate(long_diff_index_targets_total)
                    
        return (x_bar_total, sv_total, cls_total, comm_index_total, dec_total, trend_total, combine_index_total), \
                    past_target_total, future_target_total, target_class_total, price_targets_total, past_future_round_targets_total, \
                    long_diff_index_targets_total, index_round_targets_total, target_info_total        
                           
    def combine_result_data(self, output_result=None, predict_mode=False, pred_top_num=2):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # return None,None,None,None
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d, past_target_3d, future_target_3d, target_class_3d, price_targets_3d, batch_trend_data, long_diff_index_targets_3d, \
            index_round_targets_3d, target_info_3d = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d == 3)[0].shape[0]
        rate_total = []
        result_date_list = None
        
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        industry_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        industry_index_proxy = [main_index] if self.target_mode[0] in [3, 6] else industry_index
        # 按照时间索引暂存预测数据，用于全局化共享使用
        glo_match_data = []
        ref_output_index = 1
        target_len = self.cut_len-1
        for i in range(target_class_3d.shape[0]):
            past_target = past_target_3d[i]
            target_class_list = target_class_3d[i]
            target_info_list = target_info_3d[i]
            keep_index = np.where(target_class_list >= 0)[0]
            keep_index = np.intersect1d(keep_index, ins_all)              
            date = target_info_list[0]['future_start_datetime']
            ce_index = [item[i] for item in output_3d[3]]
            # 根据区间多段预测，取得关注段数值在整个区间的相对数值--Focus-lmx
            sw_index = output_3d[3][ref_output_index][i]
            sw_index_nor = normalization_axis(sw_index)      
            # 取得实际价格涨跌幅总体情况，作为评分参考
            diff_range_arr = np.array([item['diff_range'] for item in target_info_list[keep_index]])
            diff_range_arr_mean = diff_range_arr.mean(0)[:-self.output_chunk_length]
            diff_range_mean = diff_range_arr_mean.mean()
            past_target_arr_mean = past_target[keep_index,:,ref_output_index].mean(0)
            # 把目标过去数值以及预测数值拼接，并映射到价格区间预测
            combine_pred_tar = np.concatenate([past_target_arr_mean,sw_index])
            price_pred_all = linear_map(combine_pred_tar,diff_range_arr_mean.min(),diff_range_arr_mean.max())
            price_pred = price_pred_all[-self.output_chunk_length:]
            # 直接预定义阈值范围(依据全局数据计算出并取25%-75%的区间阈值)
            index_mean_range = [-0.5,0.4]
            diff_range_ref = min_max_norm(price_pred[target_len],range=index_mean_range)
            # 主要使用预测段计分，辅助使用整体计分，综合生成实际得分
            ref_weights = 0.0
            trend_value = ref_weights * diff_range_ref + (1-ref_weights) * sw_index_nor[target_len]
            target_info_list[main_index]['trend_value'] = trend_value
            # 根据配置，决定针对行业数据进行处理还是针对整体指数数据进行处理
            for j, indus_index in enumerate(industry_index_proxy):
                target_info = target_info_list[indus_index]
                indus_code = target_info["instrument"]
                date = target_info["future_start_datetime"]  
                # 因为预测的是最后一个未来日期和前面的差值，因此按照最后一个时间序号作为序列编号
                time_index = target_info["future_end"] - 1 
                # 预测数据放入记录，与最后一个日期序号对应
                pred_data = ce_index[0][-1]
                glo_match_data.append([indus_index, date, indus_code, time_index, pred_data])
        
        columns = ["indus_index", "date", "indus_code", "time_index", "pred_data"]       
        glo_match_data = pd.DataFrame(np.array(glo_match_data), columns=columns)
        glo_match_data['time_index'] = glo_match_data['time_index'].astype(int)
        glo_match_data['date'] = glo_match_data['date'].astype(float).astype(int)
        
        import_price_result_list = None
        result_total_list = None
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
            ce_index = [item[i] for item in output_3d[3]]
            cls_index = [item[i] for item in output_3d[2]]
            dec_out = output_3d[4][i]
            output_list = [cls_index, ce_index, output_3d[4][i], output_3d[5], output_3d[6][i], dec_out]
            price_target_list = price_targets_3d[i]
            date = int(target_info_list[np.where(target_class_list >= 0)[0][0]]["future_start_datetime"])
            index_round_targets = index_round_targets_3d[i]
            if not (date >= STAT_DATE[0] and date <= STAT_DATE[1]):
                continue              
            # 把之前生成的预测值，植入到target_info基础信息中，后续使用
            for target_info in target_info_list[industry_index]:
                if target_info is None:
                    continue
                past_start = target_info["past_start"]
                future_end = target_info["future_end"]
                pred_data_combine = glo_match_data[(glo_match_data["indus_code"] == target_info["instrument"]) & (glo_match_data["time_index"] > past_start)
                                & (glo_match_data["time_index"] <= future_end)][["time_index", "pred_data"]]
                if pred_data_combine.shape[0] == 0:
                    continue
                if pred_data_combine.shape[0] == (self.input_chunk_length + self.output_chunk_length):
                    # 如果长度匹配，则全部使用
                    target_info["pred_data"] = pred_data_combine.values[:, 1].astype(float)
                else:
                    # 如果长度不匹配，则根据索引号缺失的记录进行插值
                    target_info["pred_data"] = pred_data_combine.values[:, 1]    
                    tmp_data = np.array([i for i in range(target_info["past_start"], target_info["future_end"])])
                    tmp_data = np.stack([tmp_data, np.array([0 for _ in range(self.input_chunk_length + self.output_chunk_length)])]).transpose(1, 0)
                    tmp_data = pd.DataFrame(tmp_data, columns=["time_index", "pred_data"])    
                    data = pd.merge(tmp_data, pred_data_combine, on='time_index', how='left').fillna(0)  
                    target_info["pred_data"] = data.values[:, -1].astype(float)      
            
            # 生成目标索引
            result_list = self.build_import_index(output_data=output_list, target_info=target_info_list, pred_top_num=pred_top_num,
                            target=whole_target, price_target=price_target_list, index_round_targets=index_round_targets, date=date,
                            combine_instrument=combine_content)
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

            # 验证准确性
            coll_results = self.collect_result_compindex(date=date, target_info=target_info_list, result_list=result_list, keep_index=keep_index)  
            # 把结果数据整合到预测记录中
            if result_total_list is None:
                result_total_list = coll_results
            else:
                result_total_list = pd.concat([result_total_list, coll_results])                

        if predict_mode:
            return result_date_list      
        
        # 统合计算准确率数值
        rate_columns = ["total_cnt", "yield_rate", "win_rate"]    
        rate_total = [result_total_list.shape[0],
                      round(result_total_list['diff_range'].sum(), 3),
                      round(np.sum(result_total_list['diff_range'] > 0) / result_total_list.shape[0], 3),
                      ]
        rate_total = pd.DataFrame(np.array([rate_total]), columns=rate_columns)
        for i in range(4):
            distribute = round(np.sum(result_total_list['target_class'] == i) / result_total_list.shape[0], 3)
            rate_total['dist_{}'.format(i)] = round(distribute, 3)
            
        if rate_total.shape[0] == 0:
            return None, None
        
        return rate_total, result_total_list
    
    def build_import_index(self, date=None, pred_top_num=2, output_data=None, target=None, price_target=None, target_info=None,
                           combine_instrument=None, index_round_targets=None): 
        """生成涨幅达标的预测数据下标"""
        
        (cls_values, ce_values, choice, trend_value, combine_index, dec_out) = output_data
        
        import_index_list = self.strategy_top_bidi(ce_values, cls_values, dec_out, pred_top_num=pred_top_num, target=target, target_info=target_info,
                                            index_round_targets=index_round_targets, combine_instrument=combine_instrument)
        self.strategy_main_index(ce_values, cls_values, dec_out, pred_top_num=pred_top_num, target=target, target_info=target_info,
                                            index_round_targets=index_round_targets, combine_instrument=combine_instrument)
 
        # 构建结果集
        result_list = pd.DataFrame(np.array(import_index_list), columns=['top_index', 'top_flag'])
        result_list['top_flag'] = result_list['top_flag'].astype(int)
        result_list['date'] = date       
        return result_list

    def strategy_main_index(self, ce, cls, dec_out, pred_top_num=2, target=None, target_info=None, index_round_targets=None, combine_instrument=None):
        """衡量指数数据"""

        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)        
        main_index_feature = target_info[main_index]['trend_value']           
        return main_index_feature

    def strategy_top_bidi(self, ce, cls, dec_out, pred_top_num=2, target=None, target_info=None, index_round_targets=None, combine_instrument=None):
        """筛选品种明细,使用双向模式"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        top_num = pred_top_num
        cancidate_list = []
        mode = 'single'
        # 同时从正反2个方向选取品种
        pre_index = self.compute_arg_sort(cls, dec_out, mode=mode, trend=1, top_num=top_num)
        for i in range(top_num): 
            import_index_real = ins_all[pre_index[i]]
            cancidate_list.append([import_index_real, 1])    
        pre_index = self.compute_arg_sort(cls, dec_out, mode=mode, trend=0, top_num=top_num)
        for i in range(top_num): 
            import_index_real = ins_all[pre_index[i]]
            cancidate_list.append([import_index_real, 0])                          
        
        return cancidate_list
        
    def strategy_top_bidi_old(self, ce, cls, dec_out, pred_top_num=2, target=None, target_info=None, index_round_targets=None, combine_instrument=None):
        """筛选品种明细,使用双向模式，结合整体走势预测"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)        
        trend_value = target_info[main_index]['trend_value']              
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        cancidate_list = []
        mode = 'single'
        # 同时从正反2个方向选取品种
        if trend_value>0.75:
            # 如果整体趋势看涨，则增加多方候选数量
            top_num_long = pred_top_num + 1
            top_num_short = pred_top_num - 1
        elif trend_value<0.25:
            # 如果整体趋势看跌，则增加空方候选数量
            top_num_short = pred_top_num + 1
            top_num_long = pred_top_num - 1
        else:
            top_num_long = pred_top_num
            top_num_short = pred_top_num
        pre_index = self.compute_arg_sort(cls, dec_out, mode=mode, trend=1, top_num=top_num_long)
        for i in range(top_num_long): 
            import_index_real = ins_all[pre_index[i]]
            cancidate_list.append([import_index_real, 1])    
        pre_index = self.compute_arg_sort(cls, dec_out, mode=mode, trend=0, top_num=top_num_short)
        for i in range(top_num_short): 
            import_index_real = ins_all[pre_index[i]]
            cancidate_list.append([import_index_real, 0])                          
        
        return cancidate_list

    def compute_arg_sort(self, cls, dec_out, mode='single', trend=1, top_num=2):
        """根据输出进行排序"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        node_num = ins_all.shape[0]
        cls_main = cls[0][:node_num]
        cls_main = cls[0][2*node_num:3*node_num]
        topk_mask_weights = cls[0][node_num:2*node_num]
        long_index = np.argwhere(topk_mask_weights==1)[:,0][:top_num]
        short_index = np.argwhere(topk_mask_weights==-1)[:,0][:top_num]
        top_index = np.concatenate([long_index,short_index])
        pred_top = cls_main[top_index]
        if trend == 1:
            pre_index = np.argsort(-cls_main)[:top_num]
        else:
            pre_index = np.argsort(cls_main)[:top_num]
        # if trend == 1:
        #     pre_index = long_index
        # else:
        #     pre_index = short_index   
        # if trend == 1:
        #     pre_index = np.argsort(-pred_top)[:top_num]
        #     pre_index = top_index[pre_index]
        # else:
        #     pre_index = np.argsort(pred_top)[:top_num]
        #     pre_index = top_index[pre_index] 
        return pre_index.astype(int)
       
    def compute_arg_sort_by_index(self, cls, dec_out, mode='single', trend=1, top_num=2):
        """根据输出进行排序"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_all = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
        cls_main = cls[0][:ins_all.shape[0]]
        index_long,index_short = self.criterion.filter_top_index(torch.Tensor(cls_main),top_num=self.top_num,ins_rel_index=ins_all)
        index_long = index_long.numpy()
        index_short = index_short.numpy()
        if self.pred_mode == 'single':
            if trend == 1:
                pre_index = index_long[:top_num]
            else:
                pre_index = index_short[:top_num]
        else:
            cls_att = cls[0][-ins_all.shape[0]:]
            if trend == 1:
                pre_index = self.compute_comprehensive_info(cls_main, cls_att)[0].values[:, 0][:top_num]
            else:
                pre_index = self.compute_comprehensive_info(cls_main, cls_att)[0].values[:, 0][-top_num:]
        return pre_index.astype(int)
        
    def compute_comprehensive_info(self, cls, dec_out):
        """计算综合排序"""
        
        if self.candidate_inverse:
            second_can = -dec_out[:, -1, 0]
        else:
            second_can = dec_out[:, -1, 0]
        data = pd.DataFrame(np.stack([cls[0], second_can]).transpose(1, 0), columns=['cls', 'dec'])
        from cus_utils.wise_corrcef import calculate_comprehensive_rank_from_scores
        return calculate_comprehensive_rank_from_scores(data, weights=self.pred_weights, normalization_method='minmax')

    def collect_result_compindex(self, date=None, target_info=None, result_list=None, keep_index=None):
 
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage == RunningStage.TRAINING else self.valid_sw_ins_mappings
        industry_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        
        coll_results = []
        # 对于预测数据，生成对应涨跌幅类别
        for index, row in result_list.iterrows():
            imp_idx = row['top_index']
            overroll_trend = row['top_flag']
            ts = target_info[imp_idx]
            diff_range, p_taraget_class, _ = self.criterion.compute_diff_range_class(ts)
            # 根据多空判断取得实际对应的类别
            if overroll_trend == 0:
                diff_range_with_trend = -diff_range
                p_taraget_class = np.array([3, 2, 1, 0])[p_taraget_class]
            else:
                diff_range_with_trend = diff_range
            coll_results.append([imp_idx, ts["instrument"], diff_range_with_trend, p_taraget_class, overroll_trend])    
        
        coll_results = np.array(coll_results)
        coll_results = pd.DataFrame(coll_results, columns=['top_index', 'instrument', 'diff_range', 'target_class', 'trend_value'])
        coll_results['diff_range'] = coll_results['diff_range'].astype(float)
        coll_results['target_class'] = coll_results['target_class'].astype(int)
        coll_results['date'] = date
        
        return coll_results        
        
    def compute_total_trend(self, main_index_feature, batch_price):
        """计算指数数据,进行批次内匹配"""
        
        # 取得最后一条指数特征值，并分别与前值进行距离匹配，取得最相近的几条记录，并取这几条记录对应的实际价格的平均值
        distances = self.compute_feature_distance(main_index_feature)
        match_indexes = np.argsort(distances)[:3]
        trend_price = batch_price[match_indexes].mean()
        
        return trend_price
    
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
            long_diff_index_targets,
            target_info
        ) = batch
               
        inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
               static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class)     
        input_batch = self._process_input_batch(inp)
        (output, vr_class, vr_class_list) = self(input_batch, optimizer_idx=-1)
        choice_out, trend_value, combine_index = vr_class
        
        # 只获取整体指数的价格数据
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output, choice_out.cpu().numpy(), trend_value.cpu().numpy(), combine_index.cpu().numpy(), past_target.cpu().numpy(),
                      future_target.cpu().numpy(), target_class.cpu().numpy(),
                      price_targets.cpu().numpy(), past_future_round_targets.cpu().numpy(), whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(), long_diff_index_targets.cpu().numpy(), target_info)
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
                         
