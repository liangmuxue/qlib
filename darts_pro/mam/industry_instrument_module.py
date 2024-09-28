import os

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import tsaug

from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from cus_utils.tensor_viz import TensorViz

import cus_utils.global_var as global_var
from darts_pro.act_model.indus3d_ts import Indus3D
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.clustering_loss import Indus3DLoss
from cus_utils.common_compute import compute_average_precision
from darts_pro.data_extension.industry_mapping_util import IndustryMappingUtil
from gunicorn import instrument

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from .mlp_module import MlpModule


class IndustryTogeModule(MlpModule):
    """聚合行业数据一起预测的模型"""
    
    def __init__(
        self,
        ins_dim: int,
        output_dim: Tuple[int, int],
        variables_meta_array: Tuple[Dict[str, Dict[str, List[str]]],Dict[str, Dict[str, List[str]]]],
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
        use_weighted_loss_func=False,
        past_split=None,
        batch_file_path=None,
        device="cpu",
        train_sw_ins_mappings=None,
        valid_sw_ins_mappings=None,
        **kwargs,
    ):
        self.ins_dim = ins_dim
        self.mode = None
        self.train_sw_ins_mappings = train_sw_ins_mappings
        self.valid_sw_ins_mappings = valid_sw_ins_mappings
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,batch_file_path=batch_file_path,
                                    device=device,**kwargs)  
        
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
                _
            ) = self.train_sample
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariate[...,past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            historic_future_covariates_shape = historic_future_covariate.shape[-1]
            # 记录动态数据长度，后续需要切片
            self.dynamic_conv_shape = past_target_shape + past_covariates_shape
            input_dim = (
                past_target_shape
                + past_covariates_shape
                + historic_future_covariates_shape
            )
    
            output_dim = 1
    
            future_cov_dim = (
                future_covariate.shape[-1] if future_covariate is not None else 0
            )
            
            static_cov_dim = (
                # 只保留归一化字段
                static_covariates.shape[-1] - 1
                if static_covariates is not None
                else 0
            )
    
            model = Indus3D(
                input_dim=input_dim,
                output_dim=output_dim,
                future_cov_dim=future_cov_dim,
                static_cov_dim=static_cov_dim,
                ins_dim=self.ins_dim,
                train_sw_ins_mappings=self.train_sw_ins_mappings, # 新增行业分类和股票映射关系,需要分为训练和验证两组
                valid_sw_ins_mappings=self.valid_sw_ins_mappings,
                num_encoder_layers=3,
                num_decoder_layers=3,
                decoder_output_dim=16,
                hidden_size=hidden_size,
                temporal_width_past=9,
                temporal_width_future=None,
                temporal_decoder_hidden=32,
                use_layer_norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            )           
            
            return model

    def create_loss(self,model,device="cpu"):
        return Indus3DLoss(self.ins_dim,device=device,ref_model=model) 

    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """整合多种模型，主要使用MLP方式"""
        
        out_total = []
        out_class_total = []
        batch_size = x_in[1].shape[0]
        
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]            
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                x_in_item = (past_convs_item,x_in[1],x_in[2],x_in[3])
                out = m(x_in_item)
                out_class = torch.ones([batch_size,self.output_chunk_length,1]).to(self.device)
            else:
                # 模拟数据
                out = torch.ones([batch_size,self.output_chunk_length,self.output_dim[0],1]).to(self.device)
                out_class = torch.ones([batch_size,1]).to(self.device)
            out_total.append(out)    
            out_class_total.append(out_class)
            
        # 根据预测数据进行二次分析
        vr_class = torch.ones([batch_size,len(CLASS_SIMPLE_VALUES.keys())]).to(self.device) # vr_class = self.classify_vr_layer(focus_data)
        return out_total,vr_class,out_class_total
        
    def on_train_epoch_start(self):  
        self.loss_data = []
    def on_train_epoch_end(self):  
        pass
        
    def on_validation_epoch_start(self):  
        self.import_price_result = None
        self.total_imp_cnt = 0
                    
    def training_step_real(self, train_batch, batch_idx): 
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
            last_targets,
            sw_indus_targets_total,
            target_info
        ) = train_batch
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,sw_indus_targets_total)     
        past_target = train_batch[0]
        input_batch,future_indus_targets = self._process_input_batch(inp)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,last_targets,future_indus_targets),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            # self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            # self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            self.loss_data.append(detail_loss)
            total_loss += loss     
            # 手动更新参数
            opt = self.trainer.optimizers[i]
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
            self.lr_schedulers()[i].step()                           
        self.log("train_loss", total_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr0",self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)  
        
        # 手动维护global_step变量  
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()
        return total_loss,detail_loss,output      

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""

        loss,detail_loss,output = self.validation_step_real(val_batch, batch_idx)  
       
        if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag or True:
            self.dump_val_data(val_batch,output,detail_loss)
            
        return loss,detail_loss
           
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
            last_targets,
            sw_indus_targets_total,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,sw_indus_targets_total) 
        input_batch,future_indus_targets = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,last_targets,future_indus_targets),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,last_targets,future_indus_targets)
        return loss,detail_loss,output_combine

    def _process_input_batch(
        self, input_batch
    ):
        """重载方法，以适应数据结构变化"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            sw_indus_targets,
        ) = input_batch
        dim_variable = -1

        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]
            # 修改协变量生成模式，只取自相关目标作为协变量
            conv_defs = [
                        past_target[...,i:i+1],
                        past_covariates_item,
                        historic_future_covariates,
                ]            
            x_past = torch.cat(
                [
                    tensor
                    for tensor in conv_defs if tensor is not None
                ],
                dim=dim_variable,
            )
            x_past_array.append(x_past)
            
        # 忽略静态协变量第一列(索引列),后边的都是经过归一化的
        static_covariates = static_covariates[...,1:]
        # 行业分类过去目标值
        past_indus_targets = sw_indus_targets[...,:self.input_chunk_length]
        # 行业分类未来数值幅度区间,计算3个交易日的差值
        future_indus_targets = sw_indus_targets[...,self.input_chunk_length:]
        future_indus_targets = torch.sum(future_indus_targets[...,:3],dim=-1)
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, future_covariates, static_covariates,past_indus_targets),future_indus_targets
               
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,last_target,future_indus_targets) = target   
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,last_target,future_indus_targets),sw_ins_mappings=sw_ins_mappings,mode=self.step_mode,optimizers_idx=optimizers_idx)        

    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        # SANITY CHECKING模式下，不进行处理
        # if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
        #     return    
        return
    
        rate_total,total_imp_cnt,target_top_match = self.combine_result_data(self.output_result)  
        sr = []
        for item in list(rate_total.values()):
            if len(item)==0:
                continue
            item = np.array(item)
            sr.append(item)
        sr = np.stack(sr)      
        for i in range(len(self.past_split)):
            industry_score = np.mean(target_top_match[:,0,i])
            instrument_score = np.mean(target_top_match[:,1,i])
            self.log("industry_score_{}".format(i), industry_score, prog_bar=True) 
            self.log("instrument_score_{}".format(i), instrument_score, prog_bar=True) 
        
        if sr.shape[0]==0:
            return
        
        # 汇总计算准确率,取平均数
        sum_v = sr[:,-1]
        sr_rate = sr/sum_v[:, np.newaxis]
        combine_rate = np.mean(sr_rate,axis=0)
        # 按照日期计算最小准确率
        combine_rate_min = np.min(sr_rate,axis=0)
        for i in range(len(CLASS_SIMPLE_VALUES.keys())):
            self.log("score_{} rate".format(i), combine_rate[i], prog_bar=True) 
            # self.log("score_{} min rate".format(i), combine_rate_min[i], prog_bar=True) 
        self.log("total cnt", sr[:,-1].sum(), prog_bar=True)  
        self.log("total_imp_cnt", total_imp_cnt, prog_bar=True)  
        if self.mode is not None and self.mode.startswith("pred"):
            for date in rate_total.keys():
                stat_data = rate_total[date]
                # print("date_{} stat:{}".format(date,stat_data))

        # 如果是测试模式，则在此进行可视化
        if self.mode is not None and self.mode.startswith("pred_"):
            tar_viz = global_var.get_value("viz_data")
            viz_total_size = 0
            output_3d,past_target_3d,future_target_3d,target_class_3d,last_targets_3d,future_indus_targets,target_info_3d = self.combine_output_total(self.output_result)
            size = 3      
            industry_info = IndustryMappingUtil.get_industry_info(self.valid_sw_ins_mappings)
            rownames = industry_info['name'].values.tolist()
            for index in range(target_class_3d.shape[0]):
                if viz_total_size>10:
                    break
                viz_total_size+=1
                keep_index = np.where(target_class_3d[index]>=0)[0]
                indus_targets = future_indus_targets[index]
                date = target_info_3d[index][keep_index[0]]["future_start_datetime"]
                target_title = "industry compare,date:{}".format(date)
                win = "industry_comp_{}".format(viz_total_size)
                # tar_viz.viz_matrix_var(price_target,win=win,title=target_title,names=names)
                for j in range(len(self.past_split)):
                    cls_output = output_3d[2][index,:,j]
                    view_data = np.stack([cls_output,indus_targets]).transpose(1,0)
                    win = "target_comp_{}_{}".format(j,viz_total_size)
                    target_title = "target_{} compare,date:{}".format(j,date)
                    tar_viz.viz_bar_compare(view_data,win=win,title=target_title,rownames=rownames,legends=["pred","target"])                
                
    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,last_targets,future_indus_targets = outputs
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,last_targets,sw_indus_targets,target_info) = val_batch
        # 保存数据用于后续验证
        output_res = (output,past_target.cpu().numpy(),future_target.cpu().numpy(),target_class.cpu().numpy(),
                      last_targets.cpu().numpy(),future_indus_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,last_targets_3d,future_indus_targets,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        target_top_match_total = {}
        # 遍历按日期进行评估
        for i in range(target_class_3d.shape[0]):
            target_info_list = target_info_3d[i]
            target_class_list = target_class_3d[i]
            # 有一些空值，找出对应索引后续做忽略处理
            keep_index = np.where(target_class_list>=0)[0]
            sv_list = [[output_3d[1][k][j][i] for j in range(output_3d[2].shape[1])] for k in range(len(self.past_split))]
            output_list = [output_3d[0][i][keep_index],sv_list,output_3d[2][i]]
            last_target_list = last_targets_3d[i]
            indus_targets = future_indus_targets[i]
            date = target_info_list[np.where(target_class_list>-1)[0][0]]["future_start_datetime"]
            # 生成目标索引
            import_index,indus_top_index = self.build_import_index(output_data=output_list,target_info_list=target_info_list[keep_index])
            import_index = np.intersect1d(import_index,keep_index)
            # Compute Acc Result
            import_acc, import_recall,import_price_acc,import_price_nag,price_class, \
                import_price_result = self.collect_result(import_index, target_class_list, target_info_list)
            score_arr = []
            score_total = 0
            rate_total[date] = []
            if import_price_result is not None:
                res_group = import_price_result.groupby("result")
                ins_unique = res_group.nunique()
                total_cnt = ins_unique.values[:,1].sum()
                for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                    cnt_values = ins_unique[ins_unique.index==i].values
                    if cnt_values.shape[0]==0:
                        cnt = 0
                    else:
                        cnt = cnt_values[0,1].item()
                    rate_total[date].append(cnt)
                # 预测数量以及总数量
                rate_total[date].append(total_cnt.item())    
                # 追加计算目标值排序靠前的命中率，忽略缺失值
                ind_score,total_ins_score = self.compute_top_map(output_list,indus_targets,last_target_list,keep_index=keep_index)
                target_top_match_total[date] = (ind_score,total_ins_score)
                
        target_top_match_total = np.array(list(target_top_match_total.values()))
        
        return rate_total,total_imp_cnt,target_top_match_total
    
    def compute_top_map(self,output_list,indus_targets,last_target_list,keep_index=None):
        """分别计算行业分类排序准确率和分类内股票排序平均精度"""
        
        industry_topk = 10
        instrument_topk = 5
        acc = []
        # 排序正反参数
        sort_flag = [1,1,-1]
        sv_list,cls_list = output_list[1],output_list[2]
        # 计算行业分类总体查准率（AP）
        ind_score = [0 for _ in range(last_target_list.shape[-1])]
        for i in range(last_target_list.shape[-1]):
            ind_score[i] = compute_average_precision(cls_list[:,i],indus_targets,topk=industry_topk)
        # 分别计算每个分类下股票的查准率（AP）
        ins_scores = [[] for _ in range(last_target_list.shape[-1])]
        sw_ins_mappings = self.valid_sw_ins_mappings
        for i in range(last_target_list.shape[-1]):
            sv = sv_list[i]
            for j in range(len(sw_ins_mappings)):
                sv_item = sv[j]
                ins_arr = IndustryMappingUtil.get_instruments_in_industry(sw_ins_mappings,j)
                ins_data = last_target_list[ins_arr,0,i]
                ins_score = compute_average_precision(sv_item,ins_data,topk=instrument_topk)
                ins_scores[i].append(ins_score)
        total_ins_score = [np.array(ins_scores[i]).mean() for i in range(last_target_list.shape[-1])]
        return ind_score,total_ins_score
            
    def combine_output_total(self,output_result):
        """重载父类方法，以适应整合数据"""
        
        target_class_total = []
        target_info_total = []
        past_target_total = []   
        future_target_total = []  
        last_targets_total = []    
        indus_targets_total = []
        x_bar_total = []
        sv_total = [[] for _ in range(len(self.past_split))]
        cls_total = []
        for item in output_result:
            (output,past_target,future_target,target_class,last_targets,indus_targets,target_info) = item
            x_bar_inner = []
            sv_inner = []
            cls_inner = []
            for i in range(len(self.past_split)):
                output_item = output[i]
                x_bar,sv_instru,sv_indus = output_item 
                x_bar_inner.append(x_bar.cpu().numpy().squeeze(-1))
                sv_item = [item.squeeze(-1).cpu().numpy() for item in sv_instru]
                # 由于长度不一致，所以只能从里面取数对齐
                if len(sv_total[i])==0:
                    sv_total[i] = sv_item
                else:
                    for j in range(len(sv_item)):
                        sv_total[i][j] = np.concatenate([sv_total[i][j],sv_item[j]],axis=0)
                cls_inner.append(sv_indus.cpu().numpy().squeeze(-1))
            x_bar_inner = np.stack(x_bar_inner).transpose(1,2,3,0)
            cls_inner = np.stack(cls_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
            cls_total.append(cls_inner)
            
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            last_targets_total.append(last_targets)
            indus_targets_total.append(indus_targets)
            
        x_bar_total = np.concatenate(x_bar_total)
        cls_total = np.concatenate(cls_total)

        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        last_targets_total = np.concatenate(last_targets_total)
        indus_targets_total = np.concatenate(indus_targets_total)
        target_info_total = np.concatenate(target_info_total)
                    
        return (x_bar_total,sv_total,cls_total),past_target_total,future_target_total,target_class_total, \
                    last_targets_total,indus_targets_total,target_info_total        

    def build_import_index(self,output_data=None,target_info_list=None):  
        """生成涨幅达标的预测数据下标"""
        
        (fea_values,sv_values,cls_values) = output_data
        
        fea_range = (fea_values[...,-1] - fea_values[...,0])       
        
        pred_import_index,indus_top_index = self.strategy_top(sv_values,fea_range,cls_values,batch_size=self.ins_dim)
        # pred_import_index = self.strategy_threhold(sv_values,(fea_0_range,fea_1_range,fea_2_range),rank_values,batch_size=self.ins_dim)
        
            
        return pred_import_index,indus_top_index
        
    def strategy_threhold(self,sv,fea,cls,batch_size=0):
        sv_0 = sv[...,0]
        # 使用回归模式，则找出接近或大于目标值的数据
        sv_import_bool = (sv_0<-0.1) # & (sv_1<-0.02) #  & (sv_2>0.1)
        pred_import_index = np.where(sv_import_bool)[0]
        return pred_import_index

    def strategy_top(self,sv,fea,cls,batch_size=0):
        """排名方式筛选候选者"""
        
        sw_ins_mappings = self.valid_sw_ins_mappings
        
        cls_0 = cls[...,0]
        cls_1 = cls[...,1]
        sv_0 = sv[0]
        sv_1 = sv[1]
        
        indus_top_k = 6
        ins_top_k = 6
        # 先查找排名靠前的行业
        indus_top_index = np.argsort(cls_0)[:indus_top_k]
        # 然后从行业中找到排名靠前的股票
        ins_top_index = []
        for idx in indus_top_index:
            ins_arr = sv_0[idx]
            match_index = np.argsort(ins_arr)[:ins_top_k]
            ins_top_index.append(IndustryMappingUtil.get_instrument_with_industry(sw_ins_mappings,idx,match_index))
        ins_top_index = np.concatenate(ins_top_index)
        pred_import_index = ins_top_index 
        return pred_import_index,indus_top_index
           
        