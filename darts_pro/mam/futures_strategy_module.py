import os

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tsaug

from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from pytorch_lightning.trainer.states import RunningStage
from sklearn.preprocessing import MinMaxScaler

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from darts_pro.act_model.mixer_fur_ts import FurTimeMixer,FurStrategy
from losses.mixer_loss import FuturesCombineLoss,FuturesStrategyLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from tft.class_define import CLASS_SIMPLE_VALUES,OVERROLL_TREND_UNKNOWN,OVERROLL_TREND_RAISE,OVERROLL_TREND_FALL2RAISE,OVERROLL_TREND_RAISE2FALL,OVERROLL_TREND_FALL
from cus_utils.tensor_viz import TensorViz
from .futures_module import FuturesTogeModule,TRACK_DATE


class FuturesStrategyModule(FuturesTogeModule):
    """预测加策略选取"""              

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
                _
            ) = self.train_sample
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariate[...,past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            
            # 过去协变量维度计算,不使用时间协变量
            input_dim = (
                past_target_shape
                + past_covariates_shape
            )
           
            # 使用混合时间序列模型
            model = FurTimeMixer(
                num_nodes=self.indus_dim, # 对应多变量数量（期货品种数量）
                seq_len=self.input_chunk_length,
                pred_len=self.output_chunk_length,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
                train_sw_ins_mappings=self.train_sw_ins_mappings,
                valid_sw_ins_mappings=self.valid_sw_ins_mappings,                
            )           

            return model

    def create_loss(self,model,device="cpu"):
        return FuturesStrategyLoss(self.indus_dim,device=device,ref_model=model,lock_epoch_num=self.lock_epoch_num,target_mode=self.target_mode)       
    

    def _construct_classify_layer(self, input_dim,output_dim,device=None):
        """新增策略选择模型"""
        
        self.lock_epoch_num = 100
        
        self.top_num = 5 # 选取目标数量
        self.select_num = 10 # 一次筛选的数量
        self.trend_threhold = 0.55 # 整体趋势阈值
        self.past_tar_dim = 3 # 参考过去数值的时间段
        
        strategy_model = FurStrategy(features_dim=self.indus_dim,target_num=len(self.past_split),
                                     select_num=self.select_num,trend_threhold=self.trend_threhold)        
        return strategy_model
    
        
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """在原来基础上，添加策略选择模式"""
        
        out_total = []
        out_class_total = []
        batch_size = x_in[1].shape[0]
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        sub_model_length = len(self.sub_models) 
        vr_class = (torch.ones([batch_size,self.select_num]).to(self.device),
                    torch.ones([batch_size]).long().to(self.device),
                    torch.ones([batch_size,self.select_num]).long().to(self.device))
        past_price_target = x_in[4][...,:self.input_chunk_length] 
        past_index_targets = x_in[5]
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]  
            # 使用指标整体数据作为输入部分  
            past_index_round_targets = past_index_targets[...,i]
            # 根据优化器编号匹配计算,当编号超出模型数量时，也需要全部进行向前传播，此时没有梯度回传，主要用于生成二次模型输入数据
            if optimizer_idx==i or optimizer_idx>=sub_model_length or optimizer_idx==-1:
                x_in_item = (past_convs_item,x_in[1],x_in[2],past_index_round_targets)
                out = m(x_in_item)
                out_class = torch.ones([batch_size,self.output_chunk_length,1]).to(self.device)
            else:
                # 模拟数据
                out = (None,
                       torch.ones([batch_size,past_convs_item.shape[1],sub_model_length]).to(self.device),
                       None
                       )
                out_class = torch.ones([batch_size,1]).to(self.device)
            out_total.append(out)    
            out_class_total.append(out_class)
        
        sv_total = []
        # 使用前一个模型的部分输出，作为二次模型的输入
        for i in range(sub_model_length):
            _,sv,_ = out_total[i]
            sv_total.append(sv)
        sv_total = torch.concat(sv_total,dim=-1)
        # 拆解为2类输出
        x1 = sv_total[...,0]
        x2 = sv_total[...,1]
        # 只针对品种进行二次计算
        x1 = x1[:,instrument_index]
        x2 = x2[:,instrument_index]
        if self.current_epoch>=self.lock_epoch_num:
            # 当优化器编号大于子模型数量后，就可以进行二次模型分析了
            if optimizer_idx==sub_model_length or optimizer_idx==-1:
                # 拼接过去参考数值，包含CCI目标数据和价格数据
                past_ref_target =  x_in[5][...,1][...,-self.past_tar_dim:]
                past_price_target = past_price_target[...,-self.past_tar_dim:]
                past_targets = torch.cat([past_price_target,past_ref_target],dim=-1)
                past_targets = past_targets[:,instrument_index,:]
                vr_class = self.classify_vr_layer(x1,x2,past_targets,ignore_next=False)
        else:
            vr_class = self.classify_vr_layer(x1,x2,None,ignore_next=True)
        
        return out_total,vr_class,out_class_total  

    def get_optimizer_size(self):
        return len(self.past_split)+1
       
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
            future_round_targets,
            last_targets,
            target_info
        ) = train_batch
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,last_targets)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(self.get_optimizer_size()):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), 
                            (future_target,target_class,future_round_targets,last_targets,price_targets),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            if i==(self.get_optimizer_size()-1):
                pass
                self.log("train_fds_loss", fds_loss, batch_size=train_batch[0].shape[0], prog_bar=False)  
            else:
                self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            self.loss_data.append(detail_loss)
            total_loss += loss     
            # 手动更新参数
            opt = self.trainer.optimizers[i]
            opt.zero_grad()
            # 前面的轮次，只更新主网络
            if self.current_epoch<self.lock_epoch_num:
                if i<(self.get_optimizer_size()-1):
                    self.manual_backward(loss)
                    opt.step()
                    self.lr_schedulers()[i].step()    
            else:
                self.manual_backward(loss)
                opt.step()
                self.lr_schedulers()[i].step()                  
                                       
        self.log("train_loss", total_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr",self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)  
        self.log("lr_last",self.trainer.optimizers[-2].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=False)  
        
        # 手动维护global_step变量  
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()
        return total_loss,detail_loss,output 

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
            future_round_targets,
            last_targets,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,last_targets) 
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,future_round_targets,last_targets,price_targets),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(self.past_split)):
            # self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_fds_loss", fds_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,vr_class,price_targets,future_round_targets)
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
            price_targets,
            index_round_targets
        ) = input_batch
        dim_variable = -1

        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]
            # 修改协变量生成模式，只取自相关目标作为协变量，不使用时间协变量（时间协变量不进行归一化，只用于EMB嵌入）
            conv_defs = [
                        past_target[...,i:i+1],
                        past_covariates_item,
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
        past_index_targets = index_round_targets[:,:self.input_chunk_length,:]
        
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, historic_future_covariates,future_covariates, static_covariates,price_targets,past_index_targets)
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,future_round_targets,index_round_targets,price_targets) = target   
        future_index_round_target = index_round_targets[:,self.input_chunk_length,:]
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,future_index_round_target,price_targets[...,self.input_chunk_length-1:]),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,top_num=self.top_num,epoch_num=self.current_epoch)        
        
    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,vr_class,price_targets,future_round_targets = outputs
        choice_out,trend_value,combine_index = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,last_targets,target_info) = val_batch
        # 保存数据用于后续验证
        output_res = (output,choice_out.cpu().numpy(),trend_value.cpu().numpy(),combine_index.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),future_round_targets.cpu().numpy(),last_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_output_total(self,output_result):
        """重载父类方法，以适应整合数据"""
        
        target_class_total = []
        target_info_total = []
        past_target_total = []   
        future_target_total = []  
        price_targets_total = []    
        future_round_targets_total = []
        x_bar_total = []
        sv_total = [[] for _ in range(len(self.past_split))]
        cls_total = []
        ce_index_total = []
        choice_total = []
        trend_total = []
        combine_index_total = []
        last_targets_total = []
        for item in output_result:
            (output,choice,trend_value,combine_index,past_target,future_target,target_class,price_targets,future_round_targets,last_targets,target_info) = item
            x_bar_inner = []
            sv_inner = []
            cls_inner = []
            ce_index_inner = []
            for i in range(len(self.past_split)):
                output_item = output[i]
                x_bar,sv_indus,ce_index = output_item 
                x_bar_inner.append(x_bar.cpu().numpy())
                cls_inner.append(sv_indus.cpu().numpy().squeeze(-1))
                ce_index_inner.append(ce_index.cpu().numpy())
                
            x_bar_inner = np.stack(x_bar_inner).transpose(1,2,3,0)
            cls_inner = np.stack(cls_inner).transpose(1,2,0)
            ce_index_inner = np.stack(ce_index_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
            cls_total.append(cls_inner)
            ce_index_total.append(ce_index_inner)
            choice_total.append(choice)
            trend_total.append(trend_value)
            combine_index_total.append(combine_index)
            
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            price_targets_total.append(price_targets)
            future_round_targets_total.append(future_round_targets)
            last_targets_total.append(last_targets)
            
        x_bar_total = np.concatenate(x_bar_total)
        cls_total = np.concatenate(cls_total)
        ce_index_total = np.concatenate(ce_index_total)
        choice_total = np.concatenate(choice_total)
        trend_total = np.concatenate(trend_total)
        combine_index_total = np.concatenate(combine_index_total)
        
        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        price_targets_total = np.concatenate(price_targets_total)
        future_round_targets_total = np.concatenate(future_round_targets_total)
        last_targets_total = np.concatenate(last_targets_total)
        target_info_total = np.concatenate(target_info_total)
                    
        return (x_bar_total,sv_total,cls_total,ce_index_total,choice_total,trend_total,combine_index_total), \
                    past_target_total,future_target_total,target_class_total, \
                    price_targets_total,future_round_targets_total,last_targets_total,target_info_total        
                           
    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_3d,_,_,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        target_top_match_total = {}
        err_date_list = {}
        
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        # 遍历按日期进行评估
        for i in range(target_class_3d.shape[0]):
            future_target = future_target_3d[i]
            past_target = past_target_3d[i]
            whole_target = np.concatenate([past_target,future_target],axis=1)
            target_info_list = target_info_3d[i]
            target_class_list = target_class_3d[i]
            # 有一些空值，找出对应索引后续做忽略处理
            keep_index = np.where(target_class_list>=0)[0]
            sub_len = output_3d[4][i].shape[0]
            # 去除指数整体及行业
            keep_index = np.intersect1d(keep_index,instrument_index)   
            output_list = [output_3d[2][i],output_3d[3][i],output_3d[4][i],output_3d[5][i],output_3d[6][i]]
            price_target_list = price_targets_3d[i]
            date = target_info_list[np.where(target_class_list>-1)[0][0]]["future_start_datetime"]
            if not date in TRACK_DATE:
                continue
            # 生成目标索引
            import_index,overroll_trend = self.build_import_index(output_data=output_list,
                            target=whole_target,price_target=price_target_list,
                            combine_instrument=combine_content,instrument_index=instrument_index)
            # 有可能没有候选数据
            if import_index is None or import_index.shape[0]==0:
                continue
            
            # 输出的是品种目标相对索引，这里转化为实际索引,并忽略无效索引
            import_index = instrument_index[import_index]
            import_index = np.intersect1d(keep_index,import_index)  
            # Compute Acc Result
            import_price_result = self.collect_result(import_index,target_class=target_class_list, target_info=target_info_list)
            rate_total[date] = []
            if import_price_result is not None:
                result_values = import_price_result["result"].values
                # 如果空头预测，则倒序匹配准确率统计
                if overroll_trend==0:
                    result_values_inverse = result_values.copy()
                    result_values_inverse[np.where(result_values==0)[0]] = 3       
                    result_values_inverse[np.where(result_values==3)[0]] = 0     
                    result_values_inverse[np.where(result_values==1)[0]] = 2       
                    result_values_inverse[np.where(result_values==2)[0]] = 1    
                    result_values = result_values_inverse                            
                suc_cnt = np.sum(result_values>=2)
                if suc_cnt==0 or True:
                    err_date_list["{}_{}".format(date,suc_cnt)] = import_price_result[result_values<2][["instrument","result"]].values
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
                # 添加多空判断预测信息 
                rate_total[date].append(overroll_trend)   
        print("result:",err_date_list)      
        
        return rate_total,total_imp_cnt

    def build_import_index(self,output_data=None,target=None,price_target=None,combine_instrument=None,instrument_index=None):  
        """生成涨幅达标的预测数据下标"""
        
        # return None,None
    
        (cls_values,ce_values,choice,trend_value,combine_index) = output_data
        
        # pred_import_index,overroll_trend = self.strategy_top(cls_values,choice,trend_value,combine_index,target=target,price_array=price_array,target_info=target_info_list)
        pred_import_index,overroll_trend = self.strategy_top_direct(cls_values,ce_values,target=target,
                                price_array=price_target,instrument_index=instrument_index,combine_instrument=combine_instrument)
            
        return pred_import_index,overroll_trend
                                          
    def strategy_top(self,cls,choice,trend_value,combine_index,target=None,price_array=None,target_info=None):
        """排名方式筛选候选者"""

        future_target = target[:,self.input_chunk_length:,:]
        past_target = target[:,:self.input_chunk_length,:]
        price_recent = price_array[:,:self.input_chunk_length]
        price_recent = MinMaxScaler().fit_transform(price_recent.transpose(1,0)).transpose(1,0)      
        rsi_past_target = past_target[...,0]
        rsv_past_target = past_target[...,1]
                
        cls_rsi = cls[...,0]
        cls_price = cls[...,1]
        # cls_3 = cls[...,3]

        # 根据策略网络输出，看多或看空的不同情况，进行正向或反向排序取得相关记录索引
        if trend_value:
            index = np.argsort(-choice)[:self.top_num]
        else:
            index = np.argsort(choice)[:self.top_num]
        pred_import_index = combine_index[index]  

    def strategy_top_direct(self,cls_ori,ce_values,target=None,price_array=None,instrument_index=None,combine_instrument=None):
        """排名方式筛选候选者"""

        past_target = target[instrument_index,:self.input_chunk_length,:]
        past_price = price_array[instrument_index,:self.input_chunk_length]
        # 使用最近价格前值进行估算   
        price_recent = past_price[:,-8:]
        price_recent_range = price_recent[:,-1] - price_recent[:,0]
        # 目标前值
        rsv_past_target = past_target[instrument_index,:,0]
        rsv_recent_range = rsv_past_target[:,-1] - rsv_past_target[:,-5]
        qtlu_past_target = past_target[instrument_index,:,1]
        cls = cls_ori[instrument_index]
        cls_rsv = cls[...,0]
        # cls_qtlu = cls[...,1]
        # cls_ma = cls[...,2]
        ce_rsv = ce_values[...,1]
        rsv_mean = cls_rsv.mean()
        
        rsv_mean_threhold = 0.6
        rsv_range_threhold = 0.15
        rsv_recent_threhold = 0.8
        qtlu_threhold = 0.75
        last_price_thredhold = 0.9
        qtlu_mean_threhold = 0.3
        
        top_num = 3
        select_num = 1
        trend_value = 0
        
        # 根据策略网络输出，看多或看空的不同情况，进行正向或反向排序取得相关记录索引
        if ce_rsv>rsv_mean_threhold:
            # RSV和QTLU总体大于阈值，则看RSV指标的多方
            pre_index = np.argsort(-cls_rsv)[:top_num]  
            # 取得RSV排序靠前的记录，从而进行多方判断
            pred_import_index = []
            for index in pre_index:
                # 如果价格前值上升过大，或者价格最近值过高，则忽略
                if self.continus_trend_judge(price_recent[index], trend_type=0)==1 \
                        or price_recent[index,-1]<last_price_thredhold \
                        or rsv_past_target[index,-1]<-last_price_thredhold :
                    continue
                pred_import_index.append(index)            
            trend_value = 1              
        # elif qtlu_mean>qtlu_threhold:
        #     # RSV小，QTLU大，看QTLU排序靠前的空方
        #     pre_index = np.argsort(-cls_qtlu)[:top_num]  
        #     pred_import_index = []
        #     for index in pre_index:
        #         # 如果价格前值上升过大，或者价格最近值过高，则忽略.或者rsv指标近期升幅过大，则忽略
        #         if self.continus_trend_judge(price_recent[index], trend_type=1)==1 \
        #                 or price_recent[index,-1]>last_price_thredhold \
        #                 or ((rsv_past_target[index,-1]-rsv_past_target[index,-4])>rsv_range_threhold and rsv_past_target[index,-1]>rsv_recent_threhold):
        #             continue
        #         pred_import_index.append(index)            
        #     trend_value = 0  
        else:
            # 取得RSV反向排序靠前的记录，从而进行空方判断
            pre_index = np.argsort(cls_rsv)[:top_num]       
            pred_import_index = []
            for index in pre_index:
                # 如果价格前值上升过大，或者价格近期值过高，则忽略.或者rsv指标近期升幅过大，则忽略
                if self.continus_trend_judge(price_recent[index], trend_type=1)==1: 
                        # or ((rsv_past_target[index,-1]-rsv_past_target[index,-4])>rsv_range_threhold and rsv_past_target[index,-1]>rsv_recent_threhold):
                    continue
                pred_import_index.append(index)
            pred_import_index = np.array(pred_import_index)
            
            # 如果没有候选者，则使用STD指标作为空方指标
            if pred_import_index.shape[0]==0:
                pred_import_index = []
            
            trend_value = 0
        pred_import_index = np.array(pred_import_index)
        pred_import_index = pred_import_index[:select_num]
        
        return pred_import_index,trend_value

    def continus_trend_judge(self,target_data,trend_type=1):
        """连续趋势判断--在某段时间内连续涨或跌的趋势
            参数： 
                trend_type： 0 判断下跌 1 判断上涨
            返回值: 0 非连续 1 连续
        """
        
        total_threhold = 0.5
        part_threhold = 0.25
        last_price_thredhold = 0.9
        
        total_range = target_data[-1]-target_data[0]
        range_data = target_data[1:] - target_data[:-1]
        
        if trend_type==1:
            # 总体幅度判断
            if total_range>total_threhold:
                return 1
            # 计算趋势上涨或下跌占比，来判断是否连续趋势
            total_range_num = np.sum(range_data>0)
            part_range_num = np.sum(range_data[-5:]>0)
            # 上涨数量多，并且幅度超出局部阈值
            if (range_data[-1]>part_threhold):
                return 1
            # 最近一段上涨幅度超出局部阈值
            if (total_range_num>=(target_data.shape[0]-2)) and (total_range>part_threhold):
                return 1     
            # 如果总体上涨一定幅度，并且近期大多数上涨
            if (part_range_num>=4) and (total_range>part_threhold):
                return 1                   
            # 最近几段上涨过快超出阈值，则判断上涨趋势  
            if (range_data[-3:].sum()>part_threhold):
                return 1
            # 近期价格曾经处于较高位置，则判断上涨趋势     
            if target_data[-4:].max()>last_price_thredhold:
                return 1       
                          

        if trend_type==0:
            # 总体幅度判断
            if total_range<-total_threhold:
                return 1
            # 计算趋势上涨或下跌占比，来判断是否连续趋势
            total_range_num = np.sum(range_data<0)
            part_range_num = np.sum(range_data[-5:]<0)
            # 上涨数量多，并且幅度超出局部阈值
            if (range_data[-1]>part_threhold):
                return 1
            # 最近一段下跌幅度超出局部阈值
            if (total_range_num>=(target_data.shape[0]-2)) and (total_range<-part_threhold):
                return 1            
            # 如果总体下跌一定幅度，并且近期大多数下跌
            if (part_range_num>=4) and (total_range<-part_threhold):
                return 1
        return 0  
        