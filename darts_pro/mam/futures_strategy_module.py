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
from .futures_module import FuturesTogeModule

TRACK_DATE = [20220506]
DRAW_SEQ = [0,1,3]
DRAW_SEQ_DETAIL = [0]

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
        return FuturesStrategyLoss(self.indus_dim,device=device,ref_model=model)       
    

    def _construct_classify_layer(self, input_dim,output_dim,device=None):
        """新增策略选择模型"""
        
        self.top_num = 5
        strategy_model = FurStrategy(features_dim=self.indus_dim,target_num=len(self.past_split),top_num=self.top_num)        
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
        sub_model_length = len(self.sub_models) 
        vr_class = (torch.ones([batch_size,self.top_num]).to(self.device) ,torch.ones([batch_size]).to(self.device))
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]    
            round_target =  x_in[4][...,i]        
            # 根据优化器编号匹配计算,当编号超出模型数量时，也需要全部进行向前传播，此时没有梯度回传，主要用于生成二次模型输入数据
            if optimizer_idx==i or optimizer_idx>=sub_model_length or optimizer_idx==-1:
                x_in_item = (past_convs_item,x_in[1],x_in[2],round_target)
                out = m(x_in_item)
                out_class = torch.ones([batch_size,self.output_chunk_length,1]).to(self.device)
            else:
                # 模拟数据
                out = torch.ones([batch_size,self.output_chunk_length,self.output_dim[0],1]).to(self.device)
                out_class = torch.ones([batch_size,1]).to(self.device)
            out_total.append(out)    
            out_class_total.append(out_class)
        
        if self.current_epoch>=50:
            # 当优化器编号大于子模型数量后，就可以进行二次模型分析了
            if optimizer_idx==sub_model_length or optimizer_idx==-1:
                sv_total = []
                # 使用前一个模型的部分输出，作为二次模型的输入
                for i in range(sub_model_length):
                    _,sv,_ = out_total[i]
                    sv_total.append(sv)
                sv_total = torch.concat(sv_total,dim=-1)
                vr_class = self.classify_vr_layer(sv_total,sw_ins_mappings=sw_ins_mappings)
        
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
            past_round_targets,
            future_round_targets,
            target_info
        ) = train_batch
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,past_round_targets)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(self.get_optimizer_size()):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,future_round_targets,target_info),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            if i==(self.get_optimizer_size()-1):
                self.log("train_fds_loss", fds_loss, batch_size=train_batch[0].shape[0], prog_bar=False)  
            else:
                self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            self.loss_data.append(detail_loss)
            total_loss += loss     
            # 手动更新参数
            opt = self.trainer.optimizers[i]
            opt.zero_grad()
            # 前面的轮次，只更新主网络，后面的只更新策略网络
            if self.current_epoch<50:
                if i<(self.get_optimizer_size()-1):
                    self.manual_backward(loss)
                    opt.step()
                    self.lr_schedulers()[i].step()    
            else:
                self.manual_backward(loss)
                if i==(self.get_optimizer_size()-1):
                    opt.step()
                    self.lr_schedulers()[i].step()                  
                                       
        self.log("train_loss", total_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr_last",self.trainer.optimizers[-2].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)  
        
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
            past_round_targets,
            future_round_targets,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,past_round_targets) 
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,future_round_targets,target_info),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(self.past_split)):
            # self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_fds_loss", fds_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,vr_class,past_round_targets,future_round_targets)
        return loss,detail_loss,output_combine       

    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,future_round_targets,target_info) = target   
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,target_info),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,epoch_num=self.current_epoch)        
        
    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,vr_class,past_round_targets,future_round_targets = outputs
        choice,trend_value = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,target_info) = val_batch
        # 保存数据用于后续验证
        output_res = (output,choice.cpu().numpy(),trend_value.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      past_round_targets.cpu().numpy(),future_round_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_output_total(self,output_result):
        """重载父类方法，以适应整合数据"""
        
        target_class_total = []
        target_info_total = []
        past_target_total = []   
        future_target_total = []  
        past_round_targets_total = []    
        future_round_targets_total = []
        x_bar_total = []
        sv_total = [[] for _ in range(len(self.past_split))]
        cls_total = []
        ce_index_total = []
        choice_total = []
        trend_total = []
        for item in output_result:
            (output,choice,trend_value,past_target,future_target,target_class,past_round_targets,future_round_targets,target_info) = item
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
            
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            past_round_targets_total.append(past_round_targets)
            future_round_targets_total.append(future_round_targets)
            
        x_bar_total = np.concatenate(x_bar_total)
        cls_total = np.concatenate(cls_total)
        ce_index_total = np.concatenate(ce_index_total)
        choice_total = np.concatenate(choice_total)
        trend_total = np.concatenate(trend_total)
        
        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        past_round_targets_total = np.concatenate(past_round_targets_total)
        future_round_targets_total = np.concatenate(future_round_targets_total)
        target_info_total = np.concatenate(target_info_total)
                    
        return (x_bar_total,sv_total,cls_total,ce_index_total,choice_total,trend_total),past_target_total,future_target_total,target_class_total, \
                    past_round_targets_total,future_round_targets_total,target_info_total        
                           
    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,last_targets_3d,future_indus_targets,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        target_top_match_total = {}
        err_date_list = {}
        
        instrument_index = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)[:,0].astype(np.int)      
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
            output_list = [output_3d[0][i][keep_index],output_3d[2][i][keep_index],output_3d[3][i],output_3d[4][i],output_3d[5][i]]
            last_target_list = last_targets_3d[i]
            indus_targets = future_indus_targets[i]
            date = target_info_list[np.where(target_class_list>-1)[0][0]]["future_start_datetime"]
            # if not date in TRACK_DATE:
            #     continue
            # 生成目标索引
            import_index,overroll_trend = self.build_import_index(output_data=output_list,target=whole_target[keep_index],target_info_list=target_info_list[keep_index])
            # 有可能没有候选数据
            if import_index is None:
                continue
            
            # 输出的是品种目标相对索引，这里转化为实际索引,并忽略无效索引
            import_index = instrument_index[import_index]
            import_index = np.intersect1d(keep_index,import_index)  
            # Compute Acc Result
            try:
                import_price_result = self.collect_result(import_index,target_class=target_class_list, target_info=target_info_list)
            except Exception as e:
                print("eee")
            rate_total[date] = []
            if import_price_result is not None:
                result_values = import_price_result["result"].values
                # 如果空头预测，则倒序匹配准确率统计
                if overroll_trend==OVERROLL_TREND_FALL:
                    result_values_inverse = result_values.copy()
                    result_values_inverse[np.where(result_values==0)[0]] = 3       
                    result_values_inverse[np.where(result_values==3)[0]] = 0     
                    result_values_inverse[np.where(result_values==1)[0]] = 2       
                    result_values_inverse[np.where(result_values==2)[0]] = 1    
                    result_values = result_values_inverse                            
                suc_cnt = np.sum(result_values>=2)
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
        # print("result:",err_date_list)      
        
        return rate_total,total_imp_cnt

    def build_import_index(self,output_data=None,target=None,target_info_list=None):  
        """生成涨幅达标的预测数据下标"""
        
        (fea_values,cls_values,ce_values,choice,trend_value) = output_data
        price_array = np.array([item["price_array"] for item in target_info_list])
        
        
        pred_import_index,overroll_trend = self.strategy_top(choice,trend_value,fea_values,cls_values,ce_values,target=target,price_array=price_array,target_info=target_info_list)
        # pred_import_index = self.strategy_threhold(sv_values,(fea_0_range,fea_1_range,fea_2_range),rank_values,batch_size=self.ins_dim)
            
        return pred_import_index,overroll_trend
                                          
    def strategy_top(self,choice,trend_value,fea_values,cls,ce,target=None,price_array=None,target_info=None,top_num=3):
        """排名方式筛选候选者"""

        future_target = target[:,self.input_chunk_length:,:]
        past_target = target[:,:self.input_chunk_length,:]
        price_recent = price_array[:,:self.input_chunk_length]
        price_recent = MinMaxScaler().fit_transform(price_recent.transpose(1,0)).transpose(1,0)      
        rsi_past_target = past_target[...,0]
        rsv_past_target = past_target[...,1]
                
        # 排整体涨跌判断
        overroll_trend = np.argmax(trend_value)
        
        cls_rsi = cls[...,0]
        cls_price = cls[...,1]
        # cls_3 = cls[...,3]

        # 根据策略网络输出，看多或看空的不同情况，进行正向或反向排序取得相关记录索引
        if overroll_trend==1:
            pred_import_index = np.argsort(choice)[:top_num]
        elif overroll_trend==0:
            pred_import_index = np.argsort(-choice)[:top_num]
            
        return pred_import_index,overroll_trend

    def judge_overall_trend(self,fea_values,cls,ce,target=None,past_target=None,price_recent=None):
        """排整体涨跌判断,返回值：1 确定上涨 2 确定下跌 0 其他
            先判断下跌，然后判断上涨
        """
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        indus_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        # 使用CNTP5指标判断总体趋势,查看每个板块分类数据，大于指定阈值的行业数量与数量阈值进行比较
        rsi_ce = ce[:,0]
        rsi_ce = np.concatenate([rsi_ce[:main_index],rsi_ce[main_index+1:]])
        # 超过大多数行业在指标阈值之上，则认为是上涨，否则是下跌
        trend_bool = np.sum(rsi_ce>0)>=(rsi_ce.shape[0]-2)
        if not trend_bool:
            return OVERROLL_TREND_FALL
        else:
            return OVERROLL_TREND_RAISE
        
        return OVERROLL_TREND_UNKNOWN
        
    
          