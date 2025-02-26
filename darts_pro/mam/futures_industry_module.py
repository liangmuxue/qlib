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

from .mlp_module import MlpModule
import cus_utils.global_var as global_var
from darts_pro.act_model.fur_industry_ts import FurIndustryMixer,FurStrategy
from losses.mixer_loss import FuturesIndustryLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil

from cus_utils.common_compute import compute_price_class
from tft.class_define import CLASS_SIMPLE_VALUES,get_simple_class
from .futures_module import TRACK_DATE
from cus_utils.tensor_viz import TensorViz

TRACK_DATE = [20221010,20221011,20220518,20220718,20220811,20220810,20220923]
# TRACK_DATE = [20220517,20220519]
INDEX_ITEM = 2
DRAW_SEQ = [2]
DRAW_SEQ_ITEM = [1]
DRAW_SEQ_DETAIL = [0]

class FuturesIndustryModule(MlpModule):
    """整合行业板块的总体模型"""              

    def __init__(
        self,
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
        cut_len=2,
        use_weighted_loss_func=False,
        past_split=None,
        target_mode=None,
        scale_mode=None,
        batch_file_path=None,
        device="cpu",
        train_sw_ins_mappings=None,
        valid_sw_ins_mappings=None,
        **kwargs,
    ):
        self.mode = None
        self.train_sw_ins_mappings = train_sw_ins_mappings
        self.valid_sw_ins_mappings = valid_sw_ins_mappings
        self.target_mode = target_mode
        self.scale_mode = scale_mode
        self.cut_len = cut_len
        
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
           
            combine_nodes = FuturesMappingUtil.get_industry_instrument(self.train_sw_ins_mappings,without_main=True)
            combine_nodes_num = np.array([ins.shape[0] for ins in combine_nodes])
            combine_nodes_num = torch.Tensor(combine_nodes_num).int().to(self.device)
            instrument_index = combine_nodes
            industry_index = FuturesMappingUtil.get_industry_data_index_without_main(self.train_sw_ins_mappings)
            index_num = combine_nodes_num.shape[0]
            # 加入短期指标
            if self.target_mode[seq] in [0,1,3]:
                pred_len = self.output_chunk_length
            if self.target_mode[seq]==2:
                pred_len = self.cut_len     
            if self.target_mode[seq]==5:
                pred_len = self.output_chunk_length    
                index_num = 1                          
            # 使用混合时间序列模型
            model = FurIndustryMixer(
                combine_nodes_num=combine_nodes_num, # 对应不同行业板块的期货品种数量
                index_num=index_num,
                instrument_index=instrument_index,
                industry_index=industry_index,
                seq_len=self.input_chunk_length,
                pred_len=pred_len,
                down_sampling_window=pred_len,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
            )           

            return model

    def create_loss(self,model,device="cpu"):
        return FuturesIndustryLoss(device=device,ref_model=model,lock_epoch_num=self.lock_epoch_num,target_mode=self.target_mode)       
    

    def _construct_classify_layer(self, input_dim,output_dim,device=None):
        """新增策略选择模型"""
        
        self.lock_epoch_num = 180
        
        self.top_num = 5 # 选取目标数量
        self.select_num = 10 # 一次筛选的数量
        self.trend_threhold = 0.55 # 整体趋势阈值
        self.past_tar_dim = 3 # 参考过去数值的时间段
        
        strategy_model = FurStrategy(target_num=len(self.past_split),select_num=self.select_num,trend_threhold=self.trend_threhold)        
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
        past_index_targets = x_in[-1]
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]  
            # 使用指标整体数据作为输入部分  
            past_index_round_targets = past_index_targets[...,i]
            past_round_targets = x_in[5][...,i]
            futures_convs = x_in[2]
            # 根据优化器编号匹配计算,当编号超出模型数量时，也需要全部进行向前传播，此时没有梯度回传，主要用于生成二次模型输入数据
            if optimizer_idx==i or optimizer_idx>=sub_model_length or optimizer_idx==-1:
                if self.target_mode[i]==2:
                    # 如果是短期预测模式，则裁剪对应的未来协变量,并使用短期整体过去变量
                    futures_convs = futures_convs[:,:,:self.cut_len,:]
                    past_index_round_targets = past_index_round_targets[:,:,:self.input_chunk_length-self.cut_len]
                if self.target_mode[i]==1 or self.target_mode[i]==3:
                    past_index_round_targets = past_index_round_targets[:,:,:self.input_chunk_length-self.output_chunk_length]                
                x_in_item = (past_convs_item,x_in[1],futures_convs,past_round_targets,past_index_round_targets)
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
        
        # sv_total = []
        # # 使用前一个模型的部分输出，作为二次模型的输入
        # for i in range(sub_model_length):
        #     _,sv,_ = out_total[i]
        #     sv_total.append(sv)
        # sv_total = torch.concat(sv_total,dim=-1)
        # # 拆解为2类输出
        # x1 = sv_total[...,0]
        # x2 = sv_total[...,1]
        # 只针对品种进行二次计算
        # x1 = x1[:,instrument_index]
        # x2 = x2[:,instrument_index]
        # if self.current_epoch>=self.lock_epoch_num:
        #     # 当优化器编号大于子模型数量后，就可以进行二次模型分析了
        #     if optimizer_idx==sub_model_length or optimizer_idx==-1:
        #         # 拼接过去参考数值，包含CCI目标数据和价格数据
        #         past_ref_target =  x_in[5][...,1][...,-self.past_tar_dim:]
        #         past_price_target = past_price_target[...,-self.past_tar_dim:]
        #         past_targets = torch.cat([past_price_target,past_ref_target],dim=-1)
        #         past_targets = past_targets[:,instrument_index,:]
        #         vr_class = self.classify_vr_layer(x1,x2,past_targets,ignore_next=False)
        # else:
        #     vr_class = self.classify_vr_layer(x1,x2,None,ignore_next=True)
        
        return out_total,vr_class,out_class_total  

    def on_train_epoch_start(self):  
        self.loss_data = []
        
    def on_train_epoch_end(self):  
        pass
        
    def on_validation_epoch_start(self):  
        self.import_price_result = None
        self.total_imp_cnt = 0
    
    def get_optimizer_size(self):
        return len(self.past_split) + 1
       
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
            target_info
        ) = train_batch
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,past_future_round_targets,index_round_targets)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        for i in range(self.get_optimizer_size()):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), 
                            (future_target,target_class,past_future_round_targets,index_round_targets,price_targets),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            if i==(self.get_optimizer_size()-1):
                pass
                # self.log("train_fds_loss", fds_loss, batch_size=train_batch[0].shape[0], prog_bar=False)  
            else:
                # self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            self.loss_data.append((corr_loss_combine.detach(),ce_loss.detach(),fds_loss.detach(),cls_loss.detach()))
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
            past_future_round_targets,
            index_round_targets,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,past_future_round_targets,index_round_targets) 
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,past_future_round_targets,index_round_targets,price_targets),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(self.past_split)):
            # self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_fds_loss_{}".format(i), fds_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,vr_class,price_targets,past_future_round_targets)
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
            past_future_round_targets,
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
        # 切分出过去整体round数值,规则为全部过去数值-冗余值(预测长度)-1l
        past_index_targets = index_round_targets[:,:,:-2,:]
        # 去掉正泰zsall部分
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        past_index_targets = past_index_targets[:,indus_rel_index,:,:]
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:-1,:]
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, historic_future_covariates,future_covariates, static_covariates,price_targets,past_round_targets,past_index_targets)
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,past_future_round_targets,index_round_targets,price_targets) = target 
        future_round_targets = past_future_round_targets[:,:,-1,:]  
        future_index_round_target = index_round_targets[:,:,-1,:]
        short_future_index_round_target = index_round_targets[:,:,-2,:]
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,future_index_round_target,short_future_index_round_target),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,top_num=self.top_num,epoch_num=self.current_epoch)        


    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        
        rate_total,total_imp_cnt,indus_result = self.combine_result_data(self.output_result) 
        rate_total = dict(sorted(rate_total.items(), key=lambda x:x[0]))
        indus_result_list = np.array(list(indus_result.values()))
        sr = []
        for item in list(rate_total.values()):
            if len(item)==0:
                continue
            item = np.array(item)
            sr.append(item)
                   
        if len(sr)>0:
            sr = np.stack(sr)  
            # 汇总计算准确率,取平均数
            sum_v = sr[:,-2]
            sr_rate = sr/sum_v[:, np.newaxis]
            combine_rate = np.mean(sr_rate,axis=0)
            for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                self.log("score_{} rate".format(i), combine_rate[i], prog_bar=True) 
            for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                indus_res = np.sum(indus_result_list==i)/indus_result_list.shape[0]
                self.log("indus_{} rate".format(i), indus_res, prog_bar=True) 
                # self.log("score_{} min rate".format(i), combine_rate_min[i], prog_bar=True) 
            self.log("total cnt", sr[:,-2].sum(), prog_bar=True)  
            self.log("total_imp_cnt", total_imp_cnt, prog_bar=True)  

        # 如果是测试模式，则在此进行可视化
        if self.mode is not None and self.mode.startswith("pred_") :
            tar_viz = global_var.get_value("viz_data")
            viz_result = global_var.get_value("viz_result")
            # viz_result_detail = global_var.get_value("viz_result_detail")
            
            viz_total_size = 0
                          
            output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_total, \
                past_future_round_targets_total,whole_index_round_targets_total,index_round_targets_3d,target_info_3d = self.combine_output_total(self.output_result)
            
            indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
            main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
            indus_names = FuturesMappingUtil.get_industry_names(sw_ins_mappings)
            indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)
            industry_instrument_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
            
            viz_result_detail = {}
            for code in indus_codes:
                viz_result_detail[code] = TensorViz(env="viz_result_detail_{}".format(code))
                viz_result_detail[code].remove_env()
                viz_result_detail[code] = TensorViz(env="viz_result_detail_{}".format(code))
            viz_result_detail["all"] = TensorViz(env="viz_result_detail_all")
                 
            for index in range(target_class_3d.shape[0]):
                
                viz_total_size+=1
                target_class_item = target_class_3d[index]
                keep_index = np.where(target_class_item>=0)[0]
                
                whole_index_round_targets = whole_index_round_targets_total[index]       
                round_targets = past_future_round_targets_total[index]
                index_round_targets = index_round_targets_3d[index,:,-1,:]
                infer_index_round_targets = index_round_targets_3d[index,:,-2,:]
                cls_output = output_3d[2]
                ce_output = output_3d[3]
                ts_arr = target_info_3d[index]
                date = ts_arr[keep_index][0]["future_start_datetime"]
                if not date in TRACK_DATE:
                    continue               

                ins_with_indus = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
                ins_rel_index = FuturesMappingUtil.get_instrument_rel_index(sw_ins_mappings)
                
                
                # 可视化相互关系
                for j in range(len(self.past_split)):
                    total_view_data = None
                    # 按照每个板块分类，分别显示板块内的品种整体预测数据
                    futures_names_combine = None
                    futures_index_combine = None
                    price_range_total = []
                    past_price_range_total = []
                    for k,instruments in enumerate(ins_with_indus):
                        ins_index = industry_instrument_index[k]
                        inner_class_item = target_class_item[ins_index]
                        inner_index = np.where(inner_class_item>=0)[0]           
                        instruments,k_idx,_ = np.intersect1d(instruments,keep_index,return_indices=True)
                        indus_code = indus_codes[k]
                        indus_name = indus_names[k]
                        futures_names = FuturesMappingUtil.get_futures_names(sw_ins_mappings,k)[k_idx].tolist()
                        if k==0:
                            rel_index = [0,ins_rel_index[k]]
                        else:
                            rel_index = [ins_rel_index[k-1],ins_rel_index[k-1]+ins_rel_index[k]]
                        ins_cls_output = cls_output[index,rel_index[0]:rel_index[1],j]
                        ins_cls_output = ins_cls_output[inner_index]
                        fur_round_target = round_targets[instruments,-1,j]
                        # 添加价格显示
                        price_array = np.array([ts_arr[h]["price_array"] for h in instruments])
                        scaler = MinMaxScaler(feature_range=(0.001, 1))
                        scaler.fit(price_array[:,:-self.output_chunk_length].transpose(1,0))
                        price_array_norm = scaler.transform(price_array.transpose(1,0)).transpose(1,0)
                        price_array_range = price_array_norm[:,-1] - price_array_norm[:,-self.output_chunk_length-1]     
                        past_price_array_range = price_array_norm[:,-self.output_chunk_length-1] - price_array_norm[:,self.input_chunk_length-self.output_chunk_length]  
                        price_range_total.append(price_array_range.mean())   
                        past_price_range_total.append(past_price_array_range.mean())    
                        view_data = np.stack([ins_cls_output,fur_round_target,price_array_range]).transpose(1,0)
                        win = "round_target_{}_{}_{}".format(j,k,viz_total_size)
                        target_title = "target{}_{},date:{}".format(j,indus_name,date)
                        if j in DRAW_SEQ_ITEM and len(futures_names)>1:
                            viz_result.viz_bar_compare(view_data,win=win,title=target_title,rownames=futures_names,legends=["pred","target","price"])   
                        if total_view_data is None:
                            total_view_data = view_data
                        else:
                            total_view_data = np.concatenate([total_view_data,view_data],axis=0) 
                        if futures_names_combine is None:
                            futures_names_combine = np.array(futures_names)
                            futures_index_combine = np.array(instruments)
                        else:
                            futures_names_combine = np.concatenate([futures_names_combine,futures_names],axis=0)    
                            futures_index_combine = np.concatenate([futures_index_combine,instruments],axis=0)    
                        
                        # 行业round数据取均值，画行业趋势线
                        whole_round_targets = whole_index_round_targets[k,:,j]
                        # indus_round_data = whole_round_targets.mean(axis=0)
                        indus_round_data = np.pad(whole_round_targets,((0),(2*self.output_chunk_length-1)),'constant')  
                        whole_target = np.concatenate([past_target_3d,future_target_3d],axis=2)[index,instruments,:,j].mean(axis=0)
                        # 取得行业数据，并整合到一起显示
                        indus_inner_index = indus_index[k]
                        indus_target = np.concatenate([past_target_3d,future_target_3d],axis=2)[index,indus_inner_index,:,j]
                        view_data = np.stack([indus_target,price_array_norm.mean(axis=0)]).transpose(1,0)
                        win = "whole_round_target_{}_{}_{}".format(j,k,viz_total_size)                        
                        target_title = "target_{}_{},date:{}".format(indus_name,j,date)
                        viz_detail = viz_result_detail["all"]
                        names=["target","price"]
                        if j in DRAW_SEQ_ITEM:
                            viz_detail.viz_matrix_var(view_data,win=win,title=target_title,names=names)                          

                    # 显示板块分类整体预测数据
                    indust_output_value = ce_output[j][index]
                    indust_target = index_round_targets[:-1,j]
                    price_range_total = np.array(price_range_total)
                    win = "indus_round_target_{}_{}".format(j,viz_total_size)
                    target_title = "industry compare_{},mean_tar:{}_{},date:{}".format(j,round(ce_output[INDEX_ITEM][index][0],2),round(index_round_targets[-1,INDEX_ITEM],2),date)
                    if j in DRAW_SEQ:
                        indust_output_value = indust_output_value.repeat(6)
                        view_data = np.stack([indust_output_value,indust_target,price_range_total]).transpose(1,0)
                        tar_viz.viz_bar_compare(view_data,win=win,title=target_title,rownames=indus_names.tolist()[:-1],legends=["pred","target","price"])   
                        
                    # 显示预测走势数据    
                    def show_trend_data(k,indus_code,ins_name,type=0):
                        past_target_item = past_target_3d[index,k,:,j]
                        future_target_item = future_target_3d[index,k,:,j]
                        target_data = np.concatenate([past_target_item,future_target_item],axis=0)    
                        zero_index = np.where(target_data==0)
                        target_data[zero_index] = 0.001
                        pad_data = np.array([0 for i in range(self.input_chunk_length)])
                        # Add Price data and Norm
                        price_array = ts_arr[k]["price_array"]
                        scaler = MinMaxScaler(feature_range=(0.001, 1))
                        scaler.fit(np.expand_dims(price_array[:self.input_chunk_length],-1))
                        price_array = scaler.transform(np.expand_dims(price_array,-1)).squeeze()
                        view_data = np.stack((target_data,price_array)).transpose(1,0)
                        win = "target_xbar_{}_{}_{}_{}".format(indus_code,k,j,date)
                        target_title = "{}_{},date:{}".format(ins_name,j,date)
                        if type==1:
                            viz_detail = viz_result_detail["all"]
                        else:
                            viz_detail = viz_result_detail[indus_code]
                        names=["round_target","price"]
                        # viz_detail.viz_matrix_var(view_data,win=win,title=target_title,names=names)   
                    
                    if j in DRAW_SEQ_DETAIL and date in TRACK_DATE:                            
                        # 品种趋势数据     
                        for idx in range(indus_index.shape[0]):
                            k = indus_index[idx]
                            indus_code = indus_codes[idx]
                            ins_obj = FuturesMappingUtil.get_instrument_obj_in_industry(sw_ins_mappings,idx)
                            for i in range(ins_obj.shape[0]):
                                k = int(ins_obj[i][0])
                                if not k in keep_index:
                                    continue
                                ins_name = ins_obj[i][2]
                                show_trend_data(k,indus_code,ins_name,type=0)
                    
                        # 分类及总体趋势数据       
                        for idx in range(indus_index.shape[0]):
                            k = indus_index[idx]
                            indus_name = indus_names[idx]
                            indus_code = indus_codes[idx]
                            show_trend_data(k,indus_code,indus_name,type=1)
                                    
    def dump_val_data(self,val_batch,outputs,detail_loss):
    
        output,vr_class,price_targets,past_future_round_targets = outputs
        choice_out,trend_value,combine_index = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,index_round_targets,target_info) = val_batch
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output,choice_out.cpu().numpy(),trend_value.cpu().numpy(),combine_index.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),past_future_round_targets.cpu().numpy(),whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_output_total(self,output_result):
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
        cls_total = []
        ce_index_total = [None for _ in range(len(self.past_split))]
        choice_total = []
        trend_total = []
        combine_index_total = []
        index_round_targets_total = []
        for item in output_result:
            (output,choice,trend_value,combine_index,past_target,future_target,target_class,price_targets,past_future_round_targets,whole_index_round_targets,index_round_targets,target_info) = item
            x_bar_inner = []
            sv_inner = []
            cls_inner = []
            ce_index_inner = []
            for i in range(len(self.past_split)):
                output_item = output[i]
                _,sv_indus,ce_index = output_item 
                # 合并列表中的品种维度部分
                sv_indus = torch.cat(sv_indus,dim=1).squeeze(-1)
                cls_inner.append(sv_indus.cpu().numpy())
                if ce_index_total[i] is None:
                    ce_index_total[i] = ce_index.cpu().numpy()
                else:
                    ce_index_total[i] = np.concatenate([ce_index_total[i],ce_index.cpu().numpy()],axis=0)
                
            cls_inner = np.stack(cls_inner).transpose(1,2,0)
            # ce_index_inner = np.stack(ce_index_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
            cls_total.append(cls_inner)
            choice_total.append(choice)
            trend_total.append(trend_value)
            combine_index_total.append(combine_index)
            
            target_info_total.append(target_info)
            target_class_total.append(target_class)
            past_target_total.append(past_target)
            future_target_total.append(future_target)
            price_targets_total.append(price_targets)
            past_future_round_targets_total.append(past_future_round_targets)
            whole_index_round_targets_total.append(whole_index_round_targets)
            index_round_targets_total.append(index_round_targets)
            
        x_bar_total = np.concatenate(x_bar_total)
        cls_total = np.concatenate(cls_total)
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
                    
        return (x_bar_total,sv_total,cls_total,ce_index_total,choice_total,trend_total,combine_index_total), \
                    past_target_total,future_target_total,target_class_total,price_targets_total,past_future_round_targets_total, \
                    whole_index_round_targets_total,index_round_targets_total,target_info_total        
                           
    def combine_result_data(self,output_result=None,predict_mode=False):
        """计算涨跌幅分类准确度以及相关数据"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_3d,_,_,index_round_targets_3d,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        result_date_list = {}
        indus_result_total = {}
        
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        instrument_in_indus_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        
        # 遍历按日期进行评估
        for i in range(target_class_3d.shape[0]):
            index_round_targets = index_round_targets_3d[i]
            future_target = future_target_3d[i]
            infer_index_round_targets = index_round_targets[:,-2,:]
            past_target = past_target_3d[i]
            whole_target = np.concatenate([past_target,future_target],axis=1)
            target_info_list = target_info_3d[i]
            target_class_list = target_class_3d[i]
            # 有一些空值，找出对应索引后续做忽略处理
            keep_index = np.where(target_class_list>=0)[0]
            # 去除指数整体及行业
            keep_index = np.intersect1d(keep_index,instrument_index)  
            ce_index = [item[i] for item in output_3d[3]]
            output_list = [output_3d[2][i],ce_index,output_3d[4][i],output_3d[5][i],output_3d[6][i]]
            price_target_list = price_targets_3d[i]
            date = int(target_info_list[np.where(target_class_list>-1)[0][0]]["future_start_datetime"])
            if not date in TRACK_DATE:
                continue
            # 生成目标索引
            import_index,overroll_trend,trend_value,indus_top_index = self.build_import_index(output_data=output_list,
                            target=whole_target,price_target=price_target_list,infer_index_round_targets=infer_index_round_targets,
                            combine_instrument=combine_content,instrument_index=instrument_in_indus_index)
            # 有可能没有候选数据
            if import_index is None or import_index.shape[0]==0:
                continue
            
            import_index = np.intersect1d(keep_index,import_index)  
            # 如果是预测模式，则只输出结果,不验证
            if predict_mode:
                result_date_list[date] = [import_index,overroll_trend]
                continue
            # Compute Acc Result
            import_price_result,indus_result = self.collect_result(import_index,overroll_trend=overroll_trend,
                                            indus_top_index=indus_top_index,target_info=target_info_list)
            rate_total[date] = []
            if import_price_result is not None:
                result_values = import_price_result["result"].values
                suc_cnt = np.sum(result_values>=2)
                fail_cnt = np.sum(result_values<2)
                if fail_cnt>0 or True:
                    result_date_list["{}_{}/{}_{}".format(int(date),int(overroll_trend),round(trend_value,2),suc_cnt)] = \
                        import_price_result[["instrument","result"]].values
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
                # Add Industry Result
                indus_result_total[date] = indus_result
        # print("result:",result_date_list)      

        # 如果是预测模式，则只输出结果,不验证
        if predict_mode:
            return result_date_list
        
        return rate_total,total_imp_cnt,indus_result_total

    def collect_result(self,import_index,overroll_trend=0,indus_top_index=None,target_info=None): 
        """收集预测对应的实际数据"""

        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 对于预测数据，生成对应涨跌幅类别
        import_price_result = []
        for i,imp_idx in enumerate(import_index):
            ts = target_info[imp_idx]
            price_array = ts["price_array"][self.input_chunk_length-1:]
            p_taraget_class = compute_price_class(price_array,mode="first_last")
            # 根据多空判断取得实际对应的类别
            if overroll_trend==0:
                p_taraget_class = np.array([3,2,1,0])[p_taraget_class]
            import_price_result.append([imp_idx,ts["instrument"],p_taraget_class])       
        import_price_result = np.array(import_price_result)  
        if import_price_result.shape[0]==0:
            return None
        import_price_result = pd.DataFrame(import_price_result,columns=["imp_index","instrument","result"])     
        import_price_result["result"] = import_price_result["result"].astype(np.int64)      
        
        # 同时计算行业趋势判断准确率
        ins_with_indus = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        ins_obj = FuturesMappingUtil.get_instrument_obj_in_industry(sw_ins_mappings,indus_top_index)
        price_array_range_mean = []
        for i in range(ins_obj.shape[0]):
            k = int(ins_obj[i][0])
            ins_name = ins_obj[i][2]
            ts = target_info[k]
            if ts is None:
                continue
            price_array = ts["price_array"][self.input_chunk_length-1:]    
            price_array_range = (price_array[-1] - price_array[0])/price_array[0]      
            price_array_range_mean.append(price_array_range)
        price_array_range_mean = np.array(price_array_range_mean).mean()
        p_taraget_class = get_simple_class(price_array_range_mean)      
        # 根据多空判断取得实际对应的类别
        if overroll_trend==0:
            p_taraget_class = np.array([3,2,1,0])[p_taraget_class]                  
        return import_price_result,p_taraget_class
    
    def build_import_index(self,output_data=None,target=None,price_target=None,infer_index_round_targets=None,combine_instrument=None,instrument_index=None):  
        """生成涨幅达标的预测数据下标"""
        
        # return None,None,None,None
    
        (cls_values,ce_values,choice,trend_value,combine_index) = output_data
        
        # pred_import_index,overroll_trend = self.strategy_top(cls_values,choice,trend_value,combine_index,target=target,price_array=price_array,target_info=target_info_list)
        indus_top_index,pred_import_index,overroll_trend,trend_value = self.strategy_top_direct(cls_values,ce_values,
                                    target=target,infer_index_round_targets=infer_index_round_targets,
                                price_array=price_target,instrument_index=instrument_index,combine_instrument=combine_instrument)

        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        industry_instrument_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        
        if pred_import_index is None or pred_import_index.shape[0]==0:
            return None,None,None,None
        # 输出的是品种目标相对索引，这里转化为实际索引,并忽略无效索引
        import_index = industry_instrument_index[indus_top_index][pred_import_index]     
              
        return import_index,overroll_trend,trend_value,indus_top_index
                                          
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

    def strategy_top_direct(self,cls,ce_values,target=None,price_array=None,infer_index_round_targets=None,instrument_index=None,combine_instrument=None):
        """排名方式筛选候选者"""

        # 目标前值
        # past_target = target[instrument_index,:self.input_chunk_length,:]
        # rsv_past_target = past_target[instrument_index,:,0]
        # rsv_recent_range = rsv_past_target[:,-1] - rsv_past_target[:,-5]
        cls_2 = cls[...,0]
        ce_indus = ce_values[1]
        ce_index = ce_values[2]
        ce2_index = ce_values[0]
        ce_index_mean = ce_index.mean() #ce_values[2].mean()
              
        ce_index_mean_threhold = 0.55
        ce_threhold = 0.4
        
        top_num = 3
        select_num = 3
        trend_value = 0
        
        # 取得行业板块中最高和最低的两个，并使用这2个中的一个作为目标行业板块
        raise_top_index = np.argsort(-ce_indus)[0] 
        fall_top_index = np.argsort(ce_index)[0]   
        trend_flag = ce_index.mean()>ce_index_mean_threhold
        # trend_flag = np.sum(ce_indus>ce_index_mean_threhold)<top_num
        # trend_flag = True
        # 整体预测均值和阈值比较，决定整体使用多方还是空方作为预测方向
        if trend_flag:
            trend_type = 1
            indus_top_index = raise_top_index
        else:
            trend_type = 0
            indus_top_index = fall_top_index
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        ins_rel_index = FuturesMappingUtil.get_instrument_rel_index_within_industry(sw_ins_mappings,indus_top_index)
        ins_index = instrument_index[indus_top_index]
        past_price = price_array[ins_index,:self.input_chunk_length]
        cls_can = cls_2[ins_rel_index[0]:ins_rel_index[1]]
        
        # 使用最近价格前值进行估算   
        price_recent_can = past_price[:,-8:]
        # 取得对应行业下的品种排名，并作为候选
        if trend_type==0:
            # 看CLS指标的多方
            pre_index = np.argsort(cls_can)[:top_num]  
            # 取得RSV排序靠前的记录，从而进行空方判断
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)            
            trend_value = 0              
        else:
            # 取得CLS反向排序靠前的记录，从而进行多方判断
            pre_index = np.argsort(-cls_can)[:top_num]       
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)
            trend_value = 1
        pred_import_index = np.array(pred_import_index)
        pred_import_index = pred_import_index[:select_num]
        
        return indus_top_index,pred_import_index,trend_value,ce_index_mean

    def on_predict_epoch_start(self):  
        self.output_result = []
        
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
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
            target_info
        ) = batch
               
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,past_future_round_targets,index_round_targets) 
        input_batch = self._process_input_batch(inp)
        (outputs,vr_class,out_class_total) = self(input_batch,optimizer_idx=-1)
        
        choice_out,trend_value,combine_index = vr_class        
        # 保存数据用于后续验证
        output_res = (outputs,choice_out.cpu().numpy(),trend_value.cpu().numpy(),combine_index.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),past_future_round_targets.cpu().numpy(),index_round_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)        
         
    def on_predict_epoch_end(self,args):   
        """汇总预测数据，生成实际业务预测结果"""
        
        sw_ins_mappings = self.valid_sw_ins_mappings
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        result_date_list = self.combine_result_data(self.output_result,predict_mode=True)  
        result_target = {}
        # 根据原始数组，生成实际品种信息
        for date in list(result_date_list.keys()):
            res_arr = result_date_list[date]
            res_index = res_arr[0]
            target = combine_content[np.isin(combine_content[:,0],res_index)]
            result_target[date] = [target,res_arr[1]]
        self.result_target = result_target
        
        return result_target
                         
