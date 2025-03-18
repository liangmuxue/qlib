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
from darts_pro.act_model.fur_industry_ts import FurIndustryDRollMixer,FurStrategy
from losses.mixer_loss import FuturesIndustryDRollLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil

from cus_utils.common_compute import compute_price_class
from tft.class_define import CLASS_SIMPLE_VALUES,get_simple_class
from .futures_module import TRACK_DATE
from cus_utils.tensor_viz import TensorViz

TRACK_DATE = [20221010,20221011,20220518,20220718,20220811,20220810,20220923]
# TRACK_DATE = [20220523,20220524,20220526]
INDEX_ITEM = 0
DRAW_SEQ = [0]
DRAW_SEQ_ITEM = [0]
DRAW_SEQ_DETAIL = [0]

class FuturesIndustryDRollModule(MlpModule):
    """整合行业板块的总体模型,只包含行业不包含具体品种"""              

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
        rolling_size=18,
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
        self.rolling_size = rolling_size
        
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
            
            industry_index = [i for i in range(past_target.shape[1])]
            main_index = FuturesMappingUtil.get_main_index_in_indus(self.train_sw_ins_mappings)
            industry_index.remove(main_index)
            num_nodes = len(industry_index)
            # 加入短期指标
            if self.target_mode[seq] in [0,1]:
                pred_len = self.output_chunk_length
            if self.target_mode[seq]==2:
                pred_len = self.cut_len     
            if self.target_mode[seq]==3:
                pred_len = self.output_chunk_length    
            if self.target_mode[seq]==5:    
                pred_len = self.output_chunk_length   
            # 使用混合时间序列模型
            model = FurIndustryDRollMixer(
                num_nodes=num_nodes,
                rolling_size=self.rolling_size,
                index_num=self.rolling_size,
                industry_index=industry_index,
                main_index=main_index,
                seq_len=self.input_chunk_length,
                pred_len=pred_len,
                down_sampling_window=pred_len,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
            )           

            return model

    def create_loss(self,model,device="cpu"):
        return FuturesIndustryDRollLoss(device=device,ref_model=model,lock_epoch_num=self.lock_epoch_num,target_mode=self.target_mode)       
    

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
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:-1,:]
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, historic_future_covariates,future_covariates, static_covariates,price_targets,past_round_targets,past_index_targets)
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,past_future_round_targets,index_round_targets,price_targets) = target 
        future_round_targets = past_future_round_targets[:,:,-1,:]  
        future_index_round_target = index_round_targets[:,:,-1,:]
        # 目标值维度还原
        batch_size = int(future_target.shape[0]/self.rolling_size)
        future_target = future_target.reshape(batch_size,self.rolling_size,*future_target.shape[1:])
        target_class = target_class.reshape(batch_size,self.rolling_size,*target_class.shape[1:])
        future_round_targets = future_round_targets.reshape(batch_size,self.rolling_size,*future_round_targets.shape[1:])
        future_index_round_target = future_index_round_target.reshape(batch_size,self.rolling_size,*future_index_round_target.shape[1:])
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        future_index_round_target = future_index_round_target[:,:,main_index,:]
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,future_index_round_target),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,top_num=self.top_num,epoch_num=self.current_epoch)        


    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
    
        rate_total,pred_detail_total = self.combine_result_data(self.output_result) 
        rate_total = dict(sorted(rate_total.items(), key=lambda x:x[0]))

        # 如果是测试模式，则在此进行可视化
        if self.mode is not None and self.mode.startswith("pred_") :
            tar_viz = global_var.get_value("viz_data")
            viz_result = global_var.get_value("viz_result")
            # viz_result_detail = global_var.get_value("viz_result_detail")
            
            viz_total_size = 0
                          
            output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_total, \
                past_future_round_targets_total,whole_index_round_targets_total,index_round_targets_3d,target_info_3d = self.combine_output_total(self.output_result)

            # 还原维度后，每组按照最后一个日期进行评估
            batch_size = int(target_class_3d.shape[0]/self.rolling_size)
            target_class_3d = self.reshape_to_ori(target_class_3d,batch_size)
            future_target_3d = self.reshape_to_ori(future_target_3d,batch_size)
            past_target_3d = self.reshape_to_ori(past_target_3d,batch_size)
            index_round_targets_3d = self.reshape_to_ori(index_round_targets_3d,batch_size)
                    
            indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
            main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
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
                
                index_round_targets = index_round_targets_3d[index,:,main_index,-1,:]
                ce_output = output_3d[3]
                ts_arr = target_info_3d[index]
                date = int(ts_arr[-1][main_index]["future_start_datetime"])
                if not date in TRACK_DATE:
                    continue      
                
                trend_corr = rate_total[date]
                pred_detail = pred_detail_total[date]      
                # 可视化总体指数在不同时间段的预测值和实际值
                for j in range(len(self.past_split)):
                    # 整体指数趋势线
                    index_price_array = ts_arr[-1][main_index]["price_round_data"]
                    index_price_array_norm = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(index_price_array,-1)).squeeze(-1)   
                    ce_output_item = ce_output[j][index] 
                    index_target = index_round_targets[...,j]
                    view_data = np.stack([ce_output_item,index_target,index_price_array]).transpose(1,0)
                    win = "index_round_target_{}_{}".format(j,viz_total_size)                        
                    target_title = "[{}]:Corr:{},priceInf:{},ceMean:{},trendRes:{}".format(date,trend_corr,round(pred_detail[2],3),round(pred_detail[1],1),pred_detail[0])
                    names=["pred","target","price"]
                    if j in DRAW_SEQ_ITEM:
                        viz_result.viz_matrix_var(view_data,win=win,title=target_title,names=names)      
                                    
    def dump_val_data(self,val_batch,outputs,detail_loss):
    
        output,vr_class,price_targets,past_future_round_targets = outputs
        choice_out,trend_value,combine_index = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,index_round_targets,target_info) = val_batch
        # 记录批次内价格涨跌幅，用于整体指数批次归一化数据的回溯
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        batch_size = int(index_round_targets.shape[0]/self.rolling_size)
        index_round_targets_reshape = self.reshape_to_ori(index_round_targets, batch_size)
        # 只获取整体指数的价格数据
        price_array_total = []
        for item in target_info[:,-1]:
            price_array = item[main_index]["price_array"]
            price_array_total.append(price_array)
        price_array_total = np.array(price_array_total)
        price_round_data = (price_array_total[:,-1] - price_array_total[:,-self.output_chunk_length-1])/price_array_total[:,-self.output_chunk_length-1]
        
        # 针对每组，取得整体指标的价格差分结果数据
        for index,ts in enumerate(target_info):
            price_array = np.array([item[main_index]["price_array"] for item in ts])
            price_round_data = (price_array[:,-1] - price_array[:,-self.output_chunk_length-1])/price_array[:,-self.output_chunk_length-1]
            # 保存到targetinfo中，后续使用
            ts[-1][main_index]["price_round_data"] = price_round_data
            # 默认预测对应目标都在序列最后一个
            ts[-1][main_index]["target_round_index"] = -1
            # 最后一个维度对应预测变量，目前只预测1个变量
            ts[-1][main_index]["target_round_data"] = index_round_targets_reshape.cpu().numpy()[index,:,main_index,-1,-1]
            ts[-1][main_index]["pred_round_data"] = output[-1][2][index].cpu().numpy()
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
     
    def reshape_to_ori(self,data,batch_size=8):
        return data.reshape([batch_size,self.rolling_size,*data.shape[1:]])
                           
    def combine_result_data(self,output_result=None,predict_mode=False):
        """计算涨跌幅分类准确度以及相关数据"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
    
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_3d,_,_,index_round_targets_3d,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        pred_detail_list = {}
        
        instrument_in_indus_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        
        # 还原维度后，每组按照最后一个日期进行评估
        batch_size = int(target_class_3d.shape[0]/self.rolling_size)
        target_class_3d = self.reshape_to_ori(target_class_3d,batch_size)
        future_target_3d = self.reshape_to_ori(future_target_3d,batch_size)
        past_target_3d = self.reshape_to_ori(past_target_3d,batch_size)
        price_targets_3d = self.reshape_to_ori(price_targets_3d,batch_size)
        
        # 遍历按日期进行评估
        for i in range(batch_size):
            # 只获取最后一个时间段的数据
            future_target = future_target_3d[i][-1]
            past_target = past_target_3d[i][-1]
            whole_target = np.concatenate([past_target,future_target],axis=1)
            target_info_list = target_info_3d[i][-1]
            target_class_list = target_class_3d[i][-1]
            # 由于是整体指标数据，应该没有空值 
            ce_index = np.stack([item[i] for item in output_3d[3]])
            indus_num = int(output_3d[2][i].shape[0]/self.rolling_size)
            output_list = [output_3d[2][i][-indus_num:,:],ce_index]
            cur_target_info = target_info_list[main_index]
            date = int(target_info_list[main_index]["future_start_datetime"])
            if not date in TRACK_DATE:
                continue         
            # 生成整体指标涨跌趋势
            trend_value,pred_detail = self.build_import_index(output_data=output_list,target_info=target_info_list,
                            target=whole_target,
                            combine_instrument=combine_content,instrument_index=instrument_in_indus_index)
            price_round_data = target_info_list[main_index]["price_round_data"]
            if trend_value==1 and price_round_data[-1]>0:
                rate_total[date] = 1
            elif trend_value==0 and price_round_data[-1]<=0:
                rate_total[date] = 1  
            else:
                rate_total[date] = 0   
            pred_detail_list[date] = (trend_value,*pred_detail)      
        return rate_total,pred_detail_list

  
    
    def build_import_index(self,output_data=None,target=None,target_info=None,combine_instrument=None,instrument_index=None):  
        """生成涨幅达标的预测数据下标"""
        
        # return None,None,None,None,None
    
        (cls_values,ce_values) = output_data
        
        # pred_import_index,overroll_trend = self.strategy_top(cls_values,choice,trend_value,combine_index,target=target,price_array=price_array,target_info=target_info_list)
        trend_value,pred_detail = self.strategy_top_direct(cls_values,ce_values,
                                    target=target,target_info=target_info,instrument_index=instrument_index,combine_instrument=combine_instrument)

        return trend_value,pred_detail
                                          

    def strategy_top_direct(self,cls,ce_values,target=None,target_info=None,instrument_index=None,combine_instrument=None):
        """排名方式筛选候选者"""

        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        # 拿到价格差分数据，并以此为衡量整体指数的口径
        price_round_data = target_info[main_index]["price_round_data"]
        target_round_index = target_info[main_index]["target_round_index"]
        pred_round_data = target_info[main_index]["pred_round_data"]
        target_round_data = target_info[main_index]["target_round_data"]
        price_round_data = np.delete(price_round_data, [target_round_index])
        
        # cls_ins = cls[...,1]
        cls_ins = cls[...,0]
        ce_index = ce_values[:,0]
        # 归一化批次内的预测值
        ce_index_mean = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(pred_round_data,-1)).squeeze(-1)[target_round_index]
        # 整体指数预测数据转化为价格参考指数，并设置阈值进行涨跌判断
        price_inf = price_round_data.min() + (price_round_data.max()-price_round_data.min())*ce_index_mean
        
        price_inf_threhold = 0
        trend_value = 0
        
        trend_flag = (price_inf>price_inf_threhold)
        if trend_flag:
            trend_value = 1
        else:
            trend_value = 0
        
        return trend_value,(ce_index_mean,price_inf)

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
                         
