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
TRACK_DATE = [20220707]
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
        self.result_file_path = "custom/data/results/droll_rs_diff.pkl"
        self.result_columns = ["date","indus_index","trend_flag","price_inf","ce_inf"]
        
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
                round_skip_len=self.input_chunk_length,
                down_sampling_window=pred_len,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
            )           

            return model

    def create_loss(self,model,device="cpu"):
        return FuturesIndustryDRollLoss(device=device,ref_model=model,lock_epoch_num=self.lock_epoch_num,target_mode=self.target_mode,output_chunk_length=self.output_chunk_length)       
    

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
        # if int(target_info[0][-1][0]['future_start_datetime'])==20221011:
        #     print("ggg")            
        
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
        # 切分出过去整体round数值
        past_index_targets = index_round_targets[:,:,:-self.output_chunk_length,:]
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:-self.output_chunk_length,:]
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
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_index_round_target),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,top_num=self.top_num,epoch_num=self.current_epoch)        


    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
    
        rate_total,result_list = self.combine_result_data(self.output_result) 
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        rate_indus_total = np.stack([item[0] for item in rate_total.values()])
        rate_indus_top = np.stack([item[1] for item in rate_total.values()])
        sr = rate_indus_total[:,main_index]
        date_list = np.array(list(rate_total))
                   
        if len(sr)>0:
            self.log("total cnt", sr.shape[0], prog_bar=True)  
            self.log("trend corr cnt", np.sum(sr), prog_bar=True)  
            self.log("trend corr rate", np.sum(sr)/sr.shape[0], prog_bar=True) 
        
        # print("trend err result:{}",date_list[np.where(sr==0)[0]])
        
        # print("raise err result:{}",date_list[trend_raise[np.where(sr[trend_raise,1]==0)[0]]])
        # print("fall err result:{}",date_list[trend_fall[np.where(sr[trend_fall,2]==0)[0]]])
          
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
            price_targets_total = self.reshape_to_ori(price_targets_total,batch_size)
            index_round_targets_3d = self.reshape_to_ori(index_round_targets_3d,batch_size)
                    
            indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
            main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
            indus_index_rel = [i for i in range(past_target_3d.shape[2])]
            indus_index_without_main = [i for i in range(past_target_3d.shape[2])]
            indus_index_without_main.remove(main_index)
            # indus_index_rel.remove(main_index)                
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
                index_round_targets = index_round_targets_3d[index,:,:,-1,:]
                ce_output = output_3d[3]
                cls_output = output_3d[2]
                ts_arr = target_info_3d[index]
                price_targets = []
                for item in ts_arr:
                    price_item = [s["price_array"] for s in item]
                    price_targets.append(price_item)
                price_targets = np.array(price_targets)
                date = int(ts_arr[-1][main_index]["future_start_datetime"])
                if not date in TRACK_DATE:
                    continue      
                
                viz_total_size+=1
                rate_indus = rate_total[date][0]
                pred_detail = result_list[result_list["date"]==date]   
                # 可视化总体指数在不同时间段的预测值和实际值
                for j in range(len(self.past_split)):
                    # 整体指数趋势线
                    indus_price_array = price_targets[:,indus_index_rel] 
                    price_array_range = indus_price_array[:,:,-1] - indus_price_array[:,:,-self.output_chunk_length-1]    
                    index_price_array_range = price_array_range[:,main_index]
                    index_price_array_range = ts_arr[-1][main_index]["price_round_data"]
                    idx = 0
                    pred_indus_mean = pred_detail['ce_inf']
                    price_indus_inf = pred_detail['price_inf']
                    indus_trend_flag = pred_detail['trend_flag']
                    for indus_idx in indus_index_rel:
                        if indus_idx==main_index:
                            output_item = cls_output[index,:,:,j].mean(axis=1)
                            price_array_range_item = index_price_array_range
                        else:
                            output_item = cls_output[index,:,idx,j] 
                            price_array_range_item = price_array_range[:,indus_idx]
                            price_array_range_item = ts_arr[-1][indus_idx]["price_round_data"]
                            idx += 1
                            # continue
                        corr_flag = (rate_indus[indus_idx]==1)
                        index_target = index_round_targets[:,indus_idx,j]
                        view_data = np.stack([output_item,index_target,price_array_range_item]).transpose(1,0)
                        win = "index_round_target_{}_{}_{}".format(j,indus_idx,viz_total_size)                        
                        target_title = "[{}_{}]:{},pi:{},ce:{},trend:{}".format(
                            date,indus_names[indus_idx],corr_flag,round(price_indus_inf[indus_idx],2),
                            round(pred_indus_mean[indus_idx],1),indus_trend_flag[indus_idx])
                        names=["pred","target","price"]
                        if j in DRAW_SEQ_ITEM:# and indus_idx==main_index:
                            viz_result.viz_matrix_var(view_data,win=win,title=target_title,names=names)
                          
                    # lst_output_item = ce_output[j][index] 
                    # lst_target_item = index_round_targets[-1,indus_index_without_main,j] 
                    # view_data = np.stack([lst_output_item,lst_target_item]).transpose(1,0)
                    # win = "index_lst_target_{}_{}".format(j,viz_total_size)                        
                    # target_title = "{} indus compare".format(date)
                    # names=["pred","target"]
                    # if j in DRAW_SEQ_ITEM:
                    #     viz_result.viz_bar_compare(view_data,win=win,title=target_title,
                    #         rownames=indus_names[indus_index_without_main].tolist(),legends=["pred","target"])   
                                                                              
    def dump_val_data(self,val_batch,outputs,detail_loss):
    
        output,vr_class,price_targets,past_future_round_targets = outputs
        choice_out,trend_value,combine_index = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,index_round_targets,target_info) = val_batch
        # 记录批次内价格涨跌幅，用于整体指数批次归一化数据的回溯
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index_rel = [i for i in range(past_target.shape[1])]
        indus_index_rel.remove(main_index)         
        batch_size = int(index_round_targets.shape[0]/self.rolling_size)
        index_round_targets_reshape = self.reshape_to_ori(index_round_targets, batch_size)
        # 只获取整体指数的价格数据
        diff_array_main = []
        for item in target_info[:,-1]:
            price_array = item[main_index]["price_array"]
            diff_array = item[main_index]["diff_array"]
            diff_array_main.append(diff_array)
        diff_array_main = np.array(diff_array_main)
        price_round_data = diff_array_main[:,-1]
        
        # 针对每组，取得整体指标的价格差分结果数据
        for index,ts in enumerate(target_info):
            # 取得行业均值作为整体指数数值
            diff_array_indus = []
            for item in ts:
                price_array_item = np.array([item[indus_index]["price_array"] for indus_index in indus_index_rel])
                diff_array_item = np.array([item[indus_index]["diff_array"] for indus_index in indus_index_rel])
                diff_array_indus.append(diff_array_item)
            diff_array_indus = np.stack(diff_array_indus)
            price_round_data = diff_array_indus[:,:,-1]
            price_index_round_data = price_round_data.mean(axis=1)
            total_indus_pred = []
            # 保存到targetinfo中，后续使用
            for idx,indus_index in enumerate(indus_index_rel):
                ts[-1][indus_index]["price_round_data"] = price_round_data[:,idx]
                # 默认预测对应目标都在序列最后一个
                # 最后一个维度对应预测变量，目前只预测1个变量
                ts[-1][indus_index]["target_round_data"] = index_round_targets_reshape.cpu().numpy()[index,:,indus_index,-1,-1]
                ts[-1][indus_index]["pred_round_data"] = output[-1][1][index,:,idx,0].cpu().numpy()
                total_indus_pred.append(ts[-1][indus_index]["pred_round_data"] )
            ts[-1][main_index]["price_round_data"] = price_index_round_data
            # ts[-1][main_index]["pred_round_data"] = output[-1][2][index].cpu().numpy()
            ts[-1][main_index]["pred_round_data"] = np.array(total_indus_pred).mean(axis=0)
            ts[-1][main_index]["target_round_data"] = index_round_targets_reshape[index,:,indus_index_rel,-1,-1].cpu().numpy().mean(axis=1)
            # if int(ts[-1][0]['future_start_datetime'])==20221011:
            #     print("ggg")            
                       
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
                cls_inner.append(sv_indus.cpu().numpy())
                if ce_index_total[i] is None:
                    ce_index_total[i] = ce_index.cpu().numpy()
                else:
                    ce_index_total[i] = np.concatenate([ce_index_total[i],ce_index.cpu().numpy()],axis=0)
                
            cls_inner = np.concatenate(cls_inner)
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
        output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_3d,_,_, \
            index_round_targets_3d,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = {}
        
        instrument_in_indus_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
          
        # 还原维度后，每组按照最后一个日期进行评估
        batch_size = int(target_class_3d.shape[0]/self.rolling_size)
        target_class_3d = self.reshape_to_ori(target_class_3d,batch_size)
        future_target_3d = self.reshape_to_ori(future_target_3d,batch_size)
        past_target_3d = self.reshape_to_ori(past_target_3d,batch_size)
        price_targets_3d = self.reshape_to_ori(price_targets_3d,batch_size)
        
        indus_index_rel = [i for i in range(past_target_3d.shape[2])] 
        indus_index_rel_wm = [i for i in range(past_target_3d.shape[2])] 
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index_rel_wm.remove(main_index)
        result_total_list = None
        # 遍历按日期进行评估
        for i in range(batch_size):
            # 只获取最后一个时间段的数据
            future_target = future_target_3d[i][-1]
            past_target = past_target_3d[i][-1]
            whole_target = np.concatenate([past_target,future_target],axis=1)
            target_info_list = target_info_3d[i][-1]
            target_class_list = target_class_3d[i][-1]
            # 由于是整体指标数据，应该没有空值 
            ce_index = np.stack([item[i] for item in output_3d[3]]).squeeze(0)
            indus_num = int(output_3d[2][i].shape[0]/self.rolling_size)
            output_list = [output_3d[2][i][-indus_num:,:],ce_index]
            cur_target_info = target_info_list[main_index]
            date = int(target_info_list[main_index]["future_start_datetime"])
            # if not date in TRACK_DATE:
            #     continue         
            # 生成整体指标涨跌趋势
            trend_value,top_index,result_list = self.build_import_index(output_data=output_list,target_info=target_info_list,
                            target=whole_target,date=date,
                            combine_instrument=combine_content,instrument_index=instrument_in_indus_index)
            if trend_value is None:
                continue
            if result_total_list is None:
                result_total_list = result_list
            else:
                result_total_list = pd.concat([result_total_list,result_list])
                            
            rate_indus_total = []
            # 对整体指标和行业指标进行评估
            raise_num = 0
            for indus_index in indus_index_rel:
                trend_value_indus = result_list[result_list["indus_index"]==indus_index]["trend_flag"].values[0]
                price_indus_round_data = target_info_list[indus_index]["price_round_data"]
                if price_indus_round_data[-1]>0 and indus_index!=main_index:
                    raise_num += 1
                if (trend_value_indus==1) and (price_indus_round_data[-1]>0):
                    corr_flag = 1
                elif trend_value_indus==0 and price_indus_round_data[-1]<=0:
                    corr_flag = 1  
                else:
                    corr_flag = 0   
                rate_indus_total.append(corr_flag)
            trend_value_main = result_list[result_list["indus_index"]==main_index]["trend_flag"].values[0]
            if trend_value_main==1 and raise_num>=3:
                rate_indus_total[main_index] = 1
            elif trend_value_main==0 and raise_num<=3:
                rate_indus_total[main_index] = 1
            else:
                rate_indus_total[main_index] = 0
                            
            rate_indus_total = np.array(rate_indus_total)
            # 对TOP行业进行评估
            price_top_indus_round_data = target_info_list[top_index]["price_round_data"]
            sim_class = get_simple_class(price_top_indus_round_data[-1])
            rate_total[date] = [rate_indus_total,sim_class]
        # with open(self.result_file_path, "wb") as fout:
        #     pickle.dump(result_total_list, fout)             
        return rate_total,result_total_list

    def build_import_index(self,output_data=None,target=None,target_info=None,combine_instrument=None,instrument_index=None,date=None):  
        """生成涨幅达标的预测数据下标"""
        
        # return None,None
    
        (cls_values,ce_values) = output_data
        
        trend_value,top_index,result_list = self.strategy_top_direct(cls_values,ce_values,
                                    target=target,target_info=target_info,combine_instrument=combine_instrument,date=date)

        return trend_value,top_index,result_list
                                          

    def strategy_top_direct(self,cls,ce_values,target=None,target_info=None,date=None,combine_instrument=None):
        """排名方式筛选候选者"""

        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index_rel = [i for i in range(target.shape[0])]
        indus_index_without_main = [i for i in range(target.shape[0])]
        indus_index_without_main.remove(main_index)   
        
        # 拿到价格差分数据，并以此为衡量整体指数的口径
        price_round_data_total = []
        pred_round_data_total = []
        target_round_data_total = []
        for indus_index in indus_index_rel:
            price_round_data = target_info[indus_index]["price_round_data"]
            pred_round_data = target_info[indus_index]["pred_round_data"]
            pred_round_data_total.append(pred_round_data)
            price_round_data_total.append(price_round_data)
            target_round_data = target_info[indus_index]["target_round_data"]
            target_round_data_total.append(target_round_data)
            
        price_round_data_total = np.stack(price_round_data_total)
        # 去除后几个避免数据泄露
        price_round_data_total = price_round_data_total[:,:-self.output_chunk_length]
        pred_round_data_total = np.stack(pred_round_data_total)
        target_round_data_total = np.stack(target_round_data_total)
        
        price_inf_threhold = 0
        
        # 归一化批次内的预测值
        pred_index_round = pred_round_data_total[main_index]
        target_index_round = target_round_data_total[main_index]
        # pred_index_round = pred_round_data_total[indus_index_without_main].mean(axis=0)
        ce_index_mean = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(pred_index_round,-1)).squeeze()[-1]
        target_index_mean = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(target_index_round,-1)).squeeze()[-1]
        # 整体指数预测数据转化为价格参考指数，并设置阈值进行涨跌判断
        price_inf = price_round_data_total[main_index].min() + \
            (price_round_data_total[main_index].max()-price_round_data_total[main_index].min())*ce_index_mean*10
    
        # 分别对每个板块的涨跌趋势进行计算
        ce_indus_total = []
        price_indus_inf_total = []
        target_indus_inf_total = []
        trend_indus_total = []
        result_list = []
        for indus_index in indus_index_rel:
            pred_indus_round = pred_round_data_total[indus_index]
            ce_indus_mean = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(pred_indus_round,-1)).squeeze()[-1]
            target_indus_round = target_round_data_total[indus_index]
            target_indus_mean = MinMaxScaler(feature_range=(0.001, 1)).fit_transform(np.expand_dims(target_indus_round,-1)).squeeze()[-1]
            if indus_index==main_index:      
                trend_flag_indus = 0
                price_indus_inf = price_inf
            else:          
                price_indus_inf = price_round_data_total[indus_index].min() + \
                    (price_round_data_total[indus_index].max()-price_round_data_total[indus_index].min())*ce_indus_mean
                trend_flag_indus = (price_indus_inf>price_inf_threhold)
                # trend_flag_indus = (ce_indus_mean>0.5)
            target_indus_inf = pred_round_data_total[indus_index].min() + (pred_round_data_total[indus_index].max()-pred_round_data_total[indus_index].min())*target_indus_mean
            ce_indus_total.append(ce_indus_mean)
            trend_indus_total.append(trend_flag_indus)
            price_indus_inf_total.append(price_indus_inf)
            target_indus_inf_total.append(target_indus_inf)
            result_list.append([date,indus_index,trend_flag_indus,price_indus_inf,ce_indus_mean])
        
        trend_indus_total = np.array(trend_indus_total)+0
        # 构建结果集
        result_list = np.array(result_list)
        result_list = pd.DataFrame(result_list,columns=self.result_columns)
        result_list['date'] = result_list['date'].astype(int) 
        result_list['trend_flag'] = result_list['trend_flag'].astype(int) 
        result_list['indus_index'] = result_list['indus_index'].astype(int) 
        # 根据行业涨跌数量判断整体指数涨跌
        trend_value = (np.sum(trend_indus_total[indus_index_without_main])>(trend_indus_total[indus_index_without_main].shape[0]//2))+0
        # 反填到主指标记录
        result_list.loc[result_list["indus_index"]==main_index,"trend_flag"]=trend_value
        # 根据涨跌趋势，挑选排名靠前的行业类别
        if trend_value:
            top_index = np.argmax(result_list["price_inf"].values[indus_index_without_main])
        else:
            top_index = np.argmin(result_list["price_inf"].values[indus_index_without_main])
        top_index = indus_index_rel[top_index]
        
        return trend_value,top_index,result_list

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
                         
