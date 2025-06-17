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
from sklearn.metrics import precision_score, recall_score, f1_score

from .mlp_module import MlpModule
import cus_utils.global_var as global_var
from darts_pro.act_model.fur_industry_ts import FurIndustryMixer,FurStrategy
from losses.mixer_loss import FuturesIndustryLoss
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil

from cus_utils.common_compute import compute_price_class,batch_cov_comp,normalization,\
    normalization_axis
from tft.class_define import CLASS_SIMPLE_VALUES,get_simple_class
from cus_utils.tensor_viz import TensorViz

TRACK_DATE = [20221010,20221011,20220518,20220718,20220811,20220810,20220923]
TRACK_DATE = [20220505]
STAT_DATE = [20220501,20220530]
INDEX_ITEM = 0
DRAW_SEQ = [0]
DRAW_SEQ_ITEM = [1]
DRAW_SEQ_DETAIL = [0]

warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

class FuturesIndustryDRollModule(MlpModule):
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
        step_mode='complete',
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
        # 阶段模式，0--表示全阶段， 1--表示第一阶段，先进行整体和行业预测 2--表示第二阶段，进行品种预测
        self.step_mode = step_mode
        
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,batch_file_path=batch_file_path,
                                    device=device,**kwargs)  
        self.result_file_path = "custom/data/results/step1_rs.pkl"
        self.result_columns = ["date","indus_index","trend_flag","price_inf","ce_inf"]
        
        if self.step_mode==2:
            # 第二阶段，首先加载预存结果
            with open(self.result_file_path, "rb") as fin:
                self.result_data = pickle.load(fin)           
        
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
            past_covariates_item = past_covariate[...,past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            
            # 过去协变量维度计算,不使用时间协变量
            input_dim = (
                past_target_shape
                + past_covariates_shape
            )
            
            main_index = FuturesMappingUtil.get_main_index_in_indus(self.train_sw_ins_mappings)
            combine_nodes = FuturesMappingUtil.get_industry_instrument(self.train_sw_ins_mappings,without_main=True)
            combine_nodes_num = np.array([ins.shape[0] for ins in combine_nodes])
            combine_nodes_num = torch.Tensor(combine_nodes_num).int().to(self.device)
            instrument_index = combine_nodes
            industry_index = FuturesMappingUtil.get_industry_data_index_without_main(self.train_sw_ins_mappings)
            index_num = combine_nodes_num.shape[0]
            # 加入短期指标
            if self.target_mode[seq] in [0]:
                pred_len = self.output_chunk_length
                cut_len = pred_len
            elif self.target_mode[seq] in [1,2]:
                pred_len = self.output_chunk_length
                # 对于时间序列对比模式，使用指定输出长度
                cut_len = self.cut_len            
            elif self.target_mode[seq] in [3]:
                pred_len = self.output_chunk_length    
                cut_len = self.cut_len    
                combine_nodes = FuturesMappingUtil.get_all_instrument(self.train_sw_ins_mappings)
                combine_nodes_num = np.expand_dims(combine_nodes.shape[0],0)
                combine_nodes_num = torch.Tensor(combine_nodes_num).int().to(self.device)   
                instrument_index = np.expand_dims(combine_nodes,0)  
                industry_index = [main_index]      
                index_num = 1     
            # 使用混合时间序列模型
            model = FurIndustryMixer(
                target_mode=self.target_mode[seq],
                combine_nodes_num=combine_nodes_num, # 对应不同行业板块的期货品种数量
                index_num=index_num,
                instrument_index=instrument_index,
                industry_index=industry_index,
                seq_len=self.input_chunk_length,
                pred_len=pred_len,
                cut_len=cut_len,
                round_skip_len=self.input_chunk_length,
                down_sampling_window=pred_len,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
            )           

            return model

    def create_loss(self,model,device="cpu"):
        return FuturesIndustryLoss(device=device,ref_model=model,lock_epoch_num=self.lock_epoch_num,target_mode=self.target_mode,cut_len=self.cut_len)       
    

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
            long_diff_index_targets,
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
                            (future_target,target_class,past_future_round_targets,index_round_targets,long_diff_index_targets),optimizers_idx=i)
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
        # self.log("lr_last",self.trainer.optimizers[-2].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=False)  
        
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
            long_diff_index_targets,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,past_future_round_targets,index_round_targets) 
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,past_future_round_targets,index_round_targets,long_diff_index_targets),optimizers_idx=-1)
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
        past_index_targets = index_round_targets[:,:,:self.input_chunk_length,:]
        # 去掉正泰zsall部分
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
        past_index_targets = past_index_targets[:,indus_rel_index,:,:]
        # 切分单独的过去round数值
        past_round_targets = past_future_round_targets[:,:,:self.input_chunk_length,:]
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, historic_future_covariates,future_covariates, static_covariates,price_targets,past_round_targets,past_index_targets)
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,past_future_round_targets,index_round_targets,long_diff_index_targets) = target 
        future_round_targets = past_future_round_targets[:,:,-1,:]  
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,index_round_targets,long_diff_index_targets),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx,top_num=self.top_num,epoch_num=self.current_epoch)        


    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        rate_total,indus_result,total_imp_cnt,import_price_result = self.combine_result_data(self.output_result) 
        if rate_total is not None and rate_total.shape[0]>0:
            # indus_bitop_diff = rate_total['indus_bitop_diff'].sum()
            indus_top_diff = rate_total['indus_top_diff'].sum()
            trend_corr_cnt = rate_total['trend_correct'].sum()
            fall_top_corr_cnt = rate_total['fall_top_corr'].sum()
            raise_top_corr_cnt = rate_total['raise_top_corr'].sum()
            if self.step_mode!=1:
                total_cnt = rate_total['total_cnt'].sum()
                yield_rate = rate_total['yield_rate'].sum()
                self.log("yield_rate", yield_rate, prog_bar=True) 
                # f1_scores = rate_total['f1_score'].mean()
                for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                    name = "cls{}_cnt".format(i)
                    cnt = rate_total[name].sum()
                    self.log("score_{} rate".format(i), round(cnt/total_cnt,3), prog_bar=True) 
            for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                cnt = np.sum(rate_total['indus_top_class']==i)
                if cnt>0:
                    self.log("indus_top_{} rate".format(i), cnt/rate_total.shape[0], prog_bar=True)                 
                # self.log("score_{} min rate".format(i), combine_rate_min[i], prog_bar=True) 
            # self.log("indus_bitop_diff", round(indus_bitop_diff,3), prog_bar=True)  
            # self.log("fall_top_corr", fall_top_corr_cnt/rate_total.shape[0], prog_bar=True)  
            # self.log("raise_top_corr", raise_top_corr_cnt/rate_total.shape[0], prog_bar=True)  
            self.log("indus_top_diff", round(indus_top_diff,3), prog_bar=True)  
            self.log("trend corr rate", round(trend_corr_cnt/rate_total.shape[0],3), prog_bar=True)  
            # self.log("total cnt", total_cnt, prog_bar=True)  
            # self.log("total_imp_cnt", total_imp_cnt, prog_bar=True)  
            # self.log("f1_scores", round(f1_scores,3), prog_bar=True)  
     
        # 如果是测试模式，则在此进行可视化
        if self.mode is not None and self.mode.startswith("pred_") :
            
            # 第一阶段，存储结果
            if self.step_mode==1:
                with open(self.result_file_path, "wb") as fout:
                    pickle.dump(indus_result, fout)  


            # rate_total_stat = rate_total[(rate_total['date']>=STAT_DATE[0])&(rate_total['date']<=STAT_DATE[1])]
            # print("trend fail date:\n",rate_total_stat[rate_total_stat['trend_correct']==0]['date'])  
            # print("indus_top fail date:\n",rate_total_stat[rate_total_stat['indus_top_class']==1][['date']])

            # if import_price_result is not None:
            #     import_price_result_item = import_price_result[(import_price_result['date']>=STAT_DATE[0])&(import_price_result['date']<=STAT_DATE[1])]
            #     print("ins result:\n",import_price_result_item[['date','instrument','result','yield_rate','trend_flag']])  
            #     result_file_path = "custom/data/results/pred_coll.pkl"
            #     with open(result_file_path, "wb") as fout:
            #         pickle.dump(import_price_result_item, fout)                  
                                        
            tar_viz = global_var.get_value("viz_data")
            viz_result = global_var.get_value("viz_result_detail")
            # viz_result_detail = global_var.get_value("viz_result_detail")
            
            viz_total_size = 0
                          
            output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_total, \
                past_future_round_targets_total,long_diff_index_targets_total,index_round_targets_3d,target_info_3d = self.combine_output_total(self.output_result)
            
            indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
            indus_rel_index = FuturesMappingUtil.get_industry_rel_index(sw_ins_mappings)
            main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
            main_index_rel =  FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
            indus_names_all = FuturesMappingUtil.get_industry_names(sw_ins_mappings)
            indus_names = indus_names_all[indus_rel_index]
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
                
                round_targets = past_future_round_targets_total[index]
                index_round_targets = index_round_targets_3d[index,:,-1,:]
                index_round_targets_mul = index_round_targets_3d[index,:,:,:]
                cls_output = output_3d[2]
                ce_output = output_3d[3]
                dec_out = output_3d[4]
                ts_arr = target_info_3d[index]
                date = int(ts_arr[keep_index][0]["future_start_datetime"])
                if not date in TRACK_DATE:
                    continue    
                  
                rate_item = rate_total[rate_total['date']==date]
                trend_value = rate_item['trend_value'].values[0]   
                indus_top_diff = rate_item['indus_top_diff'].values[0]   
                indus_result_item = indus_result[indus_result['date']==date]
                ins_with_indus = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
                ins_rel_index = FuturesMappingUtil.get_instrument_rel_index(sw_ins_mappings)
                
                # 可视化相互关系
                for j in range(len(self.past_split)):
                    total_view_data = None
                    # 按照每个板块分类，分别显示板块内的品种整体预测数据
                    futures_names_combine = None
                    futures_index_combine = None
                    price_range_total = []
                    
                    if j in DRAW_SEQ_DETAIL and self.step_mode==2:
                        # 所有品种的比对图
                        instruments = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)
                        ins_output = cls_output[j][index]
                        inner_class_item = target_class_item[instruments]
                        inner_index = np.where(inner_class_item>=0)[0]    
                        ins_output = ins_output[inner_index]
                        fur_round_target = round_targets[instruments[inner_index],-1,j]       
                        futures_names = FuturesMappingUtil.get_all_instrument_code_names(sw_ins_mappings)
                        futures_names = futures_names[inner_index]                                         
                        price_array_range = np.array([item["diff_range"][-1] for item in ts_arr[instruments][inner_index]])
                        view_data = np.stack([ins_output,fur_round_target,price_array_range]).transpose(1,0)
                        win = "round_targetall_{}_{}".format(j,viz_total_size)
                        target_title = "target{}_All,date:{}".format(j,date)                            
                        viz_result.viz_bar_compare(view_data,win=win,title=target_title,rownames=futures_names.tolist(),legends=["pred","target","price"])       
                                      
                    for k,instruments in enumerate(ins_with_indus):
                        indus_index_item = indus_index[indus_rel_index[k]]
                        ins_index = industry_instrument_index[k]
                        inner_class_item = target_class_item[ins_index]
                        inner_index = np.where(inner_class_item>=0)[0]           
                        instruments,k_idx,_ = np.intersect1d(instruments,keep_index,return_indices=True)
                        indus_code = indus_codes[k]
                        indus_name = indus_names[k]
                        futures_names = FuturesMappingUtil.get_futures_names(sw_ins_mappings,indus_rel_index[k])[k_idx].tolist()
                        if k==0:
                            rel_index = [0,ins_rel_index[k]]
                        else:
                            rel_index = [ins_rel_index[k-1],ins_rel_index[k]]
                        ins_output = cls_output[j][index,rel_index[0]:rel_index[1]]
                        ins_output = ins_output[inner_index]
                        fur_round_target = round_targets[instruments,-1,j]

                        # 行业内品种比对图
                        if j in DRAW_SEQ_DETAIL and len(futures_names)>1 and self.step_mode==2:
                            price_array_range = np.array([item["diff_range"][-1] for item in ts_arr[ins_index][inner_index]])
                            view_data = np.stack([ins_output,fur_round_target,price_array_range]).transpose(1,0)
                            win = "round_target_{}_{}_{}".format(j,k,viz_total_size)
                            target_title = "target{}_{},date:{}".format(j,indus_name,date)                            
                            viz_result.viz_bar_compare(view_data,win=win,title=target_title,rownames=futures_names,legends=["pred","target","price"])   
                            

                        if j in DRAW_SEQ_ITEM and self.step_mode!=2:
                            # 行业板块历史推理数值图
                            indus_output_value = ts_arr[indus_index_item]["pred_data"]
                            indus_target = index_round_targets_3d[index,indus_rel_index[k],:,j]
                            price_target = ts_arr[indus_index_item]["diff_range"] * 10
                            win = "whole_round_target_{}_{}_{}".format(j,k,viz_total_size)    
                            target_title = "target_{}_{},date:{},price_inf:{}".format(indus_name,j,date,round(indus_result_item["price_inf"].values[k],3))
                            viz_detail = viz_result_detail["all"]
                            view_data = np.stack([indus_output_value,indus_target,price_target]).transpose(1,0)
                            names=["pred","target","price"]                            
                            viz_detail.viz_matrix_var(view_data,win=win,title=target_title,names=names)     

                            # 行业板块数值图
                            # indus_output_value = ce_output[j][index,k]
                            # indus_output_value = np.pad(indus_output_value,(self.input_chunk_length-self.cut_len,0),'constant',constant_values=(0,0))
                            # indus_target = index_round_targets_3d[index,indus_rel_index[k],:,j]
                            # price_target = ts_arr[indus_index_item]["diff_range"] * 10
                            # win = "indus_value_round_target_{}_{}_{}".format(j,k,viz_total_size)    
                            # target_title = "target_{}_{},date:{},price_inf:{}".format(indus_name,j,date,round(indus_result_item["price_inf"].values[k],3))
                            # viz_detail = viz_result_detail["all"]
                            # view_data = np.stack([indus_output_value,indus_target,price_target]).transpose(1,0)
                            # names=["pred","target","price"]                            
                            # viz_detail.viz_matrix_var(view_data,win=win,title=target_title,names=names)     
                                                    
                            # 显示时间段比对图   
                            indus_output_value = ce_output[j][index,k,:]
                            diff_index_round_target = long_diff_index_targets_total[index,indus_rel_index[k],:,j]
                            price_target = ts_arr[indus_index_item]["diff_range"][-self.output_chunk_length-self.cut_len:-self.output_chunk_length]
                            price_target =  MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(np.expand_dims(price_target,-1)).squeeze(-1)
                            win = "round_target_mul{}_{}_{}".format(j,k,viz_total_size)    
                            target_title = "seq_target_{}_{},date:{}".format(indus_name,j,date)
                            viz_detail = viz_result_detail["all"]
                            view_data = np.stack([indus_output_value,diff_index_round_target,price_target]).transpose(1,0)
                            date_list = ["S{}".format(i) for i in range(self.cut_len)]
                            viz_detail.viz_bar_compare(view_data,win=win,title=target_title,rownames=date_list,legends=["pred","target","price"])      
                                        
                    # 显示板块比对图
                    indust_output_value = ce_output[j][index]
                    indust_target = index_round_targets[indus_rel_index,j]
                    price_range_total = np.array([ts["diff_range"][-1] for ts in ts_arr[indus_index[indus_rel_index]]])
                    win = "indus_round_target_{}_{}".format(j,viz_total_size)
                    target_title = "[{}-{}] trend:{},diff:{}".format(int(date),j,trend_value,round(indus_top_diff,3))
                    if j in DRAW_SEQ and self.step_mode!=2:
                        # indust_output_value = indust_output_value.repeat(6)
                        view_data = np.stack([indust_output_value,indust_target,price_range_total*10]).transpose(1,0)
                        tar_viz.viz_bar_compare(view_data,win=win,title=target_title,rownames=indus_names.tolist(),legends=["pred","target","price"])   
                        
                                    
    def dump_val_data(self,val_batch,outputs,detail_loss):
    
        output,vr_class,price_targets,past_future_round_targets = outputs
        choice_out,trend_value,combine_index = vr_class
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,index_round_targets,long_diff_index_targets,target_info) = val_batch
        # 记录批次内价格涨跌幅，用于整体指数批次归一化数据的回溯
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        #
        # for index,ts in enumerate(target_info):
        #     ts[main_index]["price_round_data"] = price_round_data
        #     ts[main_index]["price_round_index"] = index
        #     ts[main_index]["target_round_data"] = index_round_targets.cpu().numpy()[:,-1,-1,-1]
        #     ts[main_index]["pred_round_data"] = output[-1][2].cpu().numpy().squeeze(-1)
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output,choice_out.cpu().numpy(),trend_value.cpu().numpy(),combine_index.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),past_future_round_targets.cpu().numpy(),whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(),long_diff_index_targets.cpu().numpy(),target_info)
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
        cls_total = [None for _ in range(len(self.past_split))]
        dec_total = []
        ce_index_total = [None for _ in range(len(self.past_split))]
        choice_total = []
        trend_total = []
        combine_index_total = []
        index_round_targets_total = []
        long_diff_index_targets_total = []
        for item in output_result:
            (output,choice,trend_value,combine_index,past_target,future_target,target_class,price_targets,past_future_round_targets,whole_index_round_targets,index_round_targets,long_diff_index_targets,target_info) = item
            x_bar_inner = []
            dec_inner = []
            for i in range(len(self.past_split)):
                output_item = output[i]
                dec_out,sv_indus,ce_index = output_item 
                dec_inner.append(dec_out.cpu().numpy())
                # 合并列表中的品种维度部分
                sv_indus = torch.cat(sv_indus,dim=1).squeeze(-1)
                if cls_total[i] is None:
                    cls_total[i] = sv_indus.cpu().numpy()
                else:
                    cls_total[i] = np.concatenate([cls_total[i],sv_indus.cpu().numpy()],axis=0)                
                if ce_index_total[i] is None:
                    ce_index_total[i] = ce_index.cpu().numpy()
                else:
                    ce_index_total[i] = np.concatenate([ce_index_total[i],ce_index.cpu().numpy()],axis=0)
                
            dec_inner = np.stack(dec_inner).transpose(1,2,3,0)
            dec_total.append(dec_inner)
            # ce_index_inner = np.stack(ce_index_inner).transpose(1,2,0)
            x_bar_total.append(x_bar_inner)
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
                    
        return (x_bar_total,sv_total,cls_total,ce_index_total,dec_total,trend_total,combine_index_total), \
                    past_target_total,future_target_total,target_class_total,price_targets_total,past_future_round_targets_total, \
                    long_diff_index_targets_total,index_round_targets_total,target_info_total        
                           
    def combine_result_data(self,output_result=None,predict_mode=False):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # return None,None,None,None
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_3d,_,long_diff_index_targets_3d, \
            index_round_targets_3d,target_info_3d  = self.combine_output_total(output_result)
        total_imp_cnt = np.where(target_class_3d==3)[0].shape[0]
        rate_total = []
        result_date_list = {}
        
        instrument_index = FuturesMappingUtil.get_instrument_index(sw_ins_mappings)
        instrument_in_indus_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
        main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
        industry_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        industry_index_proxy = [main_index] if self.target_mode[0]==3 else industry_index
        # 按照时间索引暂存预测数据，用于全局化共享使用
        glo_match_data = []
        for i in range(target_class_3d.shape[0]):
            target_class_list = target_class_3d[i]
            target_info_list = target_info_3d[i]
            ce_index = [item[i] for item in output_3d[3]]
            # 根据配置，决定针对行业数据进行处理还是针对整体指数数据进行处理
            for j,indus_index in enumerate(industry_index_proxy):
                target_info = target_info_list[indus_index]
                rank_code = target_info["item_rank_code"]
                indus_code = target_info["instrument"]
                date = target_info["future_start_datetime"]  
                # 因为预测的是最后一个未来日期和前面的差值，因此按照最后一个时间序号作为序列编号
                time_index = target_info["future_end"] - 1 
                # 预测数据放入记录，与最后一个日期序号对应
                if len(ce_index)>1:
                    pred_data = ce_index[1][j,-1]
                else:
                    pred_data = ce_index[0][j]
                glo_match_data.append([indus_index,date,indus_code,time_index,pred_data])
        
        columns = ["indus_index","date","indus_code","time_index","pred_data"]       
        glo_match_data = pd.DataFrame(np.array(glo_match_data),columns=columns)
        glo_match_data['time_index'] = glo_match_data['time_index'].astype(int)
        glo_match_data['date'] = glo_match_data['date'].astype(float).astype(int)
        
        import_price_result_list = None
        indus_result_total_list = None
        # 遍历按日期进行评估
        for i in range(target_class_3d.shape[0]):
            future_target = future_target_3d[i]
            past_target = past_target_3d[i]
            whole_target = np.concatenate([past_target,future_target],axis=1)
            target_info_list = target_info_3d[i]
            target_class_list = target_class_3d[i]
            # 有一些空值，找出对应索引后续做忽略处理
            keep_index = np.where(target_class_list>=0)[0]
            # 去除指数整体及行业
            keep_index = np.intersect1d(keep_index,instrument_index)  
            ce_index = [item[i] for item in output_3d[3]]
            cls_index = [item[i] for item in output_3d[2]]
            dec_out = output_3d[4][i]
            output_list = [cls_index,ce_index,output_3d[4][i],output_3d[5][i],output_3d[6][i]]
            price_target_list = price_targets_3d[i]
            date = int(target_info_list[np.where(target_class_list>-1)[0][0]]["future_start_datetime"])
            index_round_targets = index_round_targets_3d[i]
            if self.step_mode==2 and not (date>=STAT_DATE[0] and date<=STAT_DATE[1]):
                continue  
            # 把之前生成的预测值，植入到target_info基础信息中，后续使用
            for target_info in target_info_list[industry_index]:
                if target_info is None:
                    continue
                past_start = target_info["past_start"]
                future_end = target_info["future_end"]
                pred_data_combine = glo_match_data[(glo_match_data["indus_code"]==target_info["instrument"])&(glo_match_data["time_index"]>past_start)
                                &(glo_match_data["time_index"]<=future_end)][["time_index","pred_data"]]
                if pred_data_combine.shape[0]==0:
                    continue
                if pred_data_combine.shape[0]==(self.input_chunk_length + self.output_chunk_length):
                    # 如果长度匹配，则全部使用
                    target_info["pred_data"] = pred_data_combine.values[:,1].astype(float)
                else:
                    # 如果长度不匹配，则根据索引号缺失的记录进行插值
                    target_info["pred_data"] = pred_data_combine.values[:,1]    
                    tmp_data = np.array([i for i in range(target_info["past_start"],target_info["future_end"])])
                    tmp_data = np.stack([tmp_data,np.array([0 for _ in range(self.input_chunk_length + self.output_chunk_length)])]).transpose(1,0)
                    tmp_data = pd.DataFrame(tmp_data,columns=["time_index","pred_data"])    
                    data = pd.merge(tmp_data, pred_data_combine, on='time_index',how='left').fillna(0)  
                    target_info["pred_data"] = data.values[:,-1].astype(float)      
            
            # 生成目标索引
            import_index,trend_value,indus_top_index,result_list,ins_result_list = self.build_import_index(output_data=output_list,target_info=target_info_list,
                            target=whole_target,price_target=price_target_list,index_round_targets=index_round_targets,date=date,
                            combine_instrument=combine_content,instrument_index=instrument_in_indus_index)

            # 有可能没有候选数据
            if import_index is not None and import_index.shape[0]>0:
                import_index = np.intersect1d(keep_index,import_index)  
                
            # 如果是预测模式，则只输出结果,不验证
            if predict_mode:
                # result_date_list[date] = [import_index,trend_value]
                result_date_list[date] = ins_result_list
                continue
  
            collect_mode = 1
            # Compute Acc Resultes
            if collect_mode==1:
                import_price_result,indus_top_class,indus_top_diff,main_trend_correct,indus_bitop_diff,indus_top_corr = self.collect_result(import_index,overroll_trend=trend_value,
                                                indus_top_index=indus_top_index,target_info=target_info_list,indus_result_list=result_list)
            else:
                import_price_result = self.collect_result_bidi(import_index,target_info=target_info_list,result_list=ins_result_list)            
            
            # 把结果数据整合到预测记录中
            if indus_result_total_list is None:
                indus_result_total_list = result_list
            else:
                indus_result_total_list = pd.concat([indus_result_total_list,result_list])                
            
            rate_item = []
            rate_item.append(date)
            if import_price_result is not None:
                rate_columns = ["date"] + ["cls{}_cnt".format(i) for i in range(len(CLASS_SIMPLE_VALUES.keys()))] + \
                        ["total_cnt","yield_rate","trend_value","trend_correct","indus_top_class","indus_top_diff",
                         "indus_bitop_diff","fall_top_corr","raise_top_corr","f1_score"]                   
                import_price_result['date'] = date
                if import_price_result_list is None:
                    import_price_result_list = import_price_result
                else:
                    import_price_result_list = pd.concat([import_price_result_list,import_price_result])
                result_values = import_price_result["result"].values
                suc_cnt = np.sum(result_values>=2)
                fail_cnt = np.sum(result_values<2)
                if fail_cnt>0 or True:
                    trend_str = "Rise"
                    if trend_value==0:
                        trend_str = "Fall"
                    result_date_list["{}_{}/{}".format(int(date),trend_str,suc_cnt)] = \
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
                    rate_item.append(cnt)
                # 预测数量以及总数量
                rate_item.append(total_cnt.item())   
                rate_item.append(import_price_result['yield_rate'].sum()) 
            else:
                rate_columns = ["date"] + ["trend_value","trend_correct","indus_top_class","indus_top_diff",
                         "indus_bitop_diff","fall_top_corr","raise_top_corr","f1_score"]                   
            # 添加多空判断预测信息 
            if collect_mode==1:
                rate_item.append(trend_value)   
                rate_item.append(main_trend_correct)
                rate_item.append(indus_top_class)   
                rate_item.append(indus_top_diff)   
                rate_item.append(indus_bitop_diff) 
                rate_item.append(indus_top_corr[0]) 
                rate_item.append(indus_top_corr[1]) 
            else:
                rate_item.append(0) 
                rate_item.append(0) 
                rate_item.append(0) 
                rate_item.append(0) 
                rate_item.append(0) 
                trend_fall_result = import_price_result[import_price_result['trend_flag']==0]
                trend_raise_result = import_price_result[import_price_result['trend_flag']==1]
                fall_corr = np.sum(trend_fall_result['yield_rate'].values>0)/trend_fall_result.shape[0]
                rate_item.append(fall_corr) 
                raise_corr = np.sum(trend_raise_result['yield_rate'].values>0)/trend_raise_result.shape[0]
                rate_item.append(raise_corr) 
            rate_item.append(0) 
            rate_total.append(rate_item)
            
        if predict_mode:
            return result_date_list     
               
        rate_total = np.array(rate_total)
        if rate_total.shape[0]==0:
            return None,None,None,None
        rate_total = pd.DataFrame(rate_total,columns=rate_columns)
        
        return rate_total,indus_result_total_list,total_imp_cnt,import_price_result_list
       
    def collect_result(self,import_index,overroll_trend=0,indus_top_index=None,target_info=None,indus_result_list=None): 
        """收集预测对应的实际数据"""
    
        import_price_result = None
        if self.step_mode!=1:
            import_price_result = []
            # 对于预测数据，生成对应涨跌幅类别
            for i,imp_idx in enumerate(import_index):
                ts = target_info[imp_idx]
                price_array = ts["price_array"][self.input_chunk_length-1:]
                diff_range = (price_array[-1] - price_array[0])/price_array[0]
                p_taraget_class = compute_price_class(price_array,mode="first_last")
                # 根据多空判断取得实际对应的类别
                if overroll_trend==0:
                    diff_range = -diff_range
                    p_taraget_class = np.array([3,2,1,0])[p_taraget_class]
                import_price_result.append([imp_idx,ts["instrument"],diff_range,p_taraget_class])    
            import_price_result = np.array(import_price_result)  
            if import_price_result.shape[0]==0:
                return None,None,None
            import_price_result = pd.DataFrame(import_price_result,columns=["imp_index","instrument","yield_rate","result"])     
            import_price_result["result"] = import_price_result["result"].astype(np.int64)      
            import_price_result["yield_rate"] = import_price_result["yield_rate"].astype(np.float)   
    
        # 同时计算行业趋势判断准确率
        trend_results = []
        total_diff_range = []
        indus_bitop_diff = 0
        indus_top_corr = [None,None]
        for index,row in indus_result_list.iterrows():
            indus_index = int(row["indus_index"])
            trend_flag = int(row["trend_flag"])
            diff_range = target_info[indus_index]["diff_range"][-1] 
            total_diff_range.append(diff_range)
            p_taraget_class = get_simple_class(diff_range)
            if trend_flag==0:
                p_taraget_class = np.array([3,2,1,0])[p_taraget_class]                
            trend_results.append(p_taraget_class)
            # 针对行业top预测进行判断
            top_flag = row['top_flag']
            p_taraget_class = get_simple_class(diff_range)
            if top_flag==1:
                indus_top_corr[1] = (diff_range>0)
                indus_bitop_diff += diff_range
            if top_flag==0:
                indus_top_corr[0] = (diff_range<0)
                indus_bitop_diff -= diff_range               
        total_diff_range = np.array(total_diff_range)
        # 新增行业预测结果到原结果集
        indus_result_list["trend_result"] = np.array(trend_results)
        # 计算行业top准确率,以及幅度
        diff_range = target_info[indus_top_index]["diff_range"][-1] 
        indus_top_diff = 0
        indus_top_class = get_simple_class(diff_range)      
        if overroll_trend==0:
            indus_top_class = np.array([3,2,1,0])[indus_top_class]  
            indus_top_diff -= diff_range 
        else:
            indus_top_diff += diff_range 
        # 计算整体趋势准确率
        main_trend_correct = 0
        raise_num = np.sum(total_diff_range>=0)
        if overroll_trend==1 and raise_num>=3:
            main_trend_correct = 1
        elif overroll_trend==0 and raise_num<3:
            main_trend_correct = 1
        else:
            main_trend_correct = 0        
    
        return import_price_result,indus_top_class,indus_top_diff,main_trend_correct,indus_bitop_diff,indus_top_corr
    
    def collect_result_bidi(self,import_index,target_info=None,result_list=None): 
        """收集预测对应的实际数据,双向模式"""
        
        import_price_result = []
        # 对于预测数据，生成对应涨跌幅类别
        for i,imp_idx in enumerate(import_index):
            trend_flag = result_list[result_list['top_index']==imp_idx]['top_flag'].values[0]
            ts = target_info[imp_idx]
            price_array = ts["price_array"][self.input_chunk_length-1:]
            # 加入开盘数据，计算开盘目标价格与前边的收盘价格的差值
            open_array = ts["open_array"][self.input_chunk_length-1:]
            diff_range = (price_array[-1] - price_array[0])/price_array[0]
            diff_range = (open_array[-1] - price_array[0])/price_array[0]
            p_taraget_class = get_simple_class(diff_range) 
            # 根据多空判断取得实际对应的类别
            if trend_flag==0:
                diff_range = -diff_range
                p_taraget_class = np.array([3,2,1,0])[p_taraget_class]
            import_price_result.append([imp_idx,ts["instrument"],diff_range,p_taraget_class,trend_flag])    
        import_price_result = np.array(import_price_result)  
        if import_price_result.shape[0]==0:
            return None
        import_price_result = pd.DataFrame(import_price_result,columns=["imp_index","instrument","yield_rate","result","trend_flag"])     
        import_price_result["result"] = import_price_result["result"].astype(np.int64)      
        import_price_result["trend_flag"] = import_price_result["trend_flag"].astype(np.int64)  
        import_price_result["yield_rate"] = import_price_result["yield_rate"].astype(np.float)   
    
        return import_price_result
    
            
    def build_import_index(self,date=None,output_data=None,target=None,price_target=None,target_info=None,combine_instrument=None,instrument_index=None,index_round_targets=None):  
        """生成涨幅达标的预测数据下标"""
        
        (cls_values,ce_values,choice,trend_value,combine_index) = output_data
        
        # 不同阶段，使用不同的策略
        if self.step_mode==0:
            indus_top_index,trend_value,result_list = self.strategy_top_indus(cls_values,ce_values,
                                        target=target,target_info=target_info,index_round_targets=index_round_targets,
                                   combine_instrument=combine_instrument)
        elif self.step_mode==1:
            indus_top_index,trend_value,result_list = self.strategy_top_indus(cls_values,ce_values,
                                        target=target,target_info=target_info,index_round_targets=index_round_targets,
                                   combine_instrument=combine_instrument)           
            return None,trend_value, indus_top_index,result_list,None
        else:
            # 第二阶段时，先加载第一阶段结果，然后再延续使用
            result_list = self.result_data[self.result_data['date']==date]
            trend_value = (np.sum(result_list['trend_flag']>0)>result_list.shape[0]//2) + 0
            indus_top_index = result_list[result_list['top_flag']==trend_value]['indus_index'].values[0]

        # import_index_real = self.strategy_top_all(cls_values,ce_values,indus_top_index=indus_top_index,trend_value=trend_value,
        #                                 target=target,target_info=target_info,index_round_targets=index_round_targets,result_list=result_list,
        #                            combine_instrument=combine_instrument)
        import_index_real = self.strategy_top(cls_values,ce_values,indus_top_index=indus_top_index,trend_value=trend_value,
                                        target=target,target_info=target_info,index_round_targets=index_round_targets,result_list=result_list,
                                   combine_instrument=combine_instrument)        
        return import_index_real,trend_value,indus_top_index,result_list,None
        
    def strategy_top_indus(self,cls,ce_values,target=None,target_info=None,index_round_targets=None,combine_instrument=None):
        """行业排名方式筛选候选者"""

        date = int(target_info[0]['future_start_datetime'])
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index = FuturesMappingUtil.get_industry_data_index(sw_ins_mappings)
        industry_index = [i for i in range(indus_index.shape[0])]
        industry_index_real = indus_index[industry_index]
        industry_index.remove(main_index)
               
        # 整体指数预测数据转化为价格参考指数，并设置阈值进行涨跌判断
        price_inf_threhold = 0
        ce_indus = ce_values[0]
        trend_indus = ce_values[1]
        
        # 根据趋势预测数据在历史数据中的相对位置，并参照历史实际数据走势，判断涨跌趋势。并综合多个行业判断整体涨跌趋势。
        trend_flag_indus = []
        result_list = []
        price_round_data_indus = []
        trend_can_num = 1
        for i,idx in enumerate(industry_index):
            index_round_targets_item = index_round_targets[idx][-self.cut_len-self.output_chunk_length:-self.output_chunk_length,1]
            target_info_item = target_info[industry_index_real[idx]]
            price_round_data = target_info_item["diff_range"][-self.cut_len-self.output_chunk_length:-self.output_chunk_length]
            # 取得最接近的匹配索引，然后查找到对应的价格涨跌幅进行判断
            match_index = np.argsort(trend_indus[i])[:trend_can_num]
            # match_index = np.abs(index_round_targets_item-pred_data[-1]).argmin()
            price_indus_inf = np.mean(price_round_data[match_index])
            # trend_flag = np.sum(price_round_data[match_index]>price_inf_threhold)>(trend_can_num/2)
            trend_flag = np.sum(price_round_data[match_index])>0
            trend_flag_indus.append(trend_flag)
            result_list.append([date,industry_index_real[idx],trend_flag,price_indus_inf,price_indus_inf])
        price_round_data_indus = np.array(price_round_data_indus)
        trend_real = (np.sum(price_round_data_indus>0)>(len(industry_index)//2))+0
        
        # 构建结果集
        result_list = np.array(result_list)
        result_list = pd.DataFrame(result_list,columns=self.result_columns)
        result_list['date'] = result_list['date'].astype(int) 
        result_list['trend_flag'] = result_list['trend_flag'].astype(int) 
        result_list['indus_index'] = result_list['indus_index'].astype(int) 
                    
        trend_flag_indus = np.array(trend_flag_indus)
        price_indus_inf_list = result_list['price_inf'].values
        
        # max_index = np.abs(price_indus_inf_list).argmax()
        # trend_value = (price_indus_inf_list[max_index]>0)
        # trend_value = (price_indus_inf_list.sum()>0)
        # 超出一半上涨，则认为整体上涨
        trend_value = (np.sum(trend_flag_indus)>(len(industry_index)//2))+0
        # trend_value = (np.sum(cls_ins>0.5)>(len(industry_index)//2))+0
        # Make All Trend Correct
        # trend_value = trend_real
        
        # 分别取得最高和最低，同时进行多方和空方操作
        raise_top_index = np.argsort(-ce_indus)[0] 
        fall_top_index = np.argsort(ce_indus)[0]   
        
        # 整体预测均值和阈值比较，决定整体使用多方还是空方作为预测方向
        if trend_value==1:
            indus_top_index = raise_top_index
        else:
            indus_top_index = fall_top_index
        
        indus_rel_index = industry_index[indus_top_index]
        indus_rel_index_raise = industry_index[raise_top_index]
        indus_rel_index_fall = industry_index[fall_top_index]
        # 转换为实际的索引
        indus_top_index_real = industry_index_real[indus_rel_index]
        indus_raise_index_real = industry_index_real[indus_rel_index_raise]
        indus_fall_index_real = industry_index_real[indus_rel_index_fall]
        
        # 涨跌top索引，写入结果集
        result_list['top_flag'] = -1
        result_list.loc[result_list['indus_index']==indus_raise_index_real,'top_flag'] = 1
        result_list.loc[result_list['indus_index']==indus_fall_index_real,'top_flag'] = 0
                
        return indus_top_index_real,trend_value,result_list        
        
    
    def strategy_top(self,cls,ce_values,target=None,target_info=None,index_round_targets=None,result_list=None,indus_top_index=None,trend_value=None,combine_instrument=None):
        """筛选品种明细"""
        
        date = int(target_info[0]['future_start_datetime'])
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        industry_index = [i for i in range(sw_ins_mappings[:,0].shape[0])]
        industry_index.remove(main_index)
        
        cls_ins = cls[0]
        
        top_num = 3
        select_num = 3
        indus_rel_index = industry_index[np.where(indus_index==indus_top_index)[0][0]]
        ins_rel_index = FuturesMappingUtil.get_indus_instrument_range_index(sw_ins_mappings,indus_rel_index)
        cls_can = cls_ins[ins_rel_index]
        
        # 取得对应行业下的品种排名，并作为候选
        if trend_value==0:
            # 看CLS指标的空方
            pre_index = np.argsort(cls_can)[:top_num]  
            # 取得RSV排序靠前的记录，从而进行空方判断
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)            
        else:
            # 取得CLS反向排序靠前的记录，从而进行多方判断
            pre_index = np.argsort(-cls_can)[:top_num]       
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)
        pred_import_index = np.array(pred_import_index)
        pred_import_index = pred_import_index[:select_num]
        import_index_real = FuturesMappingUtil.get_instrument_rel_index_within_industry(sw_ins_mappings, indus_rel_index)[pred_import_index].astype(int)
        
        return import_index_real

    def strategy_top_bidi(self,cls,ce_values,target=None,target_info=None,index_round_targets=None,date=None,combine_instrument=None):
        """行业排名方式筛选候选者,双向模式"""
        
        date = int(target_info[0]['future_start_datetime'])
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        ins_index = FuturesMappingUtil.get_all_instrument(sw_ins_mappings)

        cls_ins = cls[0]
        top_num = 5
        
        # 分别取得最高和最低，同时进行多方和空方操作
        raise_top_index = np.argsort(-cls_ins)[:top_num]
        fall_top_index = np.argsort(cls_ins)[:top_num]   
        
        raise_index_real = ins_index[raise_top_index]
        fall_index_real = ins_index[fall_top_index]

        result_list = []
        for item in raise_index_real:
            result_list.append([item,1])
        for item in fall_index_real:
            result_list.append([item,0])            
        
        # 构建结果集
        result_list = np.array(result_list)
        result_list = pd.DataFrame(result_list,columns=['top_index','top_flag'])
        result_list['date'] = date
               
        return result_list['top_index'].values,result_list
    
    def strategy_top_all(self,cls,ce_values,target=None,target_info=None,index_round_targets=None,result_list=None,indus_top_index=None,trend_value=None,combine_instrument=None):
        """筛选品种明细"""
        
        date = int(target_info[0]['future_start_datetime'])
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        main_index = FuturesMappingUtil.get_main_index_in_indus(sw_ins_mappings)
        indus_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
        industry_index = [i for i in range(sw_ins_mappings[:,0].shape[0])]
        industry_index.remove(main_index)
        
        cls_ins = cls[0]
        
        top_num = 5
        select_num = top_num
        indus_rel_index = industry_index[np.where(indus_index==indus_top_index)[0][0]]
        ins_rel_index = FuturesMappingUtil.get_indus_instrument_range_index(sw_ins_mappings,indus_rel_index)
        cls_can = cls_ins
        
        # 取得对应行业下的品种排名，并作为候选
        if trend_value==0:
            # 看CLS指标的空方
            pre_index = np.argsort(cls_can)[:top_num]  
            # 取得RSV排序靠前的记录，从而进行空方判断
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)            
        else:
            # 取得CLS反向排序靠前的记录，从而进行多方判断
            pre_index = np.argsort(-cls_can)[:top_num]       
            pred_import_index = []
            for index in pre_index:
                pred_import_index.append(index)
        pred_import_index = np.array(pred_import_index)
        pred_import_index = pred_import_index[:select_num]
        import_index_real = FuturesMappingUtil.get_instrument_rel_index_within_industry(sw_ins_mappings, main_index)[pred_import_index].astype(int)
        
        return import_index_real


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
            long_diff_index_targets,
            target_info
        ) = batch
               
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets,past_future_round_targets,index_round_targets)     
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        choice_out,trend_value,combine_index = vr_class
        
        # 只获取整体指数的价格数据
        whole_index_round_targets = index_round_targets[:,:,:-1,:]
        # 保存数据用于后续验证
        output_res = (output,choice_out.cpu().numpy(),trend_value.cpu().numpy(),combine_index.cpu().numpy(),past_target.cpu().numpy(),
                      future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),past_future_round_targets.cpu().numpy(),whole_index_round_targets.cpu().numpy(),
                      index_round_targets.cpu().numpy(),long_diff_index_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)        
         
    def on_predict_epoch_end(self,args):   
        """汇总预测数据，生成实际业务预测结果"""
        
        sw_ins_mappings = self.valid_sw_ins_mappings
        combine_content = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        result_date_list = self.combine_result_data(self.output_result,predict_mode=True)  
        result_target = {}
        # 根据原始数组，生成实际品种信息
        for date in list(result_date_list.keys()):
            res_arr = result_date_list[date].sort_values(by=['top_index'])
            res_index = res_arr['top_index']
            target = combine_content[np.isin(combine_content[:,0],res_index.values)]
            target = target[np.argsort(target[:,0])]
            res_arr['instrument'] = target[:,-1]
            result_target[date] = res_arr.copy()
        self.result_target = result_target
        
        return result_target
                         
