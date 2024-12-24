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
from darts_pro.act_model.mixer_fur_ts import FurTimeMixer
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.mixer_loss import FuturesCombineLoss
from cus_utils.common_compute import compute_average_precision,normalization
from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from tft.class_define import OVERROLL_TREND_UNKNOWN,OVERROLL_TREND_RAISE,OVERROLL_TREND_FALL2RAISE,OVERROLL_TREND_RAISE2FALL,OVERROLL_TREND_FALL
from cus_utils.tensor_viz import TensorViz

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from .mlp_module import MlpModule

TRACK_DATE = [20220524,20220520,20220516,20220718,20220519,20220804,20220923]
TRACK_DATE = [20220712,20220718]
DRAW_SEQ = [0,1,2]
DRAW_SEQ_DETAIL = [0]

class FuturesTogeModule(MlpModule):
    """期货品种和行业板块数据一起预测的模型"""
    
    def __init__(
        self,
        indus_dim: int,
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
        target_mode=None,
        batch_file_path=None,
        device="cpu",
        train_sw_ins_mappings=None,
        valid_sw_ins_mappings=None,
        **kwargs,
    ):
        self.indus_dim = indus_dim
        self.mode = None
        self.train_sw_ins_mappings = train_sw_ins_mappings
        self.valid_sw_ins_mappings = valid_sw_ins_mappings
        self.target_mode = target_mode
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
           
            # 使用混合时间序列模型
            model = FurTimeMixer(
                num_nodes=self.indus_dim, # 对应多变量数量（行业分类数量）
                seq_len=self.input_chunk_length,
                pred_len=self.output_chunk_length,
                down_sampling_window=self.output_chunk_length,
                past_cov_dim=input_dim,
                dropout=dropout,
                device=device,
                train_sw_ins_mappings=self.train_sw_ins_mappings,
                valid_sw_ins_mappings=self.valid_sw_ins_mappings,                
            )           
            
            return model

    def create_loss(self,model,device="cpu"):
        return FuturesCombineLoss(self.indus_dim,device=device,ref_model=model,target_mode=self.target_mode) 

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
            past_price_target =  x_in[4]
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                x_in_item = (past_convs_item,x_in[1],x_in[2],past_price_target)
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
    
    def get_optimizer_size(self):
        return len(self.past_split)
    
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
            price_targets,
            future_round_targets,
            last_targets,
            target_info
        ) = train_batch
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(self.get_optimizer_size()):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,future_round_targets,last_targets,target_info),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            # self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
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
            price_targets,
            future_round_targets,
            last_targets,
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,price_targets) 
        input_batch = self._process_input_batch(inp)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,future_round_targets,last_targets,target_info),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(self.past_split)):
            # self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_fds_loss_{}".format(i), fds_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        
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
        
        # 整合相关数据，分为输入值和目标值两组
        return (x_past_array, historic_future_covariates,future_covariates, static_covariates,price_targets,past_target)
               
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,future_round_targets,last_targets,target_info) = target   
        # 根据阶段使用不同的映射集合
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        return self.criterion(output,(future_target,target_class,future_round_targets,last_targets,target_info),
                    sw_ins_mappings=sw_ins_mappings,optimizers_idx=optimizers_idx)        

    def on_validation_epoch_end(self):
        """重载父类方法，修改指标计算部分"""
        # SANITY CHECKING模式下，不进行处理
        # if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
        #     return    
    
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        
        rate_total,total_imp_cnt = self.combine_result_data(self.output_result) 
        rate_total = dict(sorted(rate_total.items(), key=lambda x:x[0]))
        sr = []
        for item in list(rate_total.values()):
            if len(item)==0:
                continue
            item = np.array(item)
            # 如果空头预测，则倒序匹配准确率统计
            if item[-1]==0:
                item[:-2] = item[:-2][::-1]             
            sr.append(item)
                   
        if len(sr)>0:
            sr = np.stack(sr)  
            # 汇总计算准确率,取平均数
            sum_v = sr[:,-2]
            sr_rate = sr/sum_v[:, np.newaxis]
            combine_rate = np.mean(sr_rate,axis=0)
            # 按照日期计算最小准确率
            combine_rate_min = np.min(sr_rate,axis=0)
            for i in range(len(CLASS_SIMPLE_VALUES.keys())):
                self.log("score_{} rate".format(i), combine_rate[i], prog_bar=True) 
                # self.log("score_{} min rate".format(i), combine_rate_min[i], prog_bar=True) 
            self.log("total cnt", sr[:,-2].sum(), prog_bar=True)  
            self.log("total_imp_cnt", total_imp_cnt, prog_bar=True)  
            if self.mode is not None and self.mode.startswith("pred"):
                for date in rate_total.keys():
                    stat_data = rate_total[date]

        # 如果是测试模式，则在此进行可视化
        if self.mode is not None and self.mode.startswith("pred_"):
            tar_viz = global_var.get_value("viz_data")
            viz_result = global_var.get_value("viz_result")
            # viz_result_detail = global_var.get_value("viz_result_detail")
            
            viz_total_size = 0
                          
            output_3d,past_target_3d,future_target_3d,target_class_3d,price_targets_total, \
                future_round_targets_total,last_round_targets,target_info_3d = self.combine_output_total(self.output_result)
            
            indus_index = FuturesMappingUtil.get_industry_data_index_without_main(sw_ins_mappings)
            main_index = FuturesMappingUtil.get_main_index(sw_ins_mappings)
            indus_names = FuturesMappingUtil.get_industry_names(sw_ins_mappings)[:-1]
            indus_codes = FuturesMappingUtil.get_industry_codes(sw_ins_mappings)[:-1]  
            industry_instrument_index = FuturesMappingUtil.get_industry_instrument(sw_ins_mappings)
            
            viz_result_detail = {}
            for code in indus_codes:
                viz_result_detail[code] = TensorViz(env="viz_result_detail_{}".format(code))
                viz_result_detail[code].remove_env()
                viz_result_detail[code] = TensorViz(env="viz_result_detail_{}".format(code))
            viz_result_detail["all"] = TensorViz(env="viz_result_detail_all")
                 
            for index in range(target_class_3d.shape[0]):
                
                viz_total_size+=1
                keep_index = np.where(target_class_3d[index]>=0)[0]
                round_targets = future_round_targets_total[index]
                cls_output = output_3d[2]
                ts_arr = target_info_3d[index]
                date = ts_arr[keep_index][0]["future_start_datetime"]
                if not date in TRACK_DATE:
                    continue               

                ins_with_indus = FuturesMappingUtil.get_industry_instrument_exc_main(sw_ins_mappings)
                # 可视化相互关系
                for j in range(len(self.past_split)):
                    # 显示板块分类整体预测数据
                    indust_output_value = cls_output[index,indus_index,j]
                    indust_target = round_targets[indus_index,j]
                    main_value = cls_output[index,main_index,j]
                    main_target = round_targets[main_index,j]                    
                    win = "indus_round_target_{}_{}".format(j,viz_total_size)
                    target_title = "industry compare_{},date:{},pred_tar:{}_{}".format(j,date,round(main_value,2),round(main_target,2))
                    view_data = np.stack([indust_output_value,indust_target]).transpose(1,0)
                    # if j in DRAW_SEQ:
                    #     tar_viz.viz_bar_compare(view_data,win=win,title=target_title,rownames=indus_names.tolist(),legends=["pred","target"])   
                        
                    total_view_data = None
                    # 按照每个板块分类，分别显示板块内的品种整体预测数据
                    futures_names_combine = None
                    futures_index_combine = None
                    for k,instruments in enumerate(ins_with_indus):
                        instruments,k_idx,_ = np.intersect1d(instruments,keep_index,return_indices=True)
                        indus_code = indus_codes[k]
                        indus_name = indus_names[k]
                        futures_names = FuturesMappingUtil.get_futures_names(sw_ins_mappings,k)[k_idx].tolist()
                        ins_cls_output = cls_output[index,instruments,j]
                        fur_round_target = round_targets[instruments,j]
                        # 添加价格显示
                        price_array = np.array([ts_arr[h]["price_array"] for h in instruments])
                        scaler = MinMaxScaler(feature_range=(0.001, 1))
                        price_array = scaler.fit_transform(price_array.transpose(1,0)).transpose(1,0)
                        price_array_range = price_array[:,-1] - price_array[:,-self.output_chunk_length-1]                    
                        view_data = np.stack([ins_cls_output,fur_round_target,price_array_range]).transpose(1,0)
                        win = "round_target_{}_{}_{}".format(j,k,viz_total_size)
                        target_title = "target{}_{} pred_tar:{}_{},date:{}".format(j,indus_name,round(indust_output_value[k],2),round(indust_target[k],2),date)
                        # if j in DRAW_SEQ and len(futures_names)>1:
                        #     viz_result.viz_bar_compare(view_data,win=win,title=target_title,rownames=futures_names,legends=["pred","target"])   
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
                    # 合并所有品种并显示
                    if j in DRAW_SEQ:
                        win = "target_combt_{}_{}".format(j,viz_total_size)
                        target_title = "combine target{}_{},date:{}".format(j,np.mean(total_view_data[:,0]),date)
                        tar_viz.viz_bar_compare(total_view_data,win=win,title=target_title,rownames=futures_names_combine.tolist(),legends=["pred","target","price"])                       
                        
                    xbar_data = output_3d[0][index,...,j]
                    
                    def show_trend_data(k,indus_code,ins_name,type=0):
                        past_target_item = past_target_3d[index,k,:,j]
                        future_target_item = future_target_3d[index,k,:,j]
                        target_data = np.concatenate([past_target_item,future_target_item],axis=0)    
                        zero_index = np.where(target_data==0)
                        target_data[zero_index] = 0.001
                        xbar_data_item = xbar_data[k,:]                  
                        pad_data = np.array([0 for i in range(self.input_chunk_length)])
                        pred_data = np.concatenate((pad_data,xbar_data_item))
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
                        names=["pred","target","price"]
                        names=["target","price"]
                        viz_detail.viz_matrix_var(view_data,win=win,title=target_title,names=names)   
                     
                    # 显示预测走势数据       
                    if j in DRAW_SEQ_DETAIL:                            
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
                                                        
    def viz_results(self,output_inverse=None,target_inverse=None,import_price_result=None,batch_idx=0,target_vr_class=None,target_info=None,viz_target=None):
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        names = ["pred","label","price","obv_output","obv_tar","cci_output","cci_tar"]        
        names = ["price","macd_output","macd","rank_output","rank","qtlu_output","qtlu"]          
        result = []
              
        res_group = import_price_result.groupby("result")
        target_imp_index = np.where(target_vr_class==3)[0]
        if target_imp_index.shape[0]>0:
            for i in range(15):
                rand_index = np.random.randint(0,target_imp_index.shape[0]-1)
                s_index = target_imp_index[rand_index]
                ts = target_info[s_index]
                pred_data = output_inverse[s_index]
                pred_center_data = pred_data[:,0]
                pred_second_data = pred_data[:,1]         
                pred_third_data = pred_data[:,2]      
                target_item = target_inverse[s_index]
                win = "win_target_{}".format(batch_idx,i)
                self.draw_row(pred_center_data, pred_second_data, pred_third_data,target_item=target_item, ts=ts, names=names,viz=viz_target,win=win)
                               
    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,vr_class,price_targets,future_round_targets = outputs
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,_,_,last_targets,target_info) = val_batch
        # 保存数据用于后续验证
        output_res = (output,past_target.cpu().numpy(),future_target.cpu().numpy(),target_class.cpu().numpy(),
                      price_targets.cpu().numpy(),future_round_targets.cpu().numpy(),last_targets.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        # 使用全部验证结果进行统一比较
        output_3d,past_target_3d,future_target_3d,target_class_3d,last_targets_3d,future_indus_targets,future_round_targets,target_info_3d  = self.combine_output_total(output_result)
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
            # 去除指数整体及行业
            keep_index = np.intersect1d(keep_index,instrument_index)   
            output_list = [output_3d[0][i][keep_index],output_3d[2][i][keep_index],output_3d[3][i]]
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
            # Compute Acc Result
            import_price_result = self.collect_result(import_index,target_class=target_class_list, target_info=target_info_list[keep_index])
            
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
        print("result:",err_date_list)      
        
        return rate_total,total_imp_cnt
    
    def compute_top_map(self,output_list,future_target,last_target_list,indus_targets=None):
        """分别计算行业分类排序准确率和分类内股票排序平均精度"""
        
        industry_topk = 6
        instrument_topk = 5
        acc = []
        # 排序正反参数
        sort_flag = [-1,1,-1,1,1]
        sv_item = output_list[0]
        cls_list = output_list[1]
        pred_range = sv_item[:,-1,:] - sv_item[:,0,:]
        tar_range = future_target[:,-1,:] - future_target[:,0,:]
        ind_score = [0 for _ in range(last_target_list.shape[-1])]
        for i in range(indus_targets.shape[-1]):
            match_cls = cls_list[:,i] * sort_flag[i]
            match_target = indus_targets[:,i] * sort_flag[i]
            ind_score[i] = compute_average_precision(match_cls,match_target,topk=industry_topk)        
        # for i in range(sv_item.shape[-1]):
        #     ind_score[i] = compute_average_precision(pred_range[:,i],tar_range[:,i],topk=industry_topk)
        return ind_score

                
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
        last_targets_total = []
        for item in output_result:
            (output,past_target,future_target,target_class,price_targets,future_round_targets,last_targets,target_info) = item
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

        target_class_total = np.concatenate(target_class_total)
        past_target_total = np.concatenate(past_target_total)
        future_target_total = np.concatenate(future_target_total)
        price_targets_total = np.concatenate(price_targets_total)
        future_round_targets_total = np.concatenate(future_round_targets_total)
        last_targets_total = np.concatenate(last_targets_total)
        target_info_total = np.concatenate(target_info_total)
                    
        return (x_bar_total,sv_total,cls_total,ce_index_total),past_target_total,future_target_total,target_class_total, \
                    price_targets_total,future_round_targets_total,last_targets_total,target_info_total        

    def build_import_index(self,output_data=None,target=None,target_info_list=None):  
        """生成涨幅达标的预测数据下标"""
        
        return None,None
        
        (fea_values,cls_values,ce_values) = output_data
        price_array = np.array([item["price_array"] for item in target_info_list])
        
        
        pred_import_index,overroll_trend = self.strategy_top(fea_values,cls_values,ce_values,target=target,price_array=price_array,target_info=target_info_list)
        # pred_import_index = self.strategy_threhold(sv_values,(fea_0_range,fea_1_range,fea_2_range),rank_values,batch_size=self.ins_dim)
            
        return pred_import_index,overroll_trend
        
    def strategy_threhold(self,sv,fea,cls,batch_size=0):
        sv_0 = sv[...,0]
        # 使用回归模式，则找出接近或大于目标值的数据
        sv_import_bool = (sv_0<-0.1) # & (sv_1<-0.02) #  & (sv_2>0.1)
        pred_import_index = np.where(sv_import_bool)[0]
        return pred_import_index
 
    def strategy_top(self,fea_values,cls,ce,target=None,price_array=None,target_info=None):
        """排名方式筛选候选者"""

        future_target = target[:,self.input_chunk_length:,:]
        past_target = target[:,:self.input_chunk_length,:]
        price_recent = price_array[:,:self.input_chunk_length]
        price_recent = MinMaxScaler().fit_transform(price_recent.transpose(1,0)).transpose(1,0)      
        rsi_past_target = past_target[...,0]
        rsv_past_target = past_target[...,1]
                
        # 排整体涨跌判断
        overroll_trend = self.judge_overall_trend(fea_values, cls, ce, target=future_target[0], past_target=past_target[0],price_recent=price_recent[0])
        
        cls_rsi = cls[...,0]
        cls_price = cls[...,1]
        # cls_3 = cls[...,3]

        ### 整体策略：首先检查整体趋势预测，然后根据趋势分别进行应对策略 ###
        if overroll_trend==OVERROLL_TREND_RAISE:
            rsi_per_throld = 30
            price_per_throld = 8            
            # 在整体确定上涨的时候,看price行业预测指标,进行多头判断,同时兼顾rsi行业预测指标
            pred_import_index = []
            # 分位数取得少数部分作为阈值
            rsi_per_value = np.percentile(cls_rsi, [100-rsi_per_throld])[0]
            price_per_value = np.percentile(cls_price, [100-price_per_throld])[0]
            rsi_pred_import_pre = np.where(cls_rsi>rsi_per_value)[0]
            price_pred_import_pre = np.where(cls_price>price_per_value)[0]
            pred_import_pre = price_pred_import_pre # np.intersect1d(rsi_pred_import_pre,price_pred_import_pre)
            pred_import_pre = pred_import_pre[np.argsort(cls_rsi[pred_import_pre])]            
            for index in pred_import_pre:
                # 使用RSI目标前值比较低的数据
                if rsi_past_target[index,-1]>0.3 and False:
                    continue
                pred_import_index.append(index)
        elif overroll_trend==OVERROLL_TREND_FALL:
            rsi_per_throld = 10
            price_per_throld = 50            
            # 在整体确定下跌的时候,看rsi行业预测指标,进行空头判断,同时兼顾price行业预测指标
            pred_import_index = []
            # 分位数取得少数部分作为阈值
            rsi_per_value = np.percentile(cls_rsi, [rsi_per_throld])[0]
            price_per_value = np.percentile(cls_price, [price_per_throld])[0]
            rsi_pred_import_pre = np.where(cls_rsi<rsi_per_value)[0]
            ref_pred_import_pre = np.where(cls_price<price_per_value)[0]
            pred_import_pre = np.intersect1d(rsi_pred_import_pre,ref_pred_import_pre)
            pred_import_pre = pred_import_pre[np.argsort(cls_rsi[pred_import_pre])]
            for index in pred_import_pre:
                # 使用cntp指标再次筛选，需要也在cntp指标中排序靠前
                if rsi_past_target[index,-1]>0.3 and False:
                    continue
                pred_import_index.append(index)
        pred_import_index = pred_import_index[:3]
        # instrument_info = self._get_instrument_name(pred_import_index,target_info)                                                  
        return pred_import_index,overroll_trend
    
    def _get_instrument_name(self,rank_index,target_info):
        
        sw_ins_mappings = self.train_sw_ins_mappings if self.trainer.state.stage==RunningStage.TRAINING else self.valid_sw_ins_mappings
        combine_industry = FuturesMappingUtil.get_combine_industry_instrument(sw_ins_mappings)
        infos = []
        for index in rank_index:
            instrument_code = target_info[index]['instrument']
            instrument_info = combine_industry[np.where(combine_industry[:,3]==instrument_code)[0]][0]
            infos.append(instrument_info)
        return np.stack(infos)
    
    def price_trend_ana(self,price_recent):
        """针对历史价格数据，分析历史价格走势 1上涨 2下跌 3震荡"""
        
        price_recent_range = price_recent[1:] - price_recent[:-1]
        
        if price_recent[-1]>0.6 and (price_recent[-1] - price_recent[-5])>0.1 and np.sum(price_recent_range[-5:]>0)>=3:
            # 上涨：最近值处于比较高位，并且超出前面的数值一定范围,并且最近走势中大多数为上涨
            return 1
        if price_recent[-1]>0.8 and (price_recent[-1] - price_recent[-5])>0 and np.sum(price_recent_range[-5:]>0)>=3:
            # 上涨：最近值处于高位，并且最近走势中大多数为上涨
            return 1        
        if price_recent[-1]<0.5 and (price_recent[-1] - price_recent[-5])<-0.1 and np.sum(price_recent_range[:-5]<0)>=3:
            # 下跌：最近值处于相对低位，并且低于前面的数值一定范围,并且最近走势中大多数为下跌
            return 2        
        return 3
     
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

    def collect_result(self,import_index,target_class=None,target_info=None): 
        """收集预测对应的实际数据"""

        # 对于预测数据，生成对应涨跌幅类别
        import_price_result = []
        for i,imp_idx in enumerate(import_index):
            ts = target_info[imp_idx]
            price_array = ts["price_array"][self.input_chunk_length-1:]
            p_taraget_class = compute_price_class(price_array,mode="first_last")
            import_price_result.append([imp_idx,ts["instrument"],p_taraget_class])       
        import_price_result = np.array(import_price_result)  
        if import_price_result.shape[0]==0:
            return None
        import_price_result = pd.DataFrame(import_price_result,columns=["imp_index","instrument","result"])     
        import_price_result["result"] = import_price_result["result"].astype(np.int64)      
            
        return import_price_result


