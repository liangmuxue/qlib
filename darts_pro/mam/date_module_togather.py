import os

import pickle
import sys
import numpy as np
import pandas as pd
import torch
import tsaug

import torchvision
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning.callbacks as pl_callbacks
from torch.utils.data import DataLoader
from torch.distributions import Normal
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from cus_utils.tensor_viz import TensorViz

import cus_utils.global_var as global_var
from darts_pro.act_model.mlp_ts import MlpTs,MlpTs3D
from cus_utils.metrics import pca_apply
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.clustering_loss import Mlp3DLoss
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
from cus_utils.visualization import ShowClsResult
from losses.quanlity_loss import QuanlityLoss

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from .mlp_module import MlpModule


class MlpDateAlignModule(MlpModule):
    """MlpModule for Data Togather with Per Date """
    
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
        step_mode="normal",
        batch_file_path=None,
        static_datas=None,
        device="cpu",
        **kwargs,
    ):
        self.ins_dim = ins_dim
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
                _
            ) = self.train_sample
                  
            dataset = global_var.get_value("dataset")
            
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
    
            nr_params = 1
            self.pca_dim = 10
            
            model = MlpTs(
                # Tide Part
                input_dim=input_dim,
                emb_output_dim=output_dim,
                future_cov_dim=future_cov_dim,
                static_cov_dim=static_cov_dim,
                nr_params=nr_params,
                num_encoder_layers=3,
                num_decoder_layers=3,
                decoder_output_dim=16,
                hidden_size=hidden_size,
                temporal_width_past=4,
                temporal_width_future=4,
                temporal_decoder_hidden=32,
                use_layer_norm=True,
                dropout=dropout,
                # Mlp Part
                enc_nr_params=len(QuanlityLoss().quantiles),
                n_cluster=len(CLASS_SIMPLE_VALUES.keys()),
                pca_dim=self.pca_dim,
                device=device,
                **kwargs,
            )           
            
            return model

    def create_loss(self,model,device="cpu"):
        return Mlp3DLoss(self.ins_dim,device=device,ref_model=model) 
    
    def on_train_epoch_start(self):  
        self.loss_data = []
    def on_train_epoch_end(self):  
        pass
        
    def on_validation_epoch_start(self):  
        self.import_price_result = None
        self.total_imp_cnt = 0
                    
    def training_step_real(self, train_batch, batch_idx): 
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            future_target,
            target_class,
            target_info
        ) = train_batch
        
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
            target_info
        ) = self.transfer_to_2d(train_batch)
                
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class)
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,last_targets),optimizers_idx=i)
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

        # 全部转换为2维模式进行网络计算
        val_batch_transfer = self.transfer_to_2d(val_batch)
                
        loss,detail_loss,output = self.validation_step_real(val_batch_transfer, batch_idx)  
       
        # if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag:
        self.dump_val_data(val_batch_transfer,output,detail_loss)
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
            target_info
        ) = val_batch
              
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates) 
        input_batch = self._process_input_batch(inp)
        last_targets,pca_target,_ = self._process_target_batch(future_target,target_class)
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,last_targets),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_QTLU_loss", (ce_loss[0]+corr_loss_combine[0]), batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,pca_target)
        return loss,detail_loss,output_combine
            
    def transfer_to_2d(self,batch_data):
        """把相关变量转换为2维模式"""

        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            past_future_covariates,
            future_target,
            target_class,
            target_info
        ) = batch_data    
        
        past_target_2d = past_target.reshape(-1,past_target.shape[-2],past_target.shape[-1])
        past_covariates_2d = past_covariates.reshape(-1,past_covariates.shape[-2],past_covariates.shape[-1])
        historic_future_covariates_2d = historic_future_covariates.reshape(-1,historic_future_covariates.shape[-2],historic_future_covariates.shape[-1])
        future_covariates_2d = future_covariates.reshape(-1,future_covariates.shape[-2],future_covariates.shape[-1])
        static_covariates_2d = static_covariates.reshape(-1,static_covariates.shape[-1])
        past_future_covariates_2d = past_future_covariates.reshape(-1,past_future_covariates.shape[-2],past_future_covariates.shape[-1])
        future_target_2d = future_target.reshape(-1,future_target.shape[-2],future_target.shape[-1])
        target_class_2d = target_class.reshape(-1)
        target_info_2d = [i for k in target_info for i in k]

        return (
            past_target_2d,
            past_covariates_2d,
            historic_future_covariates_2d,
            future_covariates_2d,
            static_covariates_2d,
            past_future_covariates_2d,
            future_target_2d,
            target_class_2d,
            target_info_2d
        )
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,last_target) = target   
        return self.criterion(output,(future_target,target_class,last_target),mode=self.step_mode,optimizers_idx=optimizers_idx)        

    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,_ = outputs
        (past_target,past_covariates,historic_future_covariates,future_covariates,
            static_covariates,past_future_covariates,future_target,target_class,target_info) = val_batch
        # 目标数据合并到一起  
        target = np.concatenate([past_target.cpu().numpy(),future_target.cpu().numpy()],axis=1)
        # 保存数据用于后续验证
        output_res = (output,target,target_class.cpu().numpy(),target_info)
        self.output_result.append(output_res)

    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # 使用全部验证结果进行统一比较
        output_total,target_total,target_class_total,target_info_total = self.combine_output_total(output_result)

        # 按照日期分组计算
        fur_dates = {}
        for index,ti in enumerate(target_info_total):
            if ti is None:
                continue
            future_start_datetime = ti["future_start_datetime"].item()
            if not future_start_datetime in fur_dates.keys():
                fur_dates[future_start_datetime] = [index]
            else:
                fur_dates[future_start_datetime].append(index)
        fur_dates_filter = {}
        for date in fur_dates.keys():
            # if date>=20220401 or date<20220301:
            #     continue    
            fur_dates_filter[date] = fur_dates[date]   
        fur_dates = fur_dates_filter  
        # 生成目标索引
        import_index_all,values = self.build_import_index(output_data=output_total,fur_dates=fur_dates,target_info=target_info_total)
        rate_total = {}
        total_imp_cnt = np.where(target_class_total==3)[0].shape[0]
        # 对每天的准确率进行统计，并累加
        for date in np.sort(np.array(list(import_index_all.keys()))):
            date = date.item()
            import_index = import_index_all[date]
            if len(import_index)==0:
                continue
            import_acc, import_recall,import_price_acc,import_price_nag,price_class, \
                import_price_result = self.collect_result(import_index, target_class_total, target_info_total)
            
            score_arr = []
            score_total = 0
            rate_total[date] = []
            if import_price_result is not None:
                res_group = import_price_result.groupby("result")
                ins_unique = res_group.nunique()
                total_cnt = ins_unique.values[:,1].sum()
                for i in range(4):
                    cnt_values = ins_unique[ins_unique.index==i].values
                    if cnt_values.shape[0]==0:
                        cnt = 0
                    else:
                        cnt = cnt_values[0,1].item()
                    rate_total[date].append(cnt)
                # 预测数量以及总数量
                rate_total[date].append(total_cnt.item())              
        sr = np.array(list(rate_total.values()))
        
        # with open("custom/data/pred/import_index_all.pkl", "wb") as fout:
        #     pickle.dump(import_index_all, fout)          
        # write_json(rate_total,"custom/data/pred/rate_total.json")
        
        return sr,total_imp_cnt,import_index_all
            
    def combine_output_total(self,output_result):
        """重载父类方法，以适应整合数据"""
        
        target_class_total = []
        target_info_total = []
        target_total = []        
        output_total = [[[] for _ in range(5)] for _ in range(len(self.past_split))]
        for item in output_result:
            (output,target,target_class,target_info) = item
            for i in range(len(output_total)):
                output_item = output[i]
                x_bar,z,cls,tar_cls,x_smo = output_item 
                output_total[i][0].append(x_bar)
                output_total[i][1].append(z)
                output_total[i][2].append(cls)
                output_total[i][3].append(tar_cls)
                output_total[i][4].append(x_smo)
            target_info_total = target_info_total + target_info
            target_class_total.append(target_class)
            target_total.append(target)
        for i in range(len(output_total)):
            output_total[i][0] = torch.concat(output_total[i][0])
            output_total[i][1] = torch.concat(output_total[i][1])
            output_total[i][2] = torch.concat(output_total[i][2])
            # output_total[i][3] = np.concatenate(output_total[i][3])
            output_total[i][4] = torch.concat(output_total[i][4])       

        target_class_total = np.concatenate(target_class_total)
        target_total = np.concatenate(target_total)
                    
        return output_total,target_total,target_class_total,target_info_total        
                 