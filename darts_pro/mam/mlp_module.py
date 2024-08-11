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

from darts_pro.act_model.mlp_ts import MlpTs
from cus_utils.metrics import pca_apply
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.clustering_loss import MlpLoss
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
from cus_utils.visualization import ShowClsResult
from losses.quanlity_loss import QuanlityLoss
import cus_utils.global_var as global_var

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_module import _TFTModuleBatch

class MlpModule(_TFTModuleBatch):
    """自定义基于DNN模式的时间序列模块"""
    
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
        use_weighted_loss_func=False,
        past_split=None,
        step_mode="normal",
        batch_file_path=None,
        static_datas=None,
        device="cpu",
        **kwargs,
    ):
        self.static_datas = static_datas
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,batch_file_path=batch_file_path,
                                    device=device,**kwargs)  
        self.output_data_len = len(past_split)
        self.switch_epoch_num = 0
        # 切换标志，决定训练阶段是否只计算不回传梯度的方式进行,0训练模式1计算模式
        self.switch_flag = 0
        # 模型训练模式
        self.step_mode=step_mode
        self.mode = None
        # 初始化中间结果数据
        self.training_step_outputs = [[] for _ in range(self.output_data_len)]
        self.training_step_targets = [[] for _ in range(self.output_data_len)]
        
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
                past_covariates,
                historic_future_covariate,
                future_covariates,
                static_covariates,
                target_class,
                future_target 
            ) = self.train_sample      
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]            
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
                future_covariates.shape[-1] if future_covariates is not None else 0
            )
            
            static_cov_dim = (
                static_covariates.shape[-2] * static_covariates.shape[-1] - 1
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
        
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """整合多种模型，主要使用MLP方式"""
        
        out_total = []
        out_class_total = []
        (batch_size,_,_) = x_in[1].shape
        
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]            
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                x_in_item = (past_convs_item,x_in[1],x_in[2])
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


    def create_loss(self,model,device="cpu"):
        return MlpLoss(device=device,ref_model=model) 

    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        (future_target,target_class,last_target,pca_target) = target   
        return self.criterion(output,(future_target,target_class,last_target,pca_target),mode=self.step_mode,optimizers_idx=optimizers_idx)

    def on_validation_start(self): 
        self.output_result = []
        super().on_validation_start()

    def on_train_epoch_start(self):  
        # 切换到平滑标签模式
        if self.current_epoch>=0:
            self.step_mode = "smooth"
        # 训练包括2个轮次，在此进行标记
        if self.current_epoch%2==0:
            self.switch_flag = 0
        else:
            self.switch_flag = 1
        super().on_train_epoch_start()

    def on_train_epoch_end(self):  
        """FDS模式，更新相关参数"""
        
        pass
        # self.custom_histogram_adder()
        
        # with torch.no_grad():
        #     for i in range(len(self.sub_models)):
        #         encodings = torch.concat(self.training_step_outputs[i]).detach()
        #         training_labels = torch.concat(self.training_step_targets[i]).detach()
        #         model = self.sub_models[i]
        #         model.FDS.update_last_epoch_stats(self.current_epoch)
        #         model.FDS.update_running_stats(encodings, training_labels, self.current_epoch)
                                
    def output_postprocess(self,output,targets,index):
        """用于训练阶段计算模式"""
        
        # 需要编码数据,吗，以及目标类别标签
        output_act = output[index]
        self.training_step_outputs[index].append(output_act[1])
        self.training_step_targets[index].append(targets) 

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """重载原方法，直接使用已经加工好的数据"""

        if self.mode=="pred_batch":
            # 测试模式，不进行训练
            print("no training for pred batch")
            return
        
        loss,detail_loss,output = self.training_step_real(train_batch, batch_idx) 
        # (mse_loss,value_diff_loss,corr_loss,ce_loss,mean_threhold) = detail_loss
        return loss
    
    def training_step_real(self, train_batch, batch_idx) -> torch.Tensor:
        """包括第一及第二部分数值数据,以及分类数据"""

        # 收集目标数据用于分类
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,target_class,future_target) = train_batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates)     
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class[:,0])
        target_class = target_class[:,:,0]     
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,last_targets,pca_target),optimizers_idx=i)
            (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
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
        
        # if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag:
        self.dump_val_data(val_batch,output,detail_loss)
        return loss,detail_loss
           
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,future_past_covariate,target_class,future_target,target_info,price_target) = val_batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates) 
        input_batch = self._process_input_batch(inp)
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class[:,0])
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)
        
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,last_targets,pca_target),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_fds_loss_{}".format(i), fds_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_CNTN_loss", (ce_loss[1]+corr_loss_combine[1]), batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,pca_target)
        return loss,detail_loss,output_combine
    
    def pca_viz(self,pca_target,target_class,output,batch_idx=0,root_path="custom/data/results/pred"):
        # 目标类别分类评估
        pca_output = [output_item[1] for output_item in output]
        pca_output = torch.stack(pca_output).permute(1,2,0)       
        for i in range(pca_target.shape[-1]):
            self.pca_viz_item(i,pca_target[...,i],pca_output[...,i],target_class=target_class,batch_idx=batch_idx,root_path=root_path)
    
    def pca_viz_item(self,index,pca_target,pca_output,target_class=None,batch_idx=0,root_path=None):
        pred,w = self.sub_models[index].predict_pca_cls(pca_target)
        # 可视化权重分割情况
        path = "{}/{}".format(root_path,index)
        if not os.path.exists(path):
            os.makedirs(path)                 
        save_file = "{}/pred_{}-{}.png".format(path,self.current_epoch,batch_idx)
        ShowClsResult(w[0].transpose(),np.expand_dims(w[1],axis=0),pca_output.cpu().numpy(),target_class,save_file=save_file)
                     
    def build_import_index(self,output_data=None,target_info=None,fur_dates=None):  
        """生成涨幅达标的预测数据下标"""
        
        cls_values = []
        fea_values = []
        pca_values = []
        smooth_values = []
        for i in range(len(output_data)):
            output_item = output_data[i] 
            x_bar,z,cls,_,x_smo =  output_item 
            cls_values.append(cls)
            fea_values.append(x_bar)
            pca_values.append(z)
            smooth_values.append(x_smo)
        
        cls_values = torch.stack(cls_values).cpu().numpy().transpose(1,2,0)
        fea_values = torch.stack(fea_values).cpu().numpy().transpose(1,2,0)
        pca_values = torch.stack(pca_values).cpu().numpy().transpose(1,2,0)
        smooth_values = torch.stack(smooth_values).cpu().numpy().transpose(1,2,0)

        fea_0 = fea_values[...,0]
        fea_0_range = (fea_0[:,-1] - fea_0[:,0])       
        fea_1 = fea_values[...,1]
        fea_1_range = (fea_1[:,-1] - fea_1[:,0])
        fea_2 = fea_values[...,2]
        fea_2_range = (fea_2[:,-1] - fea_2[:,0])
        
        # 按照日期分组进行计算
        pred_import_index_all = {}
        if fur_dates is None:
            pred_import_index = self.strategy_threhold(smooth_values,(fea_0_range,fea_1_range,fea_2_range),cls_values,batch_size=cls_values.shape[0])
            return pred_import_index,(cls_values,fea_values,pca_values)  
            
        for date in fur_dates.keys():
            # if date>=20220901 or date<20220801:
            #     continue
            idx = fur_dates[date]
            # pred_import_index = self.strategy_top(smooth_values[idx],(fea_0_range[idx],fea_1_range[idx],fea_2_range[idx]),cls_values[idx],batch_size=cls_values.shape[0])
            pred_import_index = self.strategy_threhold(smooth_values[idx],(fea_0_range[idx],fea_1_range[idx],fea_2_range[idx]),cls_values[idx],batch_size=len(idx))
            # 通过传统指标进行二次筛选
            target_info_cur = np.array(target_info)[idx]
            singal_index_bool = self.create_signal_macd(target_info_cur)
            # singal_index_bool = self.create_signal_rsi(target_info_cur)
            # singal_index_bool = self.create_signal_kdj(target_info_cur)
            singal_index = np.array(idx)[np.where(singal_index_bool)[0]]
            pred_index = np.array(idx)[pred_import_index]
            pred_import_index_all[date] = np.intersect1d(singal_index,pred_index)
            
        return pred_import_index_all,(cls_values,fea_values,pca_values)       

    def create_signal_macd(self,target_info):
        """macd指标判断"""
        
        diff_cov = np.array([item["macd_diff"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        dea_cov = np.array([item["macd_dea"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即diff快线向上突破dea慢线
        index_bool = (np.sum(diff_cov[:,:-2]<=dea_cov[:,:-2],axis=1)>=6) & (np.sum(diff_cov[:,-5:]>=dea_cov[:,-5:],axis=1)>=2)
        return index_bool

    def create_signal_rsi(self,target_info):
        """rsi指标判断"""
        
        rsi5_cov = np.array([item["rsi_5"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        rsi20_cov = np.array([item["rsi_20"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即rsi快线向上突破rsi慢线
        index_bool = (np.sum(rsi5_cov[:,:-2]<=rsi20_cov[:,:-2],axis=1)>=6) & (np.sum(rsi5_cov[:,-5:]>=rsi20_cov[:,-5:],axis=1)>=2)
        return index_bool

    def create_signal_kdj(self,target_info):
        """kdj指标判断"""
        
        k_cov = np.array([item["kdj_k"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        d_cov = np.array([item["kdj_d"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        j_cov = np.array([item["kdj_j"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即k线向上突破d线
        index_bool = (np.sum(k_cov[:,:-2]<=d_cov[:,:-2],axis=1)>=5) & (np.sum(k_cov[:,-5:]>=d_cov[:,-5:],axis=1)>=1)
        # 突破的时候d线也是向上的
        j_slope = j_cov[:,1:] - j_cov[:,:-1]
        return index_bool
           
    def strategy_threhold(self,sv,fea,cls,batch_size=0):
        cls_0 = cls[...,0]
        cls_1 = cls[...,1]
        cls_2 = cls[...,2]
        sv_0 = sv[...,0].squeeze(-1)
        sv_1 = sv[...,1].squeeze(-1)
        sv_2 = sv[...,2].squeeze(-1)
        (fea_0_range,fea_1_range,fea_2_range) = fea
        # 使用回归模式，则找出接近或大于目标值的数据
        sv_import_bool = (sv_1<-0.1) & (sv_2>0)
        # ce_thre_para = [[0.1,6],[-0.1,7],[-0.1,6]]
        # ce_para2 = ce_thre_para[2]
        # sv_import_bool = (np.sum(sv_2<ce_para2[0],1)>ce_para2[0])
        # sv_import_bool = (sv_2<0) & (sv_1>0) & (fea_1_range<-1)
        # 分位数回归模式下的阈值选择
        cls_thre_para = [[0.1,8],[0,8],[-0,7]]
        # 包含2个参数：分数阈值以及个数阈值
        para0 = cls_thre_para[0]
        para1 = cls_thre_para[1]
        para2 = cls_thre_para[2]
        cls_import_bool = (np.sum(cls_1>para1[0],1)>para1[1]) # & (np.sum(cls_2<para2[0],1)>para2[0]) # & (np.sum(cls_0>para0[0],1)>para0[0]) 
        pred_import_index = np.where(sv_import_bool)[0]
        
        return pred_import_index

    def strategy_top(self,sv,fea,cls,batch_size=0):
        """排名方式筛选候选者"""
        
        cls_0 = cls[...,0]
        cls_1 = cls[...,1]
        cls_2 = cls[...,2]
        sv_0 = sv[...,0].squeeze(-1)
        sv_1 = sv[...,1].squeeze(-1)
        sv_2 = sv[...,2].squeeze(-1)
        (fea_0_range,fea_1_range,fea_2_range) = fea
        
        top_k = sv_0.shape[0]//4
        # 使用2号进行sv判断（最后一段涨跌幅度），逆序
        sv_import_index = np.intersect1d(np.argsort(-sv_0)[:top_k],np.argsort(fea_1_range)[:top_k])
        # 使用0号进行corr判断（整体涨跌幅度），正序
        fea0_import_index = np.argsort(-fea_0_range)[:top_k]
        # 使用1号进行corr判断（整体涨跌幅度），逆序
        fea1_import_index = np.argsort(fea_1_range)[:top_k]        
        fea2_import_index = np.argsort(-fea_2_range)[:top_k]   
        # comp1_index = np.intersect1d(sv_import_index,fea0_import_index)
        comp1_index =  np.intersect1d(sv_import_index,fea2_import_index)

        # cls_thre_para = [[0.1,8],[-0,8],[-0,7]]
        # # 包含2个参数：分数阈值以及个数阈值
        # para0 = cls_thre_para[0]
        # para1 = cls_thre_para[1]
        # para2 = cls_thre_para[2]
        # # 使用1号进行cls判断（pca数值），逆序
        # cls_import_index = np.argsort(np.sum(cls_0>para0[0],1))[:top_k] 
        pred_import_index = comp1_index # np.intersect1d(comp1_index,cls_import_index)
        return pred_import_index
          
    def _process_target_batch(self,future_target,target_class):
        """生成目标数据,包括降维数据以及协方差数据,类别权重数据等"""
        
        pca_target = []
        last_targets = []
        weight_targets = []
        for i in range(future_target.shape[-1]):
            real_target = future_target[...,i]
            last_target = real_target[:,-1] - real_target[:,-2]
            last_targets.append(last_target)
            # Only 1 pca dim
            pca_target_item = pca_apply(real_target,1)   
            # 归一化
            pca_target_item = pca_target_item/pca_target_item.max()
            pca_target.append(pca_target_item)    

        pca_target = torch.stack(pca_target).permute(1,2,0)
        last_targets = torch.stack(last_targets).permute(1,0)
        # 根据类别设置比较权重
        weighted_data = get_weight_with_target(target_class.cpu().numpy())        
        weighted_data = torch.Tensor(weighted_data).to(self.device)
        return last_targets,pca_target,weighted_data
        
    def _process_input_batch(
        self, input_batch
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """重载方法，以适应数据结构变化"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
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
        return x_past_array, future_covariates, static_covariates     

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.sub_models[2].named_parameters():
            global_step = self.current_epoch
            if params is not None:
                self.logger.experiment.add_histogram(name + "_weights",params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram(name + "_grad",params.grad,global_step)
                                                       
    def dump_val_data(self,val_batch,outputs,detail_loss):
        output,pca_target = outputs
        pca_target = pca_target.cpu().numpy()
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,future_past_covariate,target_class,future_target,target_info,price_target) = val_batch
        data = [past_target.cpu().numpy(),target_class.cpu().numpy(),
                future_target.cpu().numpy(),pca_target,price_target.cpu().numpy(),target_info]          
        output_combine = (output,data)
        # pickle.dump(output_combine,self.valid_fout)     
        # 目标数据合并到一起  
        target = np.concatenate([past_target.cpu().numpy(),future_target.cpu().numpy()],axis=1)
        # 保存数据用于后续验证
        output_res = (output,target,target_class.cpu().numpy(),target_info)
        self.output_result.append(output_res)
    
    
    def combine_output_total(self,output_result):
        
        target_class_total = []
        target_info_total = []
        target_total = []        
        output_total = [[[] for _ in range(5)] for _ in range(3)]
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

        target_class_total = np.concatenate(target_class_total)[:,0,0]
        target_total = np.concatenate(target_total)
                    
        return output_total,target_total,target_class_total,target_info_total

    def combine_output_single(self,output_data):
        
        output_total = [[[] for _ in range(5)] for _ in range(3)]
        for i in range(len(output_data)):
            output_item = output_data[i]
            x_bar,z,cls,tar_cls,x_smo = output_item 
            output_total[i][0].append(x_bar)
            output_total[i][1].append(z)
            output_total[i][2].append(cls)
            output_total[i][3].append(tar_cls)
            output_total[i][4].append(x_smo)
        for i in range(len(output_total)):
            output_total[i][0] = torch.concat(output_total[i][0])
            output_total[i][1] = torch.concat(output_total[i][1])
            output_total[i][2] = torch.concat(output_total[i][2])
            # output_total[i][3] = np.concatenate(output_total[i][3])
            output_total[i][4] = torch.concat(output_total[i][4])       
            
        return output_total
            
    def combine_result_data(self,output_result=None):
        """计算涨跌幅分类准确度以及相关数据"""
        
        # 使用全部验证结果进行统一比较
        output_total,target_total,target_class_total,target_info_total = self.combine_output_total(output_result)

        # 按照日期分组计算
        fur_dates = {}
        for index,ti in enumerate(target_info_total):
            future_start_datetime = ti["future_start_datetime"]
            if not future_start_datetime in fur_dates.keys():
                fur_dates[future_start_datetime] = [index]
            else:
                fur_dates[future_start_datetime].append(index)
        # 生成目标索引
        import_index_all,values = self.build_import_index(output_data=output_total,fur_dates=fur_dates,target_info=target_info_total)
        rate_total = {}
        total_imp_cnt = np.where(target_class_total==3)[0].shape[0]
        # 对每天的准确率进行统计，并累加
        for date in import_index_all.keys():
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
                        cnt = cnt_values[0,1]
                    rate_total[date].append(cnt)
                # 预测数量以及总数量
                rate_total[date].append(total_cnt)              
        sr = np.array(list(rate_total.values()))
        return sr,total_imp_cnt,import_index_all
        
    def on_validation_epoch_end(self):
        """重载父类方法，实现自定义评分"""
        
        # SANITY CHECKING模式下，不进行处理
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return    

        sr,total_imp_cnt,import_index_all = self.combine_result_data(self.output_result)      
        if sr.shape[0]==0:
            return
        
        # 汇总计算准确率
        combine_cnt = np.sum(sr,axis=0)
        combine_rate = []
        for i in range(4):
            # 统一计算准确率
            rate = combine_cnt[i] / combine_cnt[4]
            combine_rate.append(rate)
            self.log("score_{} rate".format(i), rate, prog_bar=True) 
        self.log("total cnt", combine_cnt[4], prog_bar=True)  
        # 综合评估计算总的得分情况，需要满足错误分和正确分达到一定阈值
        if combine_rate[-1]>0.2 and combine_rate[0]<0.09:
            # 正确错误分比例来决定评分
            score_total = combine_rate[-1]/combine_rate[0] + combine_rate[2]/combine_rate[1]
        else:
            score_total = 0
        self.log("total_imp_cnt", total_imp_cnt, prog_bar=True)  
        self.log("score_total", score_total, prog_bar=True) 
        
        # 如果是测试模式，则在此进行可视化
        if self.mode=="pred_batch":
            viz_total_size = 0
            output_total,target_total,target_class_total,target_info_total = self.combine_output_total(self.output_result)
            for index,date in enumerate(import_index_all.keys()):
                if viz_total_size>20:
                    break
                import_index = import_index_all[date]            
                target_info = np.array(target_info_total)[import_index]
                target_vr_class = target_class_total[import_index]
                target = target_total[import_index]
                output_data = [output_total[i][0].cpu().numpy()[import_index] for i in range(3)]
                viz_total_size = self.viz_results(output_data, target_vr_class=target_vr_class,
                                        target_info=target_info, date=date,target=target,counter=viz_total_size)

    def viz_results(self,output_data,target_vr_class=None,target_info=None,date=None,target=None,counter=0):
        """Visualization Output and Target"""
        
        
        names = ["pred","label","price","obv_output","obv_tar","cci_output","cci_tar"]        
        names = ["price past","price future","CNTN5_output","CNTN5_tar"]      
        names = ["price past","price future","CNTN5_output","CNTN5_tar","macd_diff","macd_dea"]      
        result = []
            
        fea0_values,fea1_values,fea2_values = output_data
        
        target_neg_index = np.where(target_vr_class<=1)[0]
        target_suc_index = np.where(target_vr_class>=2)[0]
        pad_before = np.array([0 for i in range(self.input_chunk_length)])
        pad_after = np.array([0 for i in range(self.output_chunk_length)])
        
        # CNTN5 part
        tar1_values = target[:,:,1]
        # Price part
        tar2_values = target[:,:,2]
        # def _viz_att_data(target_info):
        #     names = ["price","rsi_5","rsi_20"]
        #     for i in range(len(target_info)):
        #         ts = target_info[i]
        #         price_item = ts["price_array"]
        #         rsi_5 = ts["rsi_5"]
        #         rsi_20 = ts["rsi_20"]
        #         view_data = np.stack([price_item,rsi_5,rsi_20]).transpose(1,0)
        #         target_title = "fur_date:{},instrument:{}".format(ts["future_start_datetime"],ts["instrument"])
        #         win = "win_{}".format(i)
        #         viz_target.viz_matrix_var(view_data,win=win,title=target_title,names=names)          
        #
        # _viz_att_data(np.array(target_info[:5]))
               
        for n in range(2):
            if n==0:
                target_index = target_neg_index
                tar_viz = global_var.get_value("viz_result_fail")
            else:
                target_index = target_suc_index
                tar_viz = global_var.get_value("viz_result_suc")
            size = target_index.shape[0] if target_index.shape[0]<2 else 2
            for i in range(size):
                s_index = target_index[i]
                ts = target_info[s_index]
                instrument = ts["instrument"] 
                # 价格部分
                price_data_past = ts["price_array"][:self.input_chunk_length]
                price_data_past = np.concatenate((price_data_past,pad_after),axis=-1)
                price_data_future = ts["price_array"][25:]
                price_data_future = np.concatenate((pad_before,price_data_future),axis=-1)
                # 预测目标部分
                view_data = np.stack((price_data_past,price_data_future)).transpose(1,0)
                fea1_data = np.concatenate((pad_before,fea1_values[s_index]),axis=-1)
                fea1_combine = np.stack([fea1_data,tar1_values[s_index]]).transpose(1,0)
                fea2_data = np.concatenate((pad_before,fea2_values[s_index]),axis=-1)
                fea2_combine = np.stack([fea2_data,tar2_values[s_index]]).transpose(1,0)
                view_data = np.concatenate([view_data,fea1_combine],axis=1)
                # 辅助数据部分
                att_data = np.stack([ts["macd_diff"],ts["macd_dea"]]).transpose(1,0)
                view_data = np.concatenate([view_data,att_data],axis=1)
                target_title = "tar_class:{},date:{},instrument:{}".format(target_vr_class[s_index],date,instrument)
                counter+=1
                win = "win_{}".format(counter)
                tar_viz.viz_matrix_var(view_data,win=win,title=target_title,names=names)
        return counter
                                        
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        """预测阶段，复用雁阵 """
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates)    
        input_batch = self._process_input_batch(inp)
        (output_pred,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)        
        past_target = batch[0]
        # 生成预测结果
        output_total = self.combine_output_single(output_pred) 
        # 生成目标索引
        import_index,_ = self.build_import_index(output_data=output_total)  
        # 根据索引取得目标
        pred_result = np.array(target_info)[import_index]   
        
        return pred_result
        