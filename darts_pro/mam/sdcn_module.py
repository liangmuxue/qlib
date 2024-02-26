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
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from darts_pro.act_model.sdcn_ts import SdcnTs
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from losses.clustering_loss import ClusteringLoss
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity
import cus_utils.global_var as global_var

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_module import _TFTModuleBatch

class SdcnModule(_TFTModuleBatch):
    """自定义基于图模式和聚类的时间序列模块"""
    
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
        filter_conv_index=0,
        batch_file_path=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,batch_file_path=batch_file_path,
                                    device=device,**kwargs)  
        self.output_data_len = len(past_split)
        self.step_mode = "pretrain"
        self.switch_epoch_num = 10
        
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
        **kwargs):
        
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                (scaler,future_past_covariate),
                target_class,
                future_target,
                target_info
            ) = self.train_sample      
                  
            past_target_shape = len(variables_meta["input"]["past_target"])
            past_covariates_shape = len(variables_meta["input"]["past_covariate"])
            historic_future_covariates_shape = len(variables_meta["input"]["historic_future_covariate"])
            input_dim = (
                past_target_shape
                + past_covariates_shape
                + historic_future_covariates_shape
            )
    
            # 不使用原设定的输出维度，而是以目标值数量作为实际维度
            dataset = global_var.get_value("dataset")
            output_dim = 1
    
            future_cov_dim = (
                future_covariates.shape[1] if future_covariates is not None else 0
            )
            static_cov_dim = (
                static_covariates.shape[0] * static_covariates.shape[1]
                if static_covariates is not None
                else 0
            )
    
            nr_params = 1

            model = SdcnTs(
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
                use_layer_norm=False,
                dropout=dropout,
                # Sdcn Part
                n_cluster=len(CLASS_SIMPLE_VALUES.keys()),
                activation="prelu",
                **kwargs,
            )           
            
            return model
        
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        future_target,
        scaler,
        past_target=None,
        target_info=None,
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """整合多种模型，主要使用深度聚类方式"""
        
        out_total = []
        out_class_total = []
        batch_size = x_in[1].shape[0]
        
            
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                if self.step_mode=="complete":
                    # 根据过去目标值(Past Target),生成邻接矩阵
                    with torch.no_grad():
                        # price_array = np.array([t["price_array"] for t in target_info])
                        # price_array_ori = price_array[:,:self.input_chunk_length]
                        # adj_target = normalization(price_array_ori,axis=1)
                        adj_target = future_target
                        # 生成symmetric邻接矩阵以及拉普拉斯矩阵
                        adj_matrix = build_symmetric_adj(adj_target,device=self.device,distance_func=self.criterion.ccc_distance_torch)
                        # 如果维度不够，则补0
                        if adj_matrix.shape[0]<batch_size:
                            pad_zize = batch_size - adj_matrix.shape[0]
                            adj_matrix = torch.nn.functional.pad(adj_matrix, (0, pad_zize, 0, pad_zize))
                        adj_matrix = adj_matrix.double().to(self.device)      
                else:
                    adj_matrix = None          
                # 根据配置，不同的模型使用不同的过去协变量
                past_convs_item = x_in[0][i]
                x_in_item = (past_convs_item,x_in[1],x_in[2])
                # 使用embedding组合以及邻接矩阵作为输入
                out = m(x_in_item,adj_matrix,mode=self.step_mode)
                # 完整模式下，需要进行2次模型处理
                if self.step_mode=="complete":
                    out_again = m(x_in_item,adj_matrix,mode=self.step_mode)
                    out = (out,out_again)
                else:
                    out = (out,None)
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
        return ClusteringLoss(device=device,ref_model=model) 

    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""
        
        # 由于使用无监督算法，故不需要target数据
        return self.criterion(output,target,optimizers_idx=optimizers_idx)

    def on_train_start(self): 
        super().on_train_start()
        # 开始阶段，只训练特征部分
        if self.current_epoch<self.switch_epoch_num:
            self.step_mode = "pretrain"
        else:
            self.step_mode = "complete"
        # 初始化训练结果数据
        self.training_step_outputs = [[] for _ in range(self.output_data_len)]

    def on_train_epoch_start(self):  
        super().on_train_epoch_start()
        # 每个轮次前，判断并进行训练内容切换
        if self.current_epoch>self.switch_epoch_num:
            # 切换时，进行聚类以取得初始化的簇心
            if self.step_mode=="pretrain":
                for model_seq in range(len(self.sub_models)):
                    # 取得最近一次的特征中间值，作为聚类输入数据MACD_SDCN_2000_202010
                    z = [output_item[-1] for output_item in self.training_step_outputs[model_seq]]
                    z = torch.concat(z,dim=0).detach().cpu().numpy()
                    n_clusters = len(CLASS_SIMPLE_VALUES.keys())
                    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                    y_pred = kmeans.fit_predict(z)
                    model = self.sub_models[model_seq]
                    # 直接把初始化簇心值赋予模型内的参数
                    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)                
            # 超过特定轮次，切换到全模式
            self.step_mode = "complete"  
                                 
    def output_postprocess(self,output,index):
        """对于单步输出的补充"""
        
        # 保存训练结果数据，用于后续分析,只在特定轮次进行
        if self.current_epoch==self.switch_epoch_num:
            # 只需要实际数据，忽略模拟数据
            output_act = output[index]
            self.training_step_outputs[index].append(output_act[0])

    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler,target_class,target,target_info,rank_targets) = val_batch    
         
        # 使用排序目标替换原数据
        val_batch_convert = (past_target,past_covariates, historic_future_covariates,future_covariates, 
                               static_covariates,scaler,target_class,target,target_info,rank_targets)
                
        loss,detail_loss,output = self.validation_step_real(val_batch_convert, batch_idx)  
        
        # if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag:
        #     output = [output_item.cpu().numpy() for output_item in output]
        #     (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info,rank_scalers) = val_batch 
        #     data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
        #                      future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]            
        #     output_combine = (output,data)
        #     pickle.dump(output_combine,self.valid_fout)         
        return loss,detail_loss 
            
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        input_batch = self._process_input_batch(val_batch[:5])
        # 收集目标数据用于分类
        scaler_tuple,target_class,future_target,target_info = val_batch[5:-1]  
        scaler = [s[0] for s in scaler_tuple]
        (output,vr_class,vr_class_list) = self(input_batch,future_target,scaler,past_target=val_batch[0],target_info=target_info,optimizer_idx=-1)
        
        raise_range_batch = np.expand_dims(np.array([ts["raise_range"] for ts in target_info]),axis=-1)
        y_transform = raise_range_batch  
        y_transform = torch.Tensor(y_transform).to(self.device)  
              
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), (future_target,target_class,target_info,y_transform),optimizers_idx=-1)
        (corr_loss_combine,kl_loss,ce_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_kl_loss_{}".format(i), kl_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        return loss,detail_loss,output

        
        
              
        