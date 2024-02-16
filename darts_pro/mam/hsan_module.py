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

from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import pairwise_distances,normalization
from tft.class_define import CLASS_SIMPLE_VALUES
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_model import _CusModule,TFTExtModel
from darts_pro.data_extension.batch_dataset import BatchDataset

class HsanModule(_CusModule):
    """"""
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
                                    use_weighted_loss_func=use_weighted_loss_func,
                                    device=device,**kwargs)  
        
        
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
        
        # 根据目标过去值(Past Target),生成邻接矩阵
        with torch.no_grad():
            price_array = np.array([t["price_array"] for t in target_info])
            price_array = price_array[:,:self.input_chunk_length]
            price_array = normalization(price_array)
            adj_matrix = pairwise_distances(torch.Tensor(price_array).to(self.device),distance_func=self.criterion.ccc_distance)
            # 如果维度不够，则补0
            if adj_matrix.shape[0]<batch_size:
                pad_zize = batch_size - adj_matrix.shape[0]
                adj_matrix = torch.nn.functional.pad(adj_matrix, (0, pad_zize, 0, pad_zize))
            
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                # 根据配置，不同的模型使用不同的过去协变量
                past_convs_item = x_in[0][i]
                x_in_item = (past_convs_item,x_in[1],x_in[2])
                # 使用embedding组合以及邻接矩阵作为输入
                out = m(x_in_item,adj_matrix)
                out_class = torch.ones([batch_size,self.output_chunk_length,1]).to(self.device)
            else:
                # 模拟数据
                out = torch.ones([batch_size,self.output_chunk_length,self.output_dim[0],1]).to(self.device)
                out_class = torch.ones([batch_size,1]).to(self.device)
            out_total.append(out)    
            out_class_total.append(out_class)
        
        # if optimizer_idx==3:
        #     print("ggg")
        # out_for_class = torch.cat(out_total,dim=2)[:,:,:,0] 
        # focus_data = self.build_focus_data(out_for_class,past_target,target_info=target_info,scalers=scaler)
        # 根据预测数据进行二次分析
        vr_class = torch.ones([batch_size,len(CLASS_SIMPLE_VALUES.keys())]).to(self.device) 
        # if optimizer_idx>=len(self.sub_models):
        #     print("ggg")
        # vr_class = self.classify_vr_layer(focus_data)
        tar_class = torch.ones(vr_class.shape).to(self.device) # self.classify_tar_layer(x_conv_transform)
        return out_total,vr_class,out_class_total
      
    def training_step_real(self, train_batch, batch_idx) -> torch.Tensor:
        """补充训练步骤"""
        
        total_loss,detail_loss,output = super().training_step_real(train_batch,batch_idx)
        
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        input_batch = self._process_input_batch(val_batch[:5])
        # 收集目标数据用于分类
        scaler_tuple,target_class,future_target,target_info = val_batch[5:]  
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
        (corr_loss_combine,triplet_loss_combine,extend_values) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        for i in range(len(corr_loss_combine)):
            acc, nmi, ari, f1, P, center = self.compute_acc_and_weight(output[i], target_vr_class, S=extend_values,index=i)
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_triplet_loss_{}".format(i), triplet_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_acc_{}".format(i), acc, batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_acc_{}".format(i), corr_acc_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("last_vr_loss", last_vr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
            
        
        # import_price_result = self.compute_real_class_acc(output_inverse=output_inverse,target_vr_class=target_vr_class,
        #             vr_class=vr_class_sf,output_data=output_combine,target_info=target_info,target_inverse=target_inverse)   
        # total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        # if self.total_imp_cnt==0:
        #     self.total_imp_cnt = total_imp_cnt
        # else:
        #     self.total_imp_cnt += total_imp_cnt
        #
        #
        # # # 累加结果集，后续统计   
        # if self.import_price_result is None:
        #     self.import_price_result = import_price_result    
        # else:
        #     if import_price_result is not None:
        #         import_price_result_array = import_price_result.values
        #         # 修改编号，避免重复
        #         import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*1000
        #         import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
        #         self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)        
        
        # for i in range(3):
        #     self.log("triplet acc_{}".format(i), corr_acc_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("output_imp_class_acc_cnt", output_imp_class_acc_cnt, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("output_imp_class_acc", output_imp_class_acc, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        return loss,detail_loss,output
 
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""
        
        (output_value,vr_class,tar_class) = output
        output_combine = (output_value,vr_class,tar_class)
        return self.criterion(output_combine, target,optimizers_idx=optimizers_idx)
    
    def compute_acc_and_weight(self,output,target_class,S=None,index=0):       
        """计算聚类准确率，以及更新权重"""
        
        # 使用批次大小作为节点数
        node_num = target_class.shape[0]
        
        Z1, Z2, E1, E2 = output
        cluster_num = len(CLASS_SIMPLE_VALUES.keys())
        
        # fusion and testing
        Z = (Z1 + Z2) / 2
        # 计算准确率,使用涨幅类别作为groud truth
        acc, nmi, ari, f1, P, center = phi(Z, target_class, cluster_num,device=self.device)

        # 选择高置信度的样本
        H, H_mat = high_confidence(Z, center)

        # 计算伪标签权重
        M, M_mat = pseudo_matrix(P, S, node_num,device=self.device)

        # 更新权重
        self.sub_models[index].pos_weight[H] = M[H].data
        self.sub_models[index].pos_neg_weight[H_mat] = M_mat[H_mat].data    
        
        return acc, nmi, ari, f1, P, center 
        
              
        