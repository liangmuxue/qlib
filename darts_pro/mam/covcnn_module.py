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
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

from darts_pro.data_extension.custom_module import viz_target,viz_result_suc,viz_result_fail,viz_result_nor
from darts_pro.act_model.cov_cnn import CovCnn,PcaCnn
from cus_utils.metrics import pca_apply
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX
from losses.clustering_loss import CovCnnLoss,cluster_acc
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
from cus_utils.visualization import ShowClsResult
import cus_utils.global_var as global_var

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_module import _TFTModuleBatch

class CovCnnModule(_TFTModuleBatch):
    """自定义基于卷积模式的时间序列模块"""
    
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
        step_mode="pretrain",
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
        self.switch_flag = 0
        self.step_mode=step_mode
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
                historic_future_covariates,
                future_covariates,
                static_covariates,
                (scaler,future_past_covariate),
                target_class,
                future_target,
                target_info,
                price_target
            ) = self.train_sample      
                  
            # 固定单目标值
            past_target_shape = 1
            past_conv_index = self.past_split[seq]
            # 只检查属于自己模型的协变量
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]            
            past_covariates_shape = past_covariates_item.shape[-1]
            historic_future_covariates_shape = len(variables_meta["input"]["historic_future_covariate"])
            # 记录动态数据长度，后续需要切片
            self.dynamic_conv_shape = past_target_shape + past_covariates_shape
            input_dim = (
                past_target_shape
                + past_covariates_shape
                # + historic_future_covariates_shape
            )
    
            output_dim = 1
    
            future_cov_dim = (
                future_covariates.shape[-1] if future_covariates is not None else 0
            )
            
            static_cov_dim = (
                static_covariates.shape[-2] * static_covariates.shape[-1]
                if static_covariates is not None
                else 0
            )
    
            nr_params = 1
            num_nodes = past_target.shape[0]
            self.pca_dim = 4
            model = PcaCnn(
                pca_dim=self.pca_dim,
                input_dim=input_dim,
                output_dim=2,
                n_cluster=len(CLASS_SIMPLE_VALUES.keys()),
                use_layer_norm=True,
                device=device,
                **kwargs,
            )           
            
            return model
        
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        future_target,
        pca_target=None,
        target_info=None,
        optimizer_idx=-1
    ) -> torch.Tensor:
        
        """整合多种模型，主要使用深度聚类方式"""
        
        out_total = []
        out_class_total = []
        (batch_size,_,_) = x_in[1].shape
        
        # 设置训练模式，预训练阶段，以及全模式的初始阶段，都使用预训练模式
        if self.switch_flag==0 or self.step_mode=="pretrain":
            step_mode = "pretrain"
        else:
            step_mode = "complete"   
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据配置，不同的模型使用不同的过去协变量
            past_convs_item = x_in[0][i]            
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                # 输入值只使用动态变量
                past_convs_item = past_convs_item[...,:m.input_dim]
                # 输入内容转化为降维数据
                pca_matrix = []
                for j in range(past_convs_item.shape[-1]):
                    pca_item = pca_apply(past_convs_item[...,j],self.pca_dim)
                    pca_matrix.append(pca_item)
                pca_matrix = torch.stack(pca_matrix).permute(1,2,0)
                out = m(pca_matrix,pca_target=pca_target[...,i])
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
        return CovCnnLoss(device=device,ref_model=model) 

    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        if self.switch_flag==0 or self.step_mode=="pretrain":
            step_mode = "pretrain"
        else:
            step_mode = "complete"   
            
        (future_target,target_class,pca_target) = target   
        return self.criterion(output,(future_target,target_class,pca_target),mode=step_mode,optimizers_idx=optimizers_idx)

    def on_validation_start(self): 
        super().on_validation_start()

    def on_train_epoch_start(self):  
        super().on_train_epoch_start()
                
    def output_postprocess(self,output,targets,index):
        """对于不同指标输出的补充"""
        
        # 只需要实际数据，忽略模拟数据
        output_act = output[index]
        self.training_step_outputs[index].append(output_act[1])
        self.training_step_targets[index].append(targets) 

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """重载原方法，直接使用已经加工好的数据"""

        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler,target_class,target,target_info,price_target) = train_batch    

        loss,detail_loss,output = self.training_step_real(train_batch, batch_idx) 
        if self.train_output_flag:
            output = [output_item.detach().cpu().numpy() for output_item in output]
            data = [past_target.detach().cpu().numpy(),past_covariates.detach().cpu().numpy(), historic_future_covariates.detach().cpu().numpy(),
                             future_covariates.detach().cpu().numpy(),static_covariates.detach().cpu().numpy(),scaler,target_class.cpu().detach().numpy(),
                             target.cpu().detach().numpy(),target_info,price_target.cpu().detach().numpy()]                
            output_combine = (output,data)
            pickle.dump(output_combine,self.train_fout)  
        # (mse_loss,value_diff_loss,corr_loss,ce_loss,mean_threhold) = detail_loss
        return loss
    
    def training_step_real(self, train_batch, batch_idx) -> torch.Tensor:
        """包括第一及第二部分数值数据,以及分类数据"""

        # 收集目标数据用于分类
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = train_batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates)     
        scaler = [s[0] for s in scaler_tuple] 
        past_target = train_batch[0]
        input_batch = self._process_input_batch(inp)
        cov_targets,pca_target = self._process_target_batch(future_target)
        target_class = target_class[:,:,0]     
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            y_transform = None 
            (output,vr_class,tar_class) = self(input_batch,future_target,pca_target=pca_target,
                                               target_info=target_info,optimizer_idx=i)
            self.output_postprocess(output,target_class,i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (cov_targets,target_class,pca_target),optimizers_idx=i)
            (corr_loss_combine,ce_loss,cls_loss,_) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
            self.loss_data.append(detail_loss)
            total_loss += loss
            # 手动更新参数
            opt = self.trainer.optimizers[i]
            opt.zero_grad()
            self.manual_backward(loss)
            # 如果已冻结则不执行更新
            if self.freeze_mode[i]==1:
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
        
        if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag:
            self.dump_val_data(val_batch,output,detail_loss)
        return loss,detail_loss
           
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = val_batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates) 
        input_batch = self._process_input_batch(inp)
        cov_targets,pca_target = self._process_target_batch(future_target)
        scaler = [s[0] for s in scaler_tuple]
        (output,vr_class,vr_class_list) = self(input_batch,future_target,pca_target=pca_target,
                                               target_info=target_info,optimizer_idx=-1)
        
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (cov_targets,target_class,pca_target),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,cls_loss,_) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(corr_loss_combine)):
            # self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_cls_loss_{}".format(i), cls_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # if comb_detail_loss is not None:
            #     self.log("val_loss_xz_{}".format(i), comb_detail_loss[1], batch_size=val_batch[0].shape[0], prog_bar=False)
            #     self.log("val_loss_xc_{}".format(i), comb_detail_loss[2], batch_size=val_batch[0].shape[0], prog_bar=False)
            
            # past_convs_item = input_batch[0][i] 
            # pred,pred_combine = self.sub_models[i].predict(past_convs_item)
            # preds_combine.append(pred_combine)
            # acc = self.cluster_acc(pred,target_vr_class)
            # self.log("val_acc_{}".format(i), acc[0], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,pca_target)
        # if self.step_mode=="pretrain" or self.switch_flag==0:
        #     real_pca_target = pca_target
        # else:
        #     real_pca_target = [output_item[1] for output_item in output]
        #     real_pca_target = torch.stack(real_pca_target).permute(1,2,0)
        # 目标类别分类评估
        # for i in range(len(corr_loss_combine)):
        #     pred,w = self.sub_models[i].predict_pca_cls(real_pca_target[...,i])
        #     acc_cnt = np.sum(pred==target_class[:,0].cpu().numpy())
        #     acc = acc_cnt/pred.shape[0]
        #     # self.log("cls_acc_{}".format(i), acc, batch_size=val_batch[0].shape[0], prog_bar=True)
        #     # 可视化权重分割情况
        #     path = "custom/data/results/pred/{}".format(i)
        #     if not os.path.exists(path):
        #         os.makedirs(path)                 
        #     save_file = "{}/pred_{}-{}.png".format(path,self.current_epoch,batch_idx)
        #     ShowClsResult(w[0].transpose(),np.expand_dims(w[1],axis=0),real_pca_target[...,i].cpu().numpy(),target_vr_class,save_file=save_file)
        
        # 准确率的计算
        # import_price_result,values = self.compute_real_class_acc(output_data=output,pca_target=pca_target,target_class=target_class,target_info=target_info)          
        # total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        # if self.total_imp_cnt==0:
        #     self.total_imp_cnt = total_imp_cnt
        # else:
        #     self.total_imp_cnt += total_imp_cnt
        # # 累加结果集，后续统计   
        # if self.import_price_result is None:
        #     self.import_price_result = import_price_result    
        # else:
        #     if import_price_result is not None:
        #         import_price_result_array = import_price_result.values
        #         # 修改编号，避免重复
        #         import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*3000
        #         import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
        #         self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)        
        #

        # 可视化
        index = "{}-{}".format(self.current_epoch,batch_idx)
        # self.viz_results(output, pca_target,target_class=target_class[:,0].cpu().numpy(),index=index)
        
        return loss,detail_loss,output_combine

    def compute_real_class_acc(self,output_data=None,pca_target=None,target_class=None,target_info=None):
        """计算涨跌幅分类准确度"""
        
        # 使用分类判断
        import_index,values = self.build_import_index(output_data=output_data,pca_target=pca_target,target_class=target_class[:,0])
        target_class = target_class.view(-1)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class, \
            import_price_result = self.collect_result(import_index, target_class.cpu().numpy(), target_info)
        
        return import_price_result,values
           
    def build_import_index(self,output_data=None,pca_target=None,target_class=None):  
        """生成涨幅达标的预测数据下标"""
        
        cls_values = []
        fea_values = []
        pca_values = []
        for i in range(len(output_data)):
            output_item = output_data[i] 
            cls,features,fea_pca  =  output_item 
            cls_values.append(cls)
            fea_values.append(features)
            pca_values.append(fea_pca)
        
        cls_values = torch.stack(cls_values).cpu().numpy().transpose(1,2,0)
        fea_values = torch.stack(fea_values).cpu().numpy().transpose(1,2,0)
        pca_values = torch.stack(pca_values).cpu().numpy().transpose(1,2,0)
        
        cls_2 = cls_values[...,2]
        cls_2 = np.argmax(cls_2,1)
        pred_import_bool = (cls_2==target_class)
        pred_import_index = np.where(pred_import_bool)[0]
        
        return pred_import_index,(cls_values,fea_values,pca_values)       

    def viz_results(self,output, pca_target,target_class=None,index=None):
        """可视化，显示聚合过程"""
        
        sampler_number = 10
        # 采样一些正负样本，查看聚合情况
        import_index = np.where(target_class==3)[0]
        import_index = import_index[:sampler_number]
        neg_index = np.where(target_class==0)[0]
        neg_index = neg_index[:sampler_number]
        combine_index = np.concatenate((import_index,neg_index))
        
        
        labels = target_class[combine_index]
        
        for i in range(3):
            output_item = output[i]
            (x_bar, z_pca,lattend,_,_) = output_item
            z_pca = z_pca.cpu().numpy()
            output_item = z_pca[combine_index]
            pca_target_single = pca_target[...,i][combine_index]
            pca_target_single = pca_target_single.cpu().numpy()
            # PCA模式下，混合输出正负样本预测结果，并与原值进行比较
            self.matrix_results_viz(output=output_item,target=pca_target_single,
                        sampler_number=sampler_number,target_index=i,name="{}_{}".format(index,i))
        
    def matrix_results_viz(self,output=None,target=None,target_index=0,sampler_number=10,name="target"):
        plt.figure(name,figsize=(12,9))
        
        # 实际目标正样本
        plt.scatter(target[:sampler_number,0],target[:sampler_number,1], marker='o',color="r", s=50)
        # 预测目标正样本
        plt.scatter(output[:sampler_number,0],output[:sampler_number,1], marker='o',color="b", s=50)
        # 实际目标负样本
        plt.scatter(target[sampler_number:,0],target[sampler_number:,1], marker='x',color="k", s=50)
        # 预测目标负样本
        plt.scatter(output[sampler_number:,0],output[sampler_number:,1], marker='x',color="y", s=50)
        for i in range(target.shape[0]):
            x_target = target[i][0]
            y_target = target[i][1]
            imp_target = "t_{}".format(i)
            x_out = output[i][0]
            y_out = output[i][1]
            imp_out = "o_{}".format(i)         
            plt.annotate(imp_target, xy = (x_target, y_target), xytext=(x_target+0.01, y_target+0.01))
            plt.annotate(imp_out, xy = (x_out, y_out), xytext=(x_out+0.01, y_out+0.01))
        # plt.show()      
        path = "./custom/data/results/sdcn/{}".format(target_index)
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig('{}/combine_result_{}.png'.format(path,name))  
            
    
    def _process_target_batch(self,future_target):
        """生成目标数据,包括降维数据以及协方差数据"""
        
        pca_target = []
        cov_targets = []
        for i in range(future_target.shape[-1]):
            real_target = future_target[...,i]
            cov_target = torch.corrcoef(real_target)
            cov_targets.append(cov_target)
            pca_target_item = pca_apply(real_target,2)   
            # 归一化
            pca_target_item = pca_target_item/pca_target_item.max()
            pca_target.append(pca_target_item)    
        pca_target = torch.stack(pca_target).permute(1,2,0)
        cov_targets = torch.stack(cov_targets).permute(1,2,0)
        return cov_targets,pca_target
        
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

        # Norm his conv
        historic_future_covariates = normalization_axis(historic_future_covariates,axis=2)
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
        return x_past_array, future_covariates, static_covariates     
                                       
    def dump_val_data(self,val_batch,outputs,detail_loss):
        if self.switch_flag==0 or self.step_mode=="pretrain":
            return
        output,pca_target = outputs
        pca_target = pca_target.cpu().numpy()
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = val_batch
        # for i,output_item in enumerate(output):
        #     pred_combine = preds_combine[i]  
        #     (yita,z,latent,cell_output) = pred_combine
        data = [past_target.cpu().numpy(),target_class.cpu().numpy(),
                future_target.cpu().numpy(),pca_target,price_target.cpu().numpy(),target_info]          
        output_combine = (output,data)
        pickle.dump(output_combine,self.valid_fout)       
           
        