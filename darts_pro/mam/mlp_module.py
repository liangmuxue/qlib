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

from darts_pro.data_extension.custom_module import viz_target,viz_result_suc,viz_result_fail,viz_result_nor
from darts_pro.act_model.mlp_ts import MlpTs
from cus_utils.metrics import pca_apply
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,get_weight_with_target
from losses.clustering_loss import MlpLoss
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
from cus_utils.visualization import ShowClsResult
import cus_utils.global_var as global_var
from losses.fds_loss import weighted_focal_l1_loss
from losses.quanlity_loss import QuanlityLoss

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
                + historic_future_covariates_shape
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
        
        self.custom_histogram_adder()
        
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
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class[:,0])
        target_class = target_class[:,:,0]     
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        # 根据标志决定是否梯度回传
        if self.switch_flag==1 and False:
            with torch.no_grad():
                for i in range(len(self.past_split)):
                    (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
                    loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,last_targets,pca_target),optimizers_idx=i)
                    (corr_loss_combine,ce_loss,fds_loss,_) = detail_loss 
                    self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
                    self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
                    # self.log("train_fds_loss_{}".format(i), fds_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
                    self.loss_data.append(detail_loss)
                    total_loss += loss
                    self.output_postprocess(output,target_class[:,0],i)
        else:
            for i in range(len(self.past_split)):
                (output,vr_class,tar_class) = self(input_batch,optimizer_idx=i)
                loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,last_targets,pca_target),optimizers_idx=i)
                (corr_loss_combine,ce_loss,fds_loss,cls_loss) = detail_loss 
                # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
                self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
                self.log("train_cls_loss_{}".format(i), cls_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)
                self.loss_data.append(detail_loss)
                total_loss += loss     
                # 手动更新参数
                opt = self.trainer.optimizers[i]
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
                # if i==1:
                #     for name,params in self.sub_models[1].named_parameters():
                #         print("{} end grad:{}".format(name,params.grad))
                self.lr_schedulers()[i].step()                             
            # Viz pca
            # self.pca_viz_item(i,pca_target[...,i],output[i][1].detach(),target_class=target_class[:,0],
            #                   batch_idx=batch_idx,root_path="custom/data/results/train")
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
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class[:,0])
        scaler = [s[0] for s in scaler_tuple]
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
        for i in range(pca_target.shape[-1]):
            pred_cls = output[i][2].cpu().numpy()
            # pred_cls = np.argmax(pred_cls)
            # # pred,w = self.sub_models[i].predict_pca_cls(pca_output[...,i])
            # acc_cnt = np.sum(pred_cls==target_class[:,0].cpu().numpy())
            # acc = acc_cnt/pred_cls.shape[0]
            # self.log("cls_acc_{}".format(i), acc, batch_size=val_batch[0].shape[0], prog_bar=True)       

        # self.pca_viz(pca_target, target_class[:,0].cpu().numpy(), output,batch_idx=batch_idx,root_path="custom/data/results/pred")
        # 准确率的计算
        import_price_result,values = self.compute_real_class_acc(output_data=output,past_target=past_target.cpu().numpy(),target_class=target_class,target_info=target_info)          
        total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        if self.total_imp_cnt==0:
            self.total_imp_cnt = total_imp_cnt
        else:
            self.total_imp_cnt += total_imp_cnt
        # 累加结果集，后续统计   
        if self.import_price_result is None:
            self.import_price_result = import_price_result    
        else:
            if import_price_result is not None:
                import_price_result_array = import_price_result.values
                # 修改编号，避免重复
                import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*3000
                import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
                self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)        
        
        # PCA可视化
        index = "{}-{}".format(self.current_epoch,batch_idx)
        # self.viz_results(output, pca_target,target_class=target_class[:,0].cpu().numpy(),index=index)

        whole_target = np.concatenate((past_target.cpu().numpy(),future_target.cpu().numpy()),axis=1)
        target_inverse = self.get_inverse_data(whole_target,target_info=target_info,scaler=scaler)
        output_inverse = [output_item[0] for output_item in output]
        output_inverse = torch.stack(output_inverse,dim=2).cpu().numpy()           
        # 预测数值可视化
        names = ["macd","rsi","qtlu","macd_output","rsi_output","qtlu_output"] 
        # self.val_metric_show(output_inverse,whole_target,target_vr_class,target_info=target_info,
        #                      import_price_result=import_price_result,batch_idx=batch_idx,names=names)
        
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
                     
    def compute_real_class_acc(self,output_data=None,past_target=None,target_class=None,target_info=None):
        """计算涨跌幅分类准确度"""
        
        # 使用分类判断
        import_index,values = self.build_import_index(output_data=output_data,past_target=past_target,target_class=target_class[:,0].cpu().numpy())
        target_class = target_class.view(-1)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class, \
            import_price_result = self.collect_result(import_index, target_class.cpu().numpy(), target_info)
        
        return import_price_result,values
           
    def build_import_index(self,output_data=None,past_target=None,target_class=None):  
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

        cls_0 = cls_values[...,0]
        cls_1 = cls_values[...,1]
        cls_2 = cls_values[...,2]
        fea_0 = fea_values[...,0]
        fea_0_range = (fea_0[:,-1] - fea_0[:,0])/fea_0[:,0]        
        fea_1 = fea_values[...,1]
        fea_1_range = (fea_1[:,-1] - fea_1[:,0])/fea_1[:,0]
        fea_2 = fea_values[...,2]
        fea_2_range = (fea_2[:,-1] - fea_2[:,0])/fea_2[:,0]/10
        sv_0 = smooth_values[...,0].squeeze(-1)
        sv_1 = smooth_values[...,1].squeeze(-1)
        sv_2 = smooth_values[...,2].squeeze(-1)
        # pred_import_index = self.strategy_threhold((sv_0,sv_1,sv_2),(fea_0_range,fea_1_range),(cls_0,cls_1,cls_2),batch_size=cls_values.shape[0])
        pred_import_index = self.strategy_top((sv_0,sv_1,sv_2),(fea_0_range,fea_1_range),(cls_0,cls_1,cls_2),batch_size=cls_values.shape[0])
        
        return pred_import_index,(cls_values,fea_values,pca_values)       
    
    def strategy_threhold(self,sv,fea,cls):
        (sv_0,sv_1,sv_2) = sv
        (fea_0_range,fea_1_range) = fea
        (cls_0,cls_1,cls_2) = cls
        # 使用回归模式，则找出接近或大于目标值的数据
        sv_import_bool = (sv_2<0) & (fea_0_range>0) & (fea_1_range<-1)
        # ce_thre_para = [[0.1,6],[-0.1,7],[-0.1,6]]
        # ce_para2 = ce_thre_para[2]
        # sv_import_bool = (np.sum(sv_2<ce_para2[0],1)>ce_para2[0])
        # sv_import_bool = (sv_2<0) & (sv_1>0) & (fea_1_range<-1)
        # 分位数回归模式下的阈值选择
        cls_thre_para = [[0.1,8],[-0,8],[-0,7]]
        # 包含2个参数：分数阈值以及个数阈值
        para0 = cls_thre_para[0]
        para1 = cls_thre_para[1]
        para2 = cls_thre_para[2]
        cls_import_bool = (np.sum(cls_1<para1[0],1)>para1[1]) # & (np.sum(cls_2<para2[0],1)>para2[0]) # & (np.sum(cls_0>para0[0],1)>para0[0]) 
        pred_import_index = np.where(cls_import_bool & sv_import_bool)[0]
        
        return pred_import_index

    def strategy_top(self,sv,fea,cls,batch_size=0):
        """排名方式筛选候选者"""
        
        (sv_0,sv_1,sv_2) = sv
        (fea_0_range,fea_1_range) = fea
        (cls_0,cls_1,cls_2) = cls
        
        top_k = batch_size//4
        # 使用2号进行sv判断（最后一段涨跌幅度），逆序
        sv_import_index = np.argsort(sv_2)[:top_k]
        # 使用0号进行corr判断（整体涨跌幅度），正序
        fea0_import_index = np.argsort(-fea_0_range)[:top_k]
        # 使用1号进行corr判断（整体涨跌幅度），逆序
        fea1_import_index = np.argsort(fea_1_range)[:top_k]        
        comp1_index = np.intersect1d(sv_import_index,fea0_import_index)
        comp1_index = np.intersect1d(comp1_index,fea1_import_index)

        cls_thre_para = [[0.1,8],[-0,8],[-0,7]]
        # 包含2个参数：分数阈值以及个数阈值
        para0 = cls_thre_para[0]
        para1 = cls_thre_para[1]
        para2 = cls_thre_para[2]
        # 使用1号进行cls判断（pca数值），逆序
        cls_import_index = np.argsort(np.sum(cls_1<para1[0],1))[:top_k] 
        pred_import_index = np.intersect1d(comp1_index,cls_import_index)
        return pred_import_index
          
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
            (x_bar, z_pca,cls,_,_) = output_item
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
        path = "./custom/data/results/mlp/{}".format(target_index)
        if not os.path.exists(path):
            os.makedirs(path)           
        plt.savefig('{}/combine_result_{}.png'.format(path,name))  
            

    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ):
        """重载原方法，服务于自定义模式"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = batch
        inp = (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates)    
        input_batch = self._process_input_batch(inp)
        last_targets,pca_target,weighted_data = self._process_target_batch(future_target,target_class[:,0])
        (output,vr_class,vr_class_list) = self(input_batch,optimizer_idx=-1)        
        past_target = batch[0]
        past_covariate = batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        preds_combine = []
        output_combine = (output,pca_target)
        for i in range(pca_target.shape[-1]):
            pred_cls = output[i][2].cpu().numpy()
            # pred_cls = np.argmax(pred_cls)
            # # pred,w = self.sub_models[i].predict_pca_cls(pca_output[...,i])
            # acc_cnt = np.sum(pred_cls==target_class[:,0].cpu().numpy())
            # acc = acc_cnt/pred_cls.shape[0]
            # self.log("cls_acc_{}".format(i), acc, batch_size=val_batch[0].shape[0], prog_bar=True)       

        # self.pca_viz(pca_target, target_class[:,0].cpu().numpy(), output,batch_idx=batch_idx,root_path="custom/data/results/pred")
        # 准确率的计算
        import_price_result,values = self.compute_real_class_acc(output_data=output,past_target=past_target.cpu().numpy(),target_class=target_class,target_info=target_info)          
        total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        if self.total_imp_cnt==0:
            self.total_imp_cnt = total_imp_cnt
        else:
            self.total_imp_cnt += total_imp_cnt
        # 累加结果集，后续统计   
        if self.import_price_result is None:
            self.import_price_result = import_price_result    
        else:
            if import_price_result is not None:
                import_price_result_array = import_price_result.values
                # 修改编号，避免重复
                import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*3000
                import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
                self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)          
                  
        return import_price_result
      
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

        # Norm his conv
        try:
            historic_future_covariates = normalization_axis(historic_future_covariates,axis=2)
        except Exception as e:
            print("ggg")
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
        # 静态协变量归一化
        static_covariates_norm = normalization_axis(static_covariates,axis=0)
        return x_past_array, future_covariates, static_covariates_norm     

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
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = val_batch
        # for i,output_item in enumerate(output):
        #     pred_combine = preds_combine[i]  
        #     (yita,z,latent,cell_output) = pred_combine
        data = [past_target.cpu().numpy(),target_class.cpu().numpy(),
                future_target.cpu().numpy(),pca_target,price_target.cpu().numpy(),target_info]          
        output_combine = (output,data)
        pickle.dump(output_combine,self.valid_fout)       


    def on_validation_epoch_end(self):
        """重载父类方法，实现自定义评分"""
        
        # SANITY CHECKING模式下，不进行处理
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return    
        score_arr = []
        score_total = 0
        if self.import_price_result is not None:
            res_group = self.import_price_result.groupby("result")
            ins_unique = res_group.nunique()
            total_cnt = ins_unique.values[:,1].sum()
            for i in range(4):
                cnt_values = ins_unique[ins_unique.index==i].values
                if cnt_values.shape[0]==0:
                    cnt = 0
                else:
                    cnt = cnt_values[0,1]
                rate = cnt/total_cnt
                score_arr.append(rate)
                # print("cnt:{} with score:{},total_cnt:{},rate:{}".format(cnt,i,total_cnt,rate))
                self.log("score_{} rate".format(i), rate, prog_bar=True) 
            self.log("total cnt", total_cnt, prog_bar=True)  
            # 综合评估计算总的得分情况，需要满足错误分和正确分达到一定阈值
            if score_arr[-1]>0.2 and score_arr[0]<0.09:
                # 正确错误分比例来决定评分
                score_total = score_arr[-1]/score_arr[0] + score_arr[2]/score_arr[1]
            else:
                score_total = 0
        self.log("total_imp_cnt", self.total_imp_cnt, prog_bar=True)  
        self.log("score_total", score_total, prog_bar=True) 
                
        
        