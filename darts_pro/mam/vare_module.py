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

from darts_pro.act_model.vare_ts import VaRE
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances,corr_compute,ccc_distance_torch,find_nearest
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX
from losses.clustering_loss import VaRELoss,cluster_acc
from cus_utils.common_compute import target_distribution,normalization_axis,intersect2d
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity
from cus_utils.visualization import clu_coords_viz
from cus_utils.clustering import get_cluster_center
import cus_utils.global_var as global_var

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_module import _TFTModuleBatch

class VaREModule(_TFTModuleBatch):
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
            sequence_length = past_target.shape[0]
            if self.static_datas is None:
                static_feat = None
            else:
                # Standard Static Data
                static_feat = torch.Tensor(StandardScaler().fit_transform(self.static_datas)).double().to(device)
                # static_feat = None 
                          
            model = VaRE(
                # Normal Part
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
                # VaRE Part
                batch_size=self.batch_size,
                n_cluster=len(CLASS_SIMPLE_VALUES.keys()),
                activation="prelu",
                sequence_length=self.output_chunk_length,
                static_feat=static_feat,
                device=device,
                **kwargs,
            )           
            
            return model
        
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        future_target,
        past_target=None,
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
                out = m(past_convs_item,mode=step_mode)
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
        return VaRELoss(device=device,ref_model=model) 

    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""

        if self.switch_flag==0 or self.step_mode=="pretrain":
            step_mode = "pretrain"
        else:
            step_mode = "complete"   
            
        (future_target,target_class,target_info,past_target,input_batch) = target   
        past_covariates = input_batch[0]
        return self.criterion(output,(future_target,target_class,past_target,past_covariates),mode=step_mode,optimizers_idx=optimizers_idx)

    def on_validation_start(self): 
        super().on_validation_start()
        if self.step_mode=="complete":
            # 如果是加载之前的权重继续训练，则需要判断并重置switch_flag变量
            if self.current_epoch>self.switch_epoch_num:
                self.switch_flag = 1

    def on_train_epoch_start(self):  
        super().on_train_epoch_start()
        # 预训练模式下，没有下面的策略
        if self.step_mode=="pretrain":
            return
        # 如果已经做过切换了，则不进行处理
        if self.switch_flag==1:
            return
        # 全模式训练时，第一个轮次不进行梯度，只取得特征数据,并生成聚类初始数据
        if self.current_epoch>self.switch_epoch_num:
            for model_seq in range(len(self.sub_models)):
                # 取得最近一次的预测特征值，作为混合高斯分布的输入数据
                z_value = torch.concat(self.training_step_outputs[model_seq])
                z_value = z_value.detach().cpu().numpy()
                Y = [target_item[...,0] for target_item in self.training_step_targets[model_seq]]
                Y = torch.concat(Y)
                Y = Y.reshape(-1).detach().cpu().numpy().astype(np.int8)
                n_clusters = len(CLASS_SIMPLE_VALUES.keys())
                gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
                # 和价格目标进行比较，检查正确率
                pre = gmm.fit_predict(z_value)
                print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))
                # 参数初始赋值
                self.sub_models[model_seq].lmbd.hidden_to_logvar.load_state_dict(self.sub_models[model_seq].lmbd.hidden_to_mean.state_dict())
                self.sub_models[model_seq].pi_.data = torch.from_numpy(gmm.weights_).to(self.device).float()
                self.sub_models[model_seq].mu_c.data = torch.from_numpy(gmm.means_).to(self.device).float()
                self.sub_models[model_seq].log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).to(self.device).float())                
                # 放开梯度冻结
                self.freeze_apply(model_seq,flag=1)       
            self.switch_flag = 1 
        else:
            # 梯度冻结
            for model_seq in range(len(self.sub_models)):
                self.freeze_apply(model_seq)
                
    def output_postprocess(self,output,targets,index):
        """对于不同指标输出的补充"""
        
        # 保存训练结果数据，用于后续分析,只在特定轮次进行
        if self.step_mode=="complete" and self.current_epoch==self.switch_epoch_num:
            # 取得对应目标
            output_act = output[index]
            # 预存隐变量（均值变量））
            z_mu = output_act[1]
            self.training_step_outputs[index].append(z_mu)
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
        target_class = target_class[:,:,0]     
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            y_transform = None 
            (output,vr_class,tar_class) = self(input_batch,future_target,past_target=train_batch[0],
                                               target_info=target_info,optimizer_idx=i)
            self.output_postprocess(output,target_class,i)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (future_target,target_class,target_info,past_target,input_batch),optimizers_idx=i)
            (corr_loss_combine,elbu_loss,comb_detail_loss) = detail_loss 
            # self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            self.log("train_elbu_loss_{}".format(i), elbu_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
            # self.log("train_ce_loss_{}".format(i), ce_loss[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
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
        scaler = [s[0] for s in scaler_tuple]
        (output,vr_class,vr_class_list) = self(input_batch,future_target,past_target=val_batch[0],
                                               target_info=target_info,optimizer_idx=-1)
        
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), 
                    (future_target,target_class,target_info,past_target,input_batch),optimizers_idx=-1)
        (corr_loss_combine,elbu_loss,comb_detail_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        preds_combine = []
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_elbu_loss_{}".format(i), elbu_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            if len(comb_detail_loss)>0:
                self.log("val_rec_loss_{}".format(i), comb_detail_loss[i][0], batch_size=val_batch[0].shape[0], prog_bar=True)
                self.log("val_clu_loss_{}".format(i), comb_detail_loss[i][1], batch_size=val_batch[0].shape[0], prog_bar=True)
                self.log("val_p_loss_{}".format(i), comb_detail_loss[i][2], batch_size=val_batch[0].shape[0], prog_bar=True)
            
            past_convs_item = input_batch[0][i] 
            pred,pred_combine = self.sub_models[i].predict(past_convs_item)
            preds_combine.append(pred_combine)
            acc = self.cluster_acc(pred,target_vr_class)
            self.log("val_acc_{}".format(i), acc[0], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = (output,preds_combine)
        if self.step_mode=="pretrain" or self.switch_flag==0:
            return loss,detail_loss,output
        
        # PCA准确率计算
        # self.compute_pac_acc(output,price_target)
        
        # 准确率的计算
        # import_price_result,values = self.compute_real_class_acc(output_data=output,target_info=self.transfer_target_info(target_info),target_class=target_class)          
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
        # # 可视化
        # whole_target = np.concatenate((past_target.cpu().numpy(),future_target.cpu().numpy()),axis=1)
        # target_inverse = self.get_inverse_data(whole_target,target_info=target_info,scaler=scaler)        
        # output_viz = z_values.transpose(1,2,0)
        # self.viz_results(z_values, target_inverse, import_price_result, batch_idx, target_vr_class, target_info, viz_target)
        
        # 聚类可视化
        # if (self.current_epoch%10==1 and batch_idx==0):
        #     self.clustring_viz(values,target_class=target_vr_class,index_no=self.current_epoch)
        
        return loss,detail_loss,output_combine

    def compute_pac_acc(self,output,price_target):
        pca_price_target = price_target[:,:,0,0].cpu().numpy()
        price_index = np.argsort(pca_price_target,axis=1)[:,:50]
        match_cnt_total = []
        for i in range(len(output)):
            output_item = output[i]
            x_bar, q, pred, pred_value,z = output_item     
            pred = pred.cpu().numpy()
            # 通过排序后的交集取得匹配数目   
            pred_index = np.argsort(pred[...,0],axis=1)[:,:50]
            match_cnt = intersect2d(price_index,pred_index)  
            match_cnt_total.append(match_cnt)
        return match_cnt_total
        
    def compute_real_class_acc(self,target_info=None,output_data=None,target_class=None):
        """计算涨跌幅分类准确度"""
        
        # 使用分类判断
        import_index,values = self.build_import_index(output_data=output_data)
        target_class = target_class.view(-1)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class, \
            import_price_result = self.collect_result(import_index, target_class.cpu().numpy(), target_info)
        
        return import_price_result,values

    def cluster_acc(self,Y_pred, Y):
        from scipy.optimize import linear_sum_assignment as linear_assignment
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum(w[ind[0],ind[1]])*1.0/Y_pred.size, w
           
    def build_import_index(self,output_data=None):  
        """生成涨幅达标的预测数据下标"""
        
        p_values = []
        q_values = []
        pred_values = []
        z_values = []
        x_bars = []
        for i in range(len(output_data)):
            output_item = output_data[i] 
            # _, tmp_q, _, _,_ = output_item
            # tmp_q = tmp_q.data
            # p = target_distribution(tmp_q)           
            x_bar, q, pred, pred_value,z =  output_item 
            # p_values.append(p)
            # q_values.append(q)
            pred_values.append(pred)
            z_values.append(z)
            x_bars.append(x_bar)
        
        # p_values = torch.stack(p_values).cpu().numpy().transpose(1,2,0)
        # q_values = torch.stack(q_values).cpu().numpy().transpose(1,2,0)
        pred_values = torch.stack(pred_values).cpu().numpy()
        # pred_values = pred_values.transpose(1,2,3,0)
        # z_values = torch.stack(z_values).cpu().numpy().transpose(1,2,3,0)
        # x_bars = torch.stack(x_bars).cpu().numpy().transpose(1,2,3,0)
        
        # p_import_index = self.compute_single_target(p_values)
        # q_import_index = self.compute_single_target(q_values)
        # pred_values = np.reshape(pred_values,
        #     [pred_values.shape[0]*pred_values.shape[1],pred_values.shape[2],pred_values.shape[3]])
        pred_import_index = self.compute_single_target(pred_values)
        # output_import_index = self.compute_values_target(pred_values)
        pred_import_index = pred_import_index.flatten()
        
        return pred_import_index,(p_values,z_values,pred_values)       
    
    def compute_single_target(self,values):   
        v2 =  values 
        # 排序筛选,取得每组排名前几个，作为候选 
        p_import_index = np.argsort(-values[2],axis=1)[:,:20]
        return p_import_index

    def compute_values_target(self,values,target=None):     
        
        range_compute = (values[:,-1,:]  - values[:,0,:])/np.abs(values[:,0,:])*100
        
        output_label_inverse = values[:,:,0] 
        output_second_inverse = values[:,:,1]
        output_third_inverse = values[:,:,2]       
    
        # 衡量最后一个指标，下跌超过一定幅度
        third_index_bool = (range_compute[:,2] < -50)
        third_index_bool = (output_third_inverse[:,-1] - output_third_inverse[:,-2])/np.abs(output_third_inverse[:,-2]) < -1
        third_index = np.where(third_index_bool)[0]
        return third_index
                   
    
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
        
    def clustring_viz(self,values,target_class=None,index_no=0):
        """使用聚类可视化方法进行测试"""
        
        len_t = len(self.sub_models)
        for i in range(len_t):
            labels = np.concatenate((np.array([0,1,2,3]),target_class))
            model = self.sub_models[i]
            # 取得簇心参数，并与特征输出进行距离比较
            cluster_center = model.cluster_layer.data.cpu().numpy()
            # pred_values= values[0][:,:,i]
            p_values = values[0][:,:,i]
            z_values = values[1][:,:,i]
            pred_values = values[2][:,:,i]
            data = np.concatenate((cluster_center,z_values),axis=0)
            data_pred = np.concatenate((cluster_center,pred_values),axis=0)
            # distance_q = corr_compute(cluster_center,pred_values)     
            # data = pairwise_distances(torch.Tensor(data).to(self.device),
            #                     distance_func=self.criterion.ccc_distance_torch,
            #                             make_symmetric=True).cpu().numpy()                   
            # 生成二维坐标数据
            # mds = MDS(n_components=2, dissimilarity='precomputed',random_state=1)
            mds = MDS(n_components=2, dissimilarity='euclidean',random_state=1)
            coords = mds.fit_transform(data)     
            coords_pred = mds.fit_transform(data_pred)    
            labels_p = np.argmax(p_values,axis=1)
            labels_p = np.concatenate((np.array([0,1,2,3]),labels_p))
            # 只关注重点类别
            combine_index = np.where((target_class==3)|(target_class==0))[0] + 4
            combine_index = np.concatenate((np.array([0,1,2,3]),combine_index))
            # coords = coords[combine_index]
            # coords_pred = coords_pred[combine_index]
            # labels_p = labels_p[combine_index]
            # labels = labels[combine_index]
            
            # 标记学习到的簇心
            imp_index = np.array([j for j in range(cluster_center.shape[0])])      
            # 对于DNN输出数值的空间分布进行作图    
            # clu_coords_viz(coords,imp_index=imp_index,labels=labels,att_data=None,
            #                name="gcn/cluster_{}_{}".format(index_no,i))   
            # 对于GCN输出数值的空间分布进行作图    
            clu_coords_viz(coords_pred,imp_index=imp_index,labels=labels,att_data=None,
                           name="pred_cluster_{}_{}".format(index_no,i))                        
            # 按照输出概率标绘
            clu_coords_viz(coords_pred,imp_index=None,labels=labels_p,att_data=cluster_center,
                           name="gcn/p_cluster_{}_{}".format(index_no,i))  
    
    def transfer_target_info(self,target_info):
        ts_rtn = []
        for t in target_info:
            for t_iner in t:
                ts_rtn.append(t_iner)
        return ts_rtn
        
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

        # Norm future conv
        historic_future_covariatesa_norm = normalization_axis(historic_future_covariates,axis=2)
        static_covariates_norm = normalization_axis(static_covariates,axis=2)[:,0,:]
        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[...,past_conv_index[0]:past_conv_index[1]]
            # 修改协变量生成模式，只取自相关目标作为协变量
            conv_defs = [
                        past_target[...,i:i+1],
                        past_covariates_item,
                        # historic_future_covariatesa_norm,
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
        # pass
        if self.switch_flag==0 or self.step_mode=="pretrain":
            return
        output,preds_combine = outputs
        output_real = []
        # output_real = np.array(output_real)  
        (past_target,past_covariates, historic_future_covariates,future_covariates,
                static_covariates,scaler_tuple,target_class,future_target,target_info,price_target) = val_batch
        for i,output_item in enumerate(output):
            pred_combine = preds_combine[i]  
            (yita,z,latent,cell_output) = pred_combine
            output_real
        data = [past_target.cpu().numpy(),target_class.cpu().numpy(),
                future_target.cpu().numpy(),price_target.cpu().numpy(),target_info]          
        output_combine = (preds_combine,data)
        pickle.dump(output_combine,self.valid_fout)      
           
        