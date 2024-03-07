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
from sklearn.manifold import MDS

from darts_pro.data_extension.custom_module import viz_target,viz_result_suc,viz_result_fail,viz_result_nor
from darts_pro.act_model.sdcn_ts import SdcnTs
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer
from cus_utils.common_compute import build_symmetric_adj,batch_cov,pairwise_distances
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX
from losses.clustering_loss import ClusteringLoss
from losses.clustering_loss import target_distribution
from losses.hsan_metirc_util import phi,high_confidence,pseudo_matrix,comprehensive_similarity
from cus_utils.visualization import clu_coords_viz

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
        step_mode="pretrain",
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
        self.switch_epoch_num = 0
        self.switch_flag = 0
        self.step_mode=step_mode
        # 初始化中间结果数据
        self.training_step_outputs = [[] for _ in range(self.output_data_len)]
        
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
            # 记录动态数据长度，后续需要切片
            self.dynamic_conv_shape = past_target_shape + past_covariates_shape
            input_dim = (
                past_target_shape
                + past_covariates_shape
                + historic_future_covariates_shape
            )
    
            output_dim = 1
    
            future_cov_dim = (
                future_covariates.shape[1] if future_covariates is not None else 0
            )
            # 由于使用自监督，则取消未来协变量
            future_cov_dim = 0
            
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
                if step_mode=="complete":
                    # 根据过去目标值(Past Target),生成邻接矩阵
                    with torch.no_grad():
                        # 使用目标协变量构造邻接矩阵
                        adj_target = past_convs_item[:,:,:1]
                        # 生成symmetric邻接矩阵以及拉普拉斯矩阵--使用协方差矩阵代替
                        adj_matrix = batch_cov(adj_target)[0]
                        # adj_matrix = build_symmetric_adj(adj_target,device=self.device,distance_func=self.criterion.ccc_distance_torch)
                        # 如果维度不够，则补0
                        if adj_matrix.shape[-1]<batch_size:
                            pad_zize = batch_size - adj_matrix.shape[0]
                            adj_matrix = torch.nn.functional.pad(adj_matrix, (0, pad_zize, 0, pad_zize))
                        adj_matrix = adj_matrix.double().to(self.device)      
                else:
                    adj_matrix = None          
                x_in_item = (past_convs_item,x_in[1],x_in[2])
                # 使用embedding组合以及邻接矩阵作为输入
                out = m(x_in_item,adj_matrix,mode=step_mode)
                # 完整模式下，需要进行2次模型处理
                if step_mode=="complete":
                    out_again = m(x_in_item,adj_matrix,mode=step_mode)
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
        # 全模式训练时，第一个轮次不进行梯度，只取得特征数据
        if self.current_epoch>self.switch_epoch_num:
            for model_seq in range(len(self.sub_models)):
                # 取得最近一次的特征中间值，作为聚类输入数据
                z = [output_item[-2] for output_item in self.training_step_outputs[model_seq]]
                z = torch.concat(z,dim=0).detach().cpu().numpy()
                n_clusters = len(CLASS_SIMPLE_VALUES.keys())
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(z)
                model = self.sub_models[model_seq]
                # 直接把初始化簇心值赋予模型内的参数
                model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)        
                # 放开梯度冻结
                self.freeze_apply(model_seq,flag=1)       
            self.switch_flag = 1 
        else:
            # 梯度冻结
            for model_seq in range(len(self.sub_models)):
                self.freeze_apply(model_seq)
                
    def output_postprocess(self,output,index):
        """对于单步输出的补充"""
        
        # 保存训练结果数据，用于后续分析,只在特定轮次进行
        if self.step_mode=="complete" and self.current_epoch==self.switch_epoch_num:
            # 只需要实际数据，忽略模拟数据
            output_act = output[index]
            self.training_step_outputs[index].append(output_act[0])
    
    
    def training_step_real(self, train_batch, batch_idx) -> torch.Tensor:
        ret =  super().training_step_real(train_batch, batch_idx)
        return ret
    
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
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), (future_target,target_class,target_info,y_transform,past_target),optimizers_idx=-1)
        (corr_loss_combine,kl_loss,ce_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_kl_loss_{}".format(i), kl_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            self.log("val_ce_loss_{}".format(i), ce_loss[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        if self.step_mode=="pretrain" or self.switch_flag==0:
            return loss,detail_loss,output
        
        # 准确率的计算
        import_price_result,z_values = self.compute_real_class_acc(output_data=output,target_info=target_info,target_class=target_class)          
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
        
        # 可视化
        whole_target = np.concatenate((past_target.cpu().numpy(),future_target.cpu().numpy()),axis=1)
        target_inverse = self.get_inverse_data(whole_target,target_info=target_info,scaler=scaler)        
        output_viz = z_values.transpose(1,2,0)
        # self.viz_results(output_viz, target_inverse, import_price_result, batch_idx, target_vr_class, target_info, viz_target)
        
        # 聚类可视化
        self.clustring_viz(z_values,target_class=target_vr_class)
        
        return loss,detail_loss,output

    def compute_real_class_acc(self,target_info=None,output_data=None,target_class=None):
        """计算涨跌幅分类准确度"""
        
        # 使用分类判断
        import_index,z_values = self.build_import_index(output_data=output_data)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_result = \
            self.collect_result(import_index, target_class.cpu().numpy(), target_info)
        
        return import_price_result,z_values
           
    def build_import_index(self,output_data=None):  
        """生成涨幅达标的预测数据下标"""
        
        p_values = []
        q_values = []
        pred_values = []
        z_values = []
        for i in range(len(output_data)):
            output_item,out_again = output_data[i] 
            _, tmp_q, _, _,_ = output_item
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)           
            x_bar, q, pred, _,z =  out_again 
            p_values.append(p)
            q_values.append(q)
            pred_values.append(pred)
            z_values.append(z)
        
        p_values = torch.stack(p_values).cpu().numpy()
        q_values = torch.stack(q_values).cpu().numpy()
        pred_values = torch.stack(pred_values).cpu().numpy()
        z_values = torch.stack(z_values).cpu().numpy()
        
        p_import_index = self.compute_single_target(p_values)
        q_import_index = self.compute_single_target(q_values)
        pred_import_index = self.compute_single_target(pred_values)
        
        return pred_import_index,z_values       
    
    def compute_single_target(self,values):     
        # 综合筛选    
        p_import_index = np.all(values.argmax(2)==3,axis=0)
        # 单指标筛选
        p_import_index = np.where(values[2].argmax(1)==3)[0]    
        return p_import_index
        
    
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
        
        # total_index = 0                    
        # for result,group in res_group:
        #     r_index = -1
        #     unique_group = group.drop_duplicates(subset=['instrument'], keep='first')
        #     for index, row in unique_group.iterrows():
        #         r_index += 1
        #         total_index += 1
        #         s_index = row["imp_index"]
        #         ts = target_info[s_index]
        #         pred_data = output_inverse[s_index]
        #         pred_center_data = pred_data[:,0]
        #         pred_second_data = pred_data[:,1]
        #         target_item = target_inverse[s_index]
        #         pred_third_data = pred_data[:,2]
        #         # 可以从全局变量中，通过索引获得实际价格
        #         # df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
        #         #                         (df_all["instrument_rank"]==ts["item_rank_code"])]            
        #         win = "win{}_{}".format(ts["instrument"],ts["future_start"])
        #         if r_index>15:
        #             break
        #         if result==CLASS_SIMPLE_VALUE_MAX:                 
        #             viz = viz_result_suc
        #         elif result==0:                 
        #             viz = viz_result_fail   
        #         else:
        #             viz = viz_result_nor  

    def clustring_viz(self,z_values,target_class=None):
        """使用聚类可视化方法进行测试"""
        
        len_t = len(self.sub_models)
        combine_index = np.where((target_class==3)|(target_class==0)|(target_class==1)|(target_class==2))[0]
        labels = target_class[combine_index]
        labels = np.concatenate((np.array([0,1,2,3]),labels))
        for i in range(len_t):
            model = self.sub_models[i]
            # 取得簇心参数，并与特征输出进行距离比较
            cluster_center = model.cluster_layer.data.cpu().numpy()
            z_value = z_values[i]
            data = np.concatenate((cluster_center,z_value),axis=0)
            # 生成二维坐标数据
            mds = MDS(n_components=2, dissimilarity='euclidean',random_state=1)
            coords = mds.fit_transform(data)      
            # 重点标记学习到的簇心
            imp_index=np.array([j for j in range(cluster_center.shape[0])])      
            # 作图    
            clu_coords_viz(coords,imp_index=imp_index,labels=labels,name="cluster_{}".format(i))             
                       
    def dump_val_data(self,val_batch,output):
        pass
        # output_real = []
        # for output_item in output:
        #     output_item,out_again = output_item
        #     x_bar, q, pred, _,z =  out_again 
        #     p = target_distribution(q.data) 
        #     output_real.append([z.cpu().numpy(),p.cpu().numpy(),pred.cpu().numpy()])
        # # output_real = np.array(output_real)  
        # (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info,rank_scalers) = val_batch 
        # data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
        #                  future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]            
        # output_combine = (output_real,data)
        # pickle.dump(output_combine,self.valid_fout)      
           
        