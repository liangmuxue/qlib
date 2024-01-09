import os

from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.models.forecasting.block_rnn_model import _BlockRNNModule
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.data.training_dataset import (
    MixedCovariatesTrainingDataset
)
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
from cus_utils.process import create_from_cls_and_kwargs

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
from darts_pro.data_extension.custom_tcn_model import LSTMClassifier
from darts_pro.data_extension.ts_transformer import TSTransformerEncoderClassiregressor
from cus_utils.common_compute import compute_price_class,compute_price_class_batch,slope_classify_compute,slope_classify_compute_batch
from tft.class_define import CLASS_SIMPLE_VALUES

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_model import _TFTCusModule,TFTExtModel
from darts_pro.data_extension.batch_dataset import BatchDataset

class _TFTModuleAsis(_TFTCusModule):
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
        loss_number=3,
        device="cpu",
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    device=device,**kwargs)  
        self.train_data = []
        self.valid_data = []
        if not os.path.exists(batch_file_path):
            os.makedirs(batch_file_path)         
        self.train_filepath = "{}/train_batch.pickel".format(batch_file_path)
        self.train_part_filepath = "{}/train_part_batch.pickel".format(batch_file_path)
        self.valid_filepath = "{}/valid_batch.pickel".format(batch_file_path)
        
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """use to export data"""
        
        train_batch = self.filter_batch_by_condition(train_batch,filter_conv_index=self.filter_conv_index)
        # 过滤后，进行数据增强
        train_batch = self.dynamic_build_training_data(train_batch).transpose(1,0)     
        pickle.dump(train_batch,self.train_fout) 
        # print("len(self.train_data):{}".format(len(self.train_data)))
        fake_loss = torch.ones(1,requires_grad=True).to(self.device)
        return fake_loss

    def dynamic_build_training_data(self,data_batch,rev_threhold=1):
        """使用数据增强，调整训练数据"""
        
        (past_targets,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler_tuple
           ,target_classes,targets,target_infos) = data_batch
        rtn_batch = []
        import_cnt = 0
        for i in range(past_targets.shape[0]):
            past_covariate = past_covariates[i].cpu().numpy()
            future_past_covariate = scaler_tuple[i][1]
            scaler = scaler_tuple[i][0]
            past_target = past_targets[i].cpu().numpy()
            future_target = targets[i].cpu().numpy()
            target_info = target_infos[i]
            target_class = target_classes[i].cpu().numpy()
            # 重点关注价格指数,只对价格达到上涨幅度的目标进行增强
            p_taraget_class = target_class[0]
            # Do Not Aug
            if not p_taraget_class in [3] or True:
                rtn_item = (past_target,past_covariate,historic_future_covariates[i].cpu().numpy(),future_covariates[i].cpu().numpy(),
                            static_covariates[i].cpu().numpy(),(scaler,future_past_covariate),target_class,future_target,target_info)
                rtn_batch.append(rtn_item)                
                continue
            
            for j in range(3):
                import_cnt += 1
                target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
                target_unscale = scaler.inverse_transform(target[0,:,:])
                
                # 把past和future重新组合，统一增强
                covariate = np.expand_dims(np.concatenate((past_covariate,future_past_covariate),axis=0),axis=0)
                if np.random.randint(0,2)==1:
                    # 量化方式调整
                    X_aug, Y_aug = tsaug.Quantize(n_levels=10).augment(covariate, target)
                else:
                    # 降低时间分辨率的方式调整
                    X_aug, Y_aug = tsaug.Pool(size=2).augment(covariate, target)
                past_covariate = X_aug[0,:self.input_chunk_length,:] 
                future_target = Y_aug[0,self.input_chunk_length:,:]
                past_target = Y_aug[0,:self.input_chunk_length,:]
                rtn_item = (past_target,past_covariate,historic_future_covariates[i].cpu().numpy(),future_covariates[i].cpu().numpy(),
                            static_covariates[i].cpu().numpy(),(scaler,future_past_covariate),target_class,future_target,target_info)
                rtn_batch.append(rtn_item)
        
        # if np.random.randint(0,900)==1:
        #     # 可视化增强的数据
        #     index = np.random.randint(0,12)
        #     win = "win_{}".format(index)
        #     target_title = "code_{}".format(target_info["item_rank_code"])
        #     names = ["label","price"]
        #     price_array = np.expand_dims(target_info["price_array"],axis=-1)
        #     view_data = np.concatenate((target_unscale[:,:1],price_array),axis=-1)
        #     viz_input_aug.viz_matrix_var(view_data,win=win,title=target_title,names=names)        

        # print("import cnt:{}/total cnt:{}".format(import_cnt,len(rtn_batch)))        
        rtn_batch = np.array(rtn_batch,dtype=object)
        return rtn_batch
            
        # 重点关注前期走势比较平的
        # slope = slope_classify_compute(focus_target,threhold=2)
        # if slope!=SLOPE_SHAPE_SMOOTH:
        #     return None
        
        # target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
        # target_unscale = self.model.get_inverse_data(target[:,:,0],target_info=target_info,single_way=True).transpose(1,0)
        
    def validation_step(self, val_batch_ori, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        # SANITY CHECKING模式下，不进行处理
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return            
        # 只关注重点部分
        val_batch = self.filter_batch_by_condition(val_batch_ori,filter_conv_index=self.filter_conv_index)
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = val_batch 
        data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
                         future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]
        print("dump valid,batch:{},target shape:{}".format(batch_idx,past_target.shape))
        pickle.dump(data,self.valid_fout) 
        fake_loss = torch.ones(1).to(self.device)
        self.log("val_loss", fake_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        return fake_loss     
     
    def on_train_start(self):  
        self.total_target_cnt = 0
        torch.set_grad_enabled(True)
        self.train_fout = open(self.train_filepath, "wb")
        self.train_part_fout = open(self.train_part_filepath, "wb")  
        self.valid_fout = open(self.valid_filepath, "wb")
                                     
    def on_train_epoch_end(self):
        print("self.total_target_cnt:{}".format(self.total_target_cnt))
        self.train_fout.close()
        self.train_part_fout.close()
        self.valid_fout.close()
        
    def on_validation_epoch_end(self):
        print("pass")
            
class TFTAsisModel(TFTExtModel):
    
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: Union[int, List[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            Dict[str, Union[int, Tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_weighted_loss_func:bool = False,
        loss_number=3,
        monitor=None,
        mode="train",
        no_dynamic_data=False,
        past_split=None,
        batch_file_path=None,
        filter_conv_index=0,
        **kwargs,
    ):
        
        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,num_attention_heads,
                         full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,add_relative_index,
                         loss_fn,likelihood,norm_type,use_weighted_loss_func,loss_number,monitor,past_split=past_split,no_dynamic_data=no_dynamic_data,**kwargs)
        self.batch_file_path = batch_file_path
        
    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            target_scaler,
            future_target_class,
            future_target,
            target_info,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = (1,1)
        
        
        # 根据拆分的过去协变量，生成多个配置
        variables_meta_array = []
        for i in range(len(self.past_split)):
            past_index = self.past_split[i]
            past_covariate_item = past_covariate[:,past_index[0]:past_index[1]]
            tensors = [
                past_target,
                past_covariate_item,
                historic_future_covariate,  # for time varying encoders
                future_covariate,
                future_target,  # for time varying decoders
                static_covariates,  # for static encoder
            ]            
            variables_meta,categorical_embedding_sizes = self._build_vriable_metas(tensors, static_covariates,seq=i)
            variables_meta_array.append(variables_meta)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        
        return _TFTModuleAsis(
            output_dim=self.output_dim,
            variables_meta_array=variables_meta_array,
            num_static_components=n_static_components,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            feed_forward=self.feed_forward,
            hidden_continuous_size=self.hidden_continuous_size,
            categorical_embedding_sizes=self.categorical_embedding_sizes,
            add_relative_index=self.add_relative_index,
            norm_type=self.norm_type,
            use_weighted_loss_func=self.use_weighted_loss_func,
            loss_number=self.loss_number,
            past_split=self.past_split,
            filter_conv_index=self.filter_conv_index,
            batch_file_path=self.batch_file_path,
            device=self.device,
            **self.pl_module_params,
        )    
    
class _TFTModuleBatch(_TFTCusModule):
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
        loss_number=3,
        batch_file_path=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,
                                    device=device,**kwargs)  
        self.lr_freq = {"interval":"epoch","frequency":1}
        
        self.train_filepath = "{}/train_output_batch.pickel".format(batch_file_path)
        self.valid_filepath = "{}/valid_output_batch.pickel".format(batch_file_path)
        
    
    def on_train_start(self): 
        super().on_train_start()
        self.train_output_flag = True
        # 先训练一段时间，然后冻结第一阶段网络，只更新第二阶段网络
        self.apply_params_freeze()
                
    def on_validation_start(self): 
        super().on_validation_start()
        self.valid_output_flag = True
                
    def on_train_epoch_start(self):  
        super().on_train_epoch_start()
        if self.train_output_flag:
            self.train_fout = open(self.train_filepath, "wb")
    def on_train_epoch_end(self):  
        super().on_train_epoch_start()
        if self.train_output_flag:
            self.train_output_flag = False
            self.train_fout.close()
    def on_validation_epoch_start(self):  
        super().on_validation_epoch_start()
        if self.valid_output_flag:
            self.valid_fout = open(self.valid_filepath, "wb")     
            
    def on_validation_epoch_end(self):  
        super().on_validation_epoch_end()
        if self.valid_output_flag:
            self.valid_output_flag = False
            self.valid_fout.close()          
        # 动态冻结网络参数
        corr_loss_combine = []
        self.apply_params_freeze()
    
    def apply_params_freeze(self):
        # 先训练一段时间，然后冻结第一阶段网络，只更新第二阶段网络
        if self.current_epoch>1000:
            for i in range(len(self.sub_models)):
                self.freeze_apply(mode=i)   
            self.freeze_apply(mode=(len(self.sub_models)+1),flag=1)       
                                           
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """重载原方法，直接使用已经加工好的数据"""

        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler,target_class,target,target_info,rank_targets) = train_batch    
         
        # 使用排序目标替换原数据--Cancel
        train_batch_convert = (past_target,past_covariates, historic_future_covariates,future_covariates, 
                               static_covariates,scaler,target_class,target,target_info)
                               
        loss,detail_loss,output = self.training_step_real(train_batch_convert, batch_idx) 
        if self.train_output_flag:
            output = [output_item.detach().cpu().numpy() for output_item in output]
            data = [past_target.detach().cpu().numpy(),past_covariates.detach().cpu().numpy(), historic_future_covariates.detach().cpu().numpy(),
                             future_covariates.detach().cpu().numpy(),static_covariates.detach().cpu().numpy(),scaler,target_class.cpu().detach().numpy(),
                             target.cpu().detach().numpy(),target_info]                
            output_combine = (output,data)
            pickle.dump(output_combine,self.train_fout)  
        # (mse_loss,value_diff_loss,corr_loss,ce_loss,mean_threhold) = detail_loss
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,
         static_covariates,scaler,target_class,target,target_info,rank_targets) = val_batch    
         
        # 使用排序目标替换原数据
        val_batch_convert = (past_target,past_covariates, historic_future_covariates,future_covariates, 
                               static_covariates,scaler,target_class,target,target_info,rank_targets)
                
        loss,detail_loss,output = self.validation_step_real(val_batch_convert, batch_idx)  
        
        if self.trainer.state.stage!=RunningStage.SANITY_CHECKING and self.valid_output_flag:
            output = [output_item.cpu().numpy() for output_item in output]
            (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info,rank_scalers) = val_batch 
            data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
                             future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]            
            output_combine = (output,data)
            pickle.dump(output_combine,self.valid_fout)         
        return loss,detail_loss   
    
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        return super().validation_step_real(val_batch[:-1], batch_idx)
        
        input_batch = self._process_input_batch(val_batch[:5])
        # 收集目标数据用于分类
        scaler_tuple,target_class,future_target,target_info,rank_targets = val_batch[5:]  
        scaler = [s[0] for s in scaler_tuple]
        # 使用排序号作为目标
        (output,vr_class,tar_class) = self(input_batch,rank_targets[0],scaler,past_target=val_batch[0],target_info=target_info,optimizer_idx=-1)

        past_target = val_batch[0]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        whole_target = np.concatenate((past_target.cpu().numpy(),future_target.cpu().numpy()),axis=1)
        target_inverse = self.get_inverse_data(whole_target,target_info=target_info,scaler=scaler)
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (rank_targets[0],target_class,target_info,None),optimizers_idx=-1)
        (corr_loss_combine,ce_loss,value_diff_loss) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        
        output_combine = [output_item[:,:,0,0] for output_item in output]
        output_combine = torch.stack(output_combine,dim=2).cpu().numpy()        
        # 涨跌幅度类别的准确率
        import_index = self.build_import_index(output_combine, target_inverse)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_result = \
            self.collect_result(import_index, target_vr_class, target_info)
        total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        if self.total_imp_cnt==0:
            self.total_imp_cnt = total_imp_cnt
        else:
            self.total_imp_cnt += total_imp_cnt
        
        past_target = val_batch[0]
 
        # 可视化
        self.val_metric_show(output,future_target,target_vr_class,output_inverse=output_combine,vr_class=vr_class,
                             target_inverse=target_inverse,target_info=target_info,import_price_result=import_price_result,past_covariate=None,
                            batch_idx=batch_idx)
               
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
                
        return loss,detail_loss,output
      
    
    # def build_import_index(self,output_inverse=None,target_inverse=None):  
    #     """重载父类方法，生成涨幅达标的预测数据下标"""
    #
    #     output_label_inverse = output_inverse[:,:,0] 
    #     output_second_inverse = output_inverse[:,:,1]
    #     output_third_inverse = output_inverse[:,:,2]
    #
    #     third_rank = np.mean(output_third_inverse,axis=1)
    #     import_index = np.argsort(third_rank,axis=0)[:10]
    #
    #     return import_index    
        
    def _process_input_batch(
        self, input_batch
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """重载方法，把过去协变量数值转换为排序数值"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        
        return super()._process_input_batch(input_batch)
        
        dim_variable = 2
        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[:,:,past_conv_index[0]:past_conv_index[1]]
            past_target_item = past_target[:,:,i]
            # 协变量数值转换为排序号
            _,indices = torch.sort(past_target_item,0)
            _, idx_unsort = torch.sort(indices, dim=0)
            
            if self.trainer.state.stage==RunningStage.TRAINING:
                idx_unsort = idx_unsort.cpu().numpy()
            else:
                idx_unsort = idx_unsort.cpu().numpy()
            past_convert = torch.Tensor(MinMaxScaler().fit_transform(idx_unsort)).to(self.device)
            past_convert = torch.unsqueeze(past_convert,-1)
            # 修改协变量生成模式，只取自相关目标作为协变量
            conv_defs = [
                        past_convert,
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

    def _construct_classify_layer(self, input_dim,output_dim,device=None):
        """分类特征值输出
          Params
            layer_num： 层数
            input_dim： 序列长度
            output_dim： 类别数
        """
        return super()._construct_classify_layer(input_dim, output_dim)
        # len = self.input_chunk_length + self.output_chunk_length
        # class_layer = TSTransformerEncoderClassiregressor(input_dim, num_classes=output_dim, max_len=len,device=device)
        # class_layer = class_layer.cuda(device)
        # return class_layer
                    
class TFTBatchModel(TFTExtModel):
    
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_size: Union[int, List[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            Dict[str, Union[int, Tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_weighted_loss_func:bool = False,
        loss_number=3,
        monitor=None,
        mode="train",
        past_split=None,
        filter_conv_index=0,
        batch_file_path=None,
        **kwargs,
    ):
        
        super().__init__(input_chunk_length,output_chunk_length,hidden_size,lstm_layers,num_attention_heads,
                         full_attention,feed_forward,dropout,hidden_continuous_size,categorical_embedding_sizes,add_relative_index,
                         loss_fn,likelihood,norm_type,use_weighted_loss_func,loss_number,monitor,
                         past_split=past_split,filter_conv_index=filter_conv_index,**kwargs)
        self.batch_file_path = batch_file_path
    
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="train"
    ) -> BatchDataset:
        
        return BatchDataset(
            filepath = "{}/{}_batch.pickel".format(self.batch_file_path,mode)
        )     

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """重载创建模型方法，使用自定义模型"""
        
        
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            target_scaler,
            future_target_class,
            future_target,
            target_info,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
        
        # 修改原内容，固定设置为1，以适应后续分别运行的独立模型
        self.output_dim = self.define_output_dim()
        
        # 根据拆分的过去协变量，生成多个配置
        variables_meta_array = []
        for i in range(len(self.past_split)):
            past_index = self.past_split[i]
            past_covariate_item = past_covariate[:,past_index[0]:past_index[1]]
            tensors = [
                past_target,
                past_covariate_item,
                historic_future_covariate,  # for time varying encoders
                future_covariate,
                future_target,  # for time varying decoders
                static_covariates,  # for static encoder
            ]            
            variables_meta,categorical_embedding_sizes = self._build_vriable_metas(tensors, static_covariates,seq=i)
            variables_meta_array.append(variables_meta)
        
        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        
        return _TFTModuleBatch(
            output_dim=self.output_dim,
            variables_meta_array=variables_meta_array,
            num_static_components=n_static_components,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            feed_forward=self.feed_forward,
            hidden_continuous_size=self.hidden_continuous_size,
            categorical_embedding_sizes=self.categorical_embedding_sizes,
            add_relative_index=self.add_relative_index,
            norm_type=self.norm_type,
            use_weighted_loss_func=self.use_weighted_loss_func,
            loss_number=self.loss_number,
            past_split=self.past_split,
            filter_conv_index=self.filter_conv_index,
            device=self.device,
            batch_file_path=self.batch_file_path,
            **self.pl_module_params,
        )   
               
    def _batch_collate_filter(self,ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """

        batch = ori_batch 
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                if elem.dtype.hasobject:
                    sample_list = [sample[i] for sample in batch]
                    aggregated.append(
                        sample_list
                    )                   
                else:
                    sample_list = [sample[i] for sample in batch]
                    aggregated.append(
                        torch.from_numpy(np.stack(sample_list, axis=0))
                    )  
            elif isinstance(elem, MinMaxScaler):
                aggregated.append([sample[i] for sample in batch])
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        
        # 添加排序号的目标
        future_target = aggregated[-2]
        _,indices = torch.sort(future_target,0)
        _, idx_unsort = torch.sort(indices, dim=0)
        # 归一化
        rank_scalers = []
        idx_unsort_verse = []
        for i in range(idx_unsort.shape[-1]):
            rank_scaler = MinMaxScaler()
            idx_unsort_item = rank_scaler.fit_transform(idx_unsort[:,:,i].numpy())
            idx_unsort_verse.append(idx_unsort_item)
            rank_scalers.append(idx_unsort_item)
        idx_unsort = torch.Tensor(np.array(idx_unsort_verse)).permute(1,2,0)
        aggregated.append([idx_unsort,rank_scalers])
        return tuple(aggregated)
        

    def dynamic_build_training_data(self,item,rev_threhold=1):
        """使用数据增强，调整训练数据"""
        
        past_covariate = item[1]
        future_past_covariate = item[5][1]
        past_target = item[0]
        future_target = item[-2]
        target_info = item[-1]
        scaler = item[5][0]
        last_target_class = item[-3][1][0]
        rev_data = past_covariate[:,0]
        
        # 重点关注价格指数,对价格涨幅达标的数据进行增强
        p_taraget_class = compute_price_class(target_info["price_array"][self.input_chunk_length:])
        if p_taraget_class in [0,1]:
            return None
        
        target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
        target_unscale = scaler.inverse_transform(target[0,:,:])
        # focus_target = target_unscale[self.input_chunk_length:]  
        # target_slope = (focus_target[1:] - focus_target[:-1])/focus_target[:-1]*100

        # 与近期高点比较，不能差太多
        # label_array = target_info["label_array"]
        # recent_data = label_array[:self.input_chunk_length]
        # recent_max = recent_data.max()
        # if (recent_max-recent_data[-1])/recent_max>2/100:
        #     return None
        
        # # 关注动量指标比较剧烈的
        # rev_cov_recent = rev_data[-5:]
        # # 近期均值大于阈值1
        # rev_bool = np.mean(rev_cov_recent)>rev_threhold
        # # 最近数值处于最高点
        # rev_cov_max = np.max(rev_data)
        # rev_bool = rev_bool & ((rev_cov_max-rev_cov_recent[-1])<=0.01)        
        # if not rev_bool:
        #     return None
              
        # 关注最后一段下跌的
        # if last_target_class==1 and np.random.randint(0,20)!=1:
        #     return None
        
        # 把past和future重新组合，统一增强
        covariate = np.expand_dims(np.concatenate((past_covariate,future_past_covariate),axis=0),axis=0)
        if np.random.randint(0,2)==1:
            # 量化方式调整
            X_aug, Y_aug = tsaug.Quantize(n_levels=10).augment(covariate, target)
        else:
            # 降低时间分辨率的方式调整
            X_aug, Y_aug = tsaug.Pool(size=2).augment(covariate, target)
        past_covariate = X_aug[0,:self.input_chunk_length,:] 
        future_target = Y_aug[0,self.input_chunk_length:,:]
        past_target = Y_aug[0,:self.input_chunk_length,:]
        rtn_item = (past_target,past_covariate,item[2],item[3],item[4],item[5][0],item[6],future_target,item[8])
        
        # if np.random.randint(0,900)==1:
        #     # 可视化增强的数据
        #     index = np.random.randint(0,12)
        #     win = "win_{}".format(index)
        #     target_title = "code_{}".format(target_info["item_rank_code"])
        #     names = ["label","price"]
        #     price_array = np.expand_dims(target_info["price_array"],axis=-1)
        #     view_data = np.concatenate((target_unscale[:,:1],price_array),axis=-1)
        #     viz_input_aug.viz_matrix_var(view_data,win=win,title=target_title,names=names)        
        return rtn_item

               
        