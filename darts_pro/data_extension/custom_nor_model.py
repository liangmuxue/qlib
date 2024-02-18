import os

from darts.models.forecasting.tide_model import _TideModule
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
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
from sklearn.preprocessing import MinMaxScaler

from cus_utils.common_compute import compute_price_class,compute_price_class_batch,slope_classify_compute,slope_classify_compute_batch
MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

from darts_pro.data_extension.custom_model import TFTExtModel
from darts_pro.data_extension.custom_module import _CusModule,_TFTModuleBatch
from darts_pro.mam.hsan_module import HsanModule
from darts_pro.mam.dink_module import DinkModule
from darts_pro.data_extension.batch_dataset import BatchDataset

class _TFTModuleAsis(_CusModule):
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
        
        fake_loss = torch.ones(1,requires_grad=True).to(self.device)
        train_batch = self.filter_batch_by_condition(train_batch,filter_conv_index=self.filter_conv_index)
        if train_batch is None:
            return fake_loss
        # 过滤后，进行数据增强
        # train_batch = self.dynamic_build_training_data(train_batch).transpose(1,0)     
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = train_batch 
        data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
                         future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]        
        pickle.dump(data,self.train_fout) 
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

    def filter_batch_by_condition(self,data_batch,filter_conv_index=0,rev_threhold=3,recent_threhold=3):
        """按照已知指标，对结果集的重点关注部分进行初步筛选"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler_tuple,target_class,target,target_info) = data_batch    
        
        recent_data = np.array([item["label_array"][:self.input_chunk_length] for item in target_info])
        # 与近期高点比较，不能差太多
        recent_max = np.max(recent_data,axis=1)
        import_index_bool = (recent_max-recent_data[:,-1])/recent_max<(recent_threhold/100)
        # 当前价格不能处于下降趋势
        # less_ret = np.subtract(recent_data.transpose(1,0),recent_data[:,-1]).transpose(1,0)
        # import_index_bool = import_index_bool & (np.sum(less_ret>0,axis=1)<=3)
        
        # 通过指标筛选(配置参数里指定哪种指标)
        # rev_cov = np.array([item["focus2_array"][:self.input_chunk_length] for item in target_info])
        
        # 使用cci指标，突破100后进入筛选池
        # cci_cov = np.array([item["focus1_array"][:self.input_chunk_length] for item in target_info])
        # # 当前记录大于100并且前一条记录明显小于100
        # import_index_bool = (cci_cov[:,-1]>100) & (cci_cov[:,-2]<50)

        # 使用bulls指标
        # bulls_cov = np.array([item["focus2_array"][self.input_chunk_length-5:self.input_chunk_length] for item in target_info])
        # # 需要近期的牛市力度一直处于上方
        # import_index_bool = np.sum(bulls_cov>0,axis=1)>=4
        
        # import_index_bool = self.create_signal_macd(target_info)       
        import_index_bool = self.create_signal_kdj(target_info)    
        if np.sum(import_index_bool)==0:
            return None
        print("total size:{},import_index_bool size:{}".format(past_target.shape[0],np.sum(import_index_bool)))
        rtn_index = np.where(import_index_bool)[0]
        
        data_batch_filter = [past_target[rtn_index,:,:],past_covariates[rtn_index,:,:],historic_future_covariates[rtn_index,:,:],
                            future_covariates[rtn_index,:,:],static_covariates[rtn_index,:,:],
                            np.array(scaler_tuple,dtype=object)[rtn_index],target_class[rtn_index,:,:],
                            target[rtn_index,:,:],np.array(target_info)[rtn_index].tolist()]
        return data_batch_filter
    
    def create_signal_macd(self,target_info):
        """macd指标判断"""
        
        diff_cov = np.array([item["focus1_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        dea_cov = np.array([item["focus2_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        macd_cov = np.array([item["focus3_array"][self.input_chunk_length-5:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即diff快线向上突破dea慢线
        index_bool = (np.sum(diff_cov[:,:-2]<=dea_cov[:,:-2],axis=1)==8) & (np.sum(diff_cov[:,-2:]>=dea_cov[:,-2:],axis=1)==2)
        return index_bool

    def create_signal_rsi(self,target_info):
        """rsi指标判断"""
        
        rsi5_cov = np.array([item["focus2_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        rsi20_cov = np.array([item["focus3_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即rsi快线向上突破rsi慢线
        index_bool = (np.sum(rsi5_cov[:,:-2]<=rsi20_cov[:,:-2],axis=1)==8) & (np.sum(rsi5_cov[:,-2:]>=rsi20_cov[:,-2:],axis=1)==2)
        return index_bool

    def create_signal_kdj(self,target_info):
        """kdj指标判断"""
        
        k_cov = np.array([item["focus1_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        d_cov = np.array([item["focus2_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        j_cov = np.array([item["focus3_array"][self.input_chunk_length-10:self.input_chunk_length] for item in target_info])
        # 规则为金叉，即k线向上突破j线
        index_bool = (np.sum(k_cov[:,:-2]<=d_cov[:,:-2],axis=1)==8) & (np.sum(d_cov[:,-2:]>=d_cov[:,-2:],axis=1)==2)
        # 突破的时候d线也是向上的
        j_slope = j_cov[:,1:] - j_cov[:,:-1]
        index_bool = index_bool &  (np.sum(j_slope[:,-3:]>0,axis=1)>=3)
        # index_bool = index_bool & (np.sum(j_cov[:,:-2]<=d_cov[:,:-2],axis=1)==8) & (np.sum(j_cov[:,-2:]>=d_cov[:,-2:],axis=1)==2)
        # 还需要满足K和D值小于一定阈值
        index_bool = index_bool & (np.sum(k_cov[:,:-2]<20,axis=1)>=8) & (np.sum(d_cov[:,:-2]<30,axis=1)>=8)
        # 还需要满足J线小于阈值
        # index_bool = index_bool & (np.sum(j_cov[:,:-2]<0,axis=1)>=8) & (np.sum(j_cov[:,-2:]>0,axis=1)>=1)
        return index_bool
                   
    def validation_step(self, val_batch_ori, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        # SANITY CHECKING模式下，不进行处理
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return            
        fake_loss = torch.ones(1).to(self.device)
        # 只关注重点部分
        val_batch = self.filter_batch_by_condition(val_batch_ori,filter_conv_index=self.filter_conv_index)
        if val_batch is None:
            return fake_loss  
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = val_batch 
        data = [past_target.cpu().numpy(),past_covariates.cpu().numpy(), historic_future_covariates.cpu().numpy(),
                         future_covariates.cpu().numpy(),static_covariates.cpu().numpy(),scaler,target_class.cpu().numpy(),target.cpu().numpy(),target_info]
        print("dump valid,batch:{},target shape:{}".format(batch_idx,past_target.shape))
        pickle.dump(data,self.valid_fout) 
        
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
        
        if self.model_type=="tft" or self.model_type=="tide":
            model = _TFTModuleBatch(
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
                model_type=self.model_type,
                train_sample=self.train_sample,
                **self.pl_module_params,
            )   
        if self.model_type=="hsan":
            model = HsanModule(
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
                past_split=self.past_split,
                filter_conv_index=self.filter_conv_index,
                device=self.device,
                batch_file_path=self.batch_file_path,
                model_type=self.model_type,
                train_sample=self.train_sample,
                **self.pl_module_params,
            )               
        if self.model_type=="dink_ts":
            model = DinkModule(
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
                past_split=self.past_split,
                filter_conv_index=self.filter_conv_index,
                device=self.device,
                batch_file_path=self.batch_file_path,
                model_type=self.model_type,
                train_sample=self.train_sample,
                **self.pl_module_params,
            )      
        return model         
               
    def _batch_collate_filter(self,ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        if self.model_type=="hsan":
            return super()._batch_collate_filter(ori_batch)
        
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

               