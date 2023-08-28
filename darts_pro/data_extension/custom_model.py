from darts.models import TFTModel
from darts.models.forecasting.tft_model import _TFTModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import _generate_new_dates
from darts_pro.data_extension.series_data_utils import get_pred_center_value,build_serices_with_ndarray,get_np_center_value
from darts.models.forecasting.torch_forecasting_model import _raise_if_wrong_type
from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
from darts.utils.utils import seq2series, series2seq

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import joblib

from .series_data_utils import StatDataAssis
from cus_utils.process import create_from_cls_and_kwargs
from cus_utils.encoder_cus import StockNormalizer,unverse_transform_slope_value
from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import compute_price_class,compute_price_class_batch,slope_classify_compute,slope_classify_compute_batch
from cus_utils.metrics import compute_cross_metrics,compute_vr_metrics
import cus_utils.global_var as global_var
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC,SLOPE_SHAPE_SMOOTH,CLASS_LAST_VALUE_MAX
from darts_pro.data_extension.custom_tcn_model import LSTMReg

import torchmetrics
from torchmetrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import tsaug
from losses.mtl_loss import UncertaintyLoss,MseLoss
from cus_utils.process import raise_if_not

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
from darts.utils.data import (
    MixedCovariatesInferenceDataset,
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    TrainingDataset,
    InferenceDataset
)
from darts_pro.data_extension.custom_dataset import CustomSequentialDataset,CustomInferenceDataset

logger = get_logger(__name__)

from cus_utils.log_util import AppLogger
app_logger = AppLogger()

viz_result_suc = TensorViz(env="train_result_suc")
viz_result_fail = TensorViz(env="train_result_fail")
viz_result_nor = TensorViz(env="train_result_nor")
viz_input_aug = TensorViz(env="data_train_aug")
viz_target = TensorViz(env="data_target")

hide_target = True
knn_clf = joblib.load("custom/data/asis/knn_clf.model")

def _build_forecast_series(
     points_preds: Union[np.ndarray, Sequence[np.ndarray]],
     input_series: TimeSeries,
     vr_class: np.ndarray
 ) -> TimeSeries:
    """
    Builds a forecast time series starting after the end of an input time series, with the
    correct time index (or after the end of the input series, if specified).
    """
    
    time_index_length = (
        len(points_preds)
        if isinstance(points_preds, np.ndarray)
        else len(points_preds[0])
    )
    time_index = _generate_new_dates(time_index_length, input_series=input_series)
    values = (
        points_preds
        if isinstance(points_preds, np.ndarray)
        else np.stack(points_preds, axis=2)
    )
    # 分类信息求平均，并返回
    if vr_class is None:
        vr_mean_class = None
    else:
        vr_mean_class = torch.mean(vr_class,dim=0)
    return (TimeSeries.from_times_and_values(
        time_index,
        values,
        freq=input_series.freq_str,
        columns=input_series.columns,
        static_covariates=input_series.static_covariates,
        hierarchy=input_series.hierarchy,
    ),None,vr_mean_class)
 
class ClasssifyLayer(nn.Module):  
    
    def __init__(self, layer_num,input_dim, output_dim,device=None):
        """使用全连接进行分类数值输出
          Params
            layer_num： 层数
            input_dim： 序列长度
            output_dim： 类别数
        """
        super().__init__()
         
        layers = []
        layer_mul_params = []
        self.device = device
        # 每层分别进行全连接
        for i in range(layer_num):
            layers.append(nn.Linear(input_dim, output_dim).to(self.device))
            layer_mul_params.append(torch.rand(1, requires_grad=True).to(self.device))
        # 统一把各层进行串接
        self.combine_layer = nn.Sequential(*layers)
        self.layer_mul_params = layer_mul_params
        
    def forward(self, input):
        
        combine_ouput = 0
        for i in range(len(input)):
            input_item = input[i][:,:,0,0]
            line_layer = self.combine_layer[i]
            layer_mul_param = self.layer_mul_params[i]
            output_item = line_layer(input_item)
            # 把各层的全连接数据进行统一的线性变换(相加)
            if i==0:
                combine_ouput = layer_mul_param * output_item
            else:
                combine_ouput = combine_ouput + layer_mul_param * output_item
                
        return combine_ouput
                
class _TFTCusModule(PLMixedCovariatesModule):
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
        device="cpu",
        **kwargs,
    ):
        # 模拟初始化，实际未使用
        super().__init__(**kwargs)
        # super().__init__(output_dim,{},num_static_components,hidden_size,lstm_layers,num_attention_heads,
        #                             full_attention,feed_forward,hidden_continuous_size,
        #                             categorical_embedding_sizes,dropout,add_relative_index,norm_type,**kwargs)
        #

        self.past_split = past_split
        self.filter_conv_index = filter_conv_index
        self.variables_meta_array = variables_meta_array
        model_list = []
        for i in range(len(past_split)):
            # 拆分过去协变量,形成不同的网络配置，给到不同的model
            model =  _TFTModule(output_dim,variables_meta_array[i],num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,categorical_embedding_sizes,
                                    dropout,add_relative_index,norm_type,**kwargs)
            pred_len = kwargs["output_chunk_length"]
            # mse损失计算                
            model.mean_squared_error = MeanSquaredError().to(device)
            model_list.append(model)
        self.sub_models = nn.ModuleList(model_list) 
        # 涨跌幅度分类
        vr_range_num = len(CLASS_SIMPLE_VALUES.keys())       
        # 序列分类层，包括目标分类和输出分类
        self.classify_vr_layer = self._construct_classify_layer(len(past_split),vr_range_num,device=device)        
        self.classify_tar_layer = self._construct_classify_layer(len(past_split),vr_range_num,device=device)  
        if use_weighted_loss_func and not isinstance(model.criterion,UncertaintyLoss):
            self.criterion = UncertaintyLoss(device=device) 
                    
        self.val_results = {}
        # 辅助数据功能
        self.data_assis = StatDataAssis()
        # 优化器执行频次
        self.lr_freq = {"interval":"epoch","frequency":1}

    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        future_target,
        optimizer_idx=-1
    ) -> torch.Tensor:
        """重载训练方法，加入分类模式"""
        out_total= []
        batch_size = x_in[1].shape[0]
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx==-1:
                # 根据配置，不同的模型使用不同的过去协变量
                past_convs_item = x_in[0][i]
                x_in_item = (past_convs_item,x_in[1],x_in[2])
                out = m(x_in_item)
            else:
                # 模拟数据
                out = torch.ones([batch_size,self.output_chunk_length,1,1]).to(self.device)
            out_total.append(out)    
        # 如果只有一个目标，则输出端模拟第二个用于数量对齐
        # if len(out_total)==1:
        #     fake_out = torch.ones(out_total[0].shape).to(self.device)
        #     out_total.append(fake_out)
        
        out_for_class = torch.cat(out_total,dim=2)[:,:,:,0] 
        #  根据预测数据进行分类
        vr_class = self.classify_tar_layer(out_for_class)
        tar_class = self.classify_tar_layer(future_target)
        return out_total,vr_class,tar_class
 
    def _construct_classify_layer(self, input_dim, output_dim,layer_dim=3,hidden_dim=64,device=None):
        """使用全连接进行分类数值输出
          Params
            layer_num： 层数
            input_dim： 序列长度
            output_dim： 类别数
        """
        
        class_layer = LSTMReg(input_dim, hidden_dim, layer_dim, output_dim)
        class_layer = class_layer.cuda(device)
        return class_layer

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
        dim_variable = 2

        # 生成多组过去协变量，用于不同子模型匹配
        x_past_array = []
        for i,p_index in enumerate(self.past_split):
            past_conv_index = self.past_split[i]
            past_covariates_item = past_covariates[:,:,past_conv_index[0]:past_conv_index[1]]
            # 修改协变量生成模式，只取自相关目标作为协变量
            if hide_target:
                conv_defs = [
                            past_target[:,:,i:i+1],
                            past_covariates_item,
                            historic_future_covariates,
                    ]
            else:
                conv_defs = [
                            past_target,
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
               
    def training_step(self, train_batch, batch_idx, optimizer_idx) -> torch.Tensor:
        """performs the training step"""
        
        train_batch = self.filter_batch_by_condition(train_batch,filter_conv_index=self.filter_conv_index)
        loss,detail_loss = self.training_step_real(train_batch, batch_idx, optimizer_idx) 
        return loss
    
    def training_step_real(self, train_batch, batch_idx, optimizer_idx) -> torch.Tensor:
        """包括第一及第二部分数值数据,以及分类数据"""
        
        input_batch = self._process_input_batch(train_batch[:5])
        # 收集目标数据用于分类
        future_target = train_batch[-2]
        (output,vr_class,tar_class) = self(input_batch,future_target,optimizer_idx=optimizer_idx)
        # 目标数据里包含分类信息
        scaler,target_class,target,target_info = train_batch[5:]
        target_class = target_class[:,:,0]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained
        
        loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (target,target_class,scaler,target_info),optimizers_idx=optimizer_idx)
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("lr",self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("m1_lr",self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("m2_lr",self.trainer.optimizers[1].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("m3_lr",self.trainer.optimizers[2].param_groups[0]["lr"])
        self.loss_data.append(detail_loss)
        return loss,detail_loss

    def on_train_epoch_start(self):
        self.loss_data = []
           
    def on_train_epoch_end(self):
        
        mse_loss = torch.stack([item[0] for item in self.loss_data]).sum()
        value_diff_loss = torch.stack([item[1] for item in self.loss_data]).sum()
        corr_loss = torch.stack([item[2] for item in self.loss_data]).sum()
        ce_loss = torch.stack([item[3] for item in self.loss_data]).sum()
        self.log("train_mse_loss", mse_loss,  prog_bar=True)
        # self.custom_histogram_adder(batch_idx)
        # self.log("train_value_diff_loss", value_diff_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("train_corr_loss", corr_loss, prog_bar=True)
        self.log("train_mse_loss", ce_loss, prog_bar=True)
        self.log("train_ce_loss", ce_loss, prog_bar=True)
        
    def on_validation_epoch_start(self):
        self.import_price_result = None
        self.total_imp_cnt = 0
        
    def on_validation_epoch_end(self):
        # SANITY CHECKING模式下，不进行处理
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return    
        if self.import_price_result is not None:
            res_group = self.import_price_result.groupby("result")
            ins_unique = res_group.nunique()
            app_logger.debug("ins_unique:{}".format(ins_unique))     
            total_cnt = ins_unique.values[:,1].sum()
            for i in range(4):
                cnt_values = ins_unique[ins_unique.index==i].values
                if cnt_values.shape[0]==0:
                    cnt = 0
                else:
                    cnt = cnt_values[0,1]
                rate = cnt/total_cnt
                print("cnt:{} with score:{},total_cnt:{},rate:{}".format(cnt,i,total_cnt,rate))
                self.log("score_{} rate".format(i), rate, prog_bar=True) 
            self.log("total cnt", total_cnt, prog_bar=True)  
        self.log("total_imp_cnt", self.total_imp_cnt, prog_bar=True)  
        # # 动态冻结网络参数
        # corr_loss = self.trainer.callback_metrics["val_corr_loss"]
        # mse_loss = self.trainer.callback_metrics["val_mse_loss"]
        # if self.current_epoch>0:
        #     if mse_loss>0.68:
        #         self.freeze_apply(mode=0)
        #     elif corr_loss>0.65:
        #         self.freeze_apply(mode=1)
        #     else:
        #         # 数值loss都达到指标后，打开分类层
        #         self.freeze_apply(mode=2)
        # else:
        #     self.freeze_apply(mode=0)

    def validation_step(self, val_batch_ori, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        # 只关注重点部分
        val_batch = self.filter_batch_by_condition(val_batch_ori,filter_conv_index=self.filter_conv_index)
        return self.validation_step_real(val_batch, batch_idx)
                                 
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""

        input_batch = self._process_input_batch(val_batch[:5])
        # 收集目标数据用于分类
        future_target = val_batch[-2]
        (output,vr_class,tar_class) = self(input_batch,future_target,optimizer_idx=-1)
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        scaler,target_class,target,target_info = val_batch[5:]  
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        last_target_vr_class = target_class[:,1]
        output_combine = [output_item[:,:,0,0] for output_item in output]
        output_combine = torch.stack(output_combine,dim=2).cpu().numpy()
        output_inverse = self.get_inverse_data(output_combine,target_info=target_info,scaler=scaler)
        past_target_inverse = self.get_inverse_data(past_target.cpu().numpy(),target_info=target_info,scaler=scaler)
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (target,target_class,scaler,target_info),optimizers_idx=-1)
        (mse_loss,value_diff_loss,corr_loss,ce_loss,mean_threhold) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_corr_loss", corr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_mse_loss", mse_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("last_vr_loss", last_vr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        # last_batch_index,last_batch_imp_index,item_codes = self.build_last_batch_index(vr_class_certainlys,target_vr_class,target_info=target_info)
        # 涨跌幅度类别的准确率
        # vr_acc,import_vr_acc,import_recall,import_price_acc,import_price_nag,price_class,sec_acc, \
        #     sec_recall,sec_price_acc,sec_price_nag,import_index,sec_index,last_section_acc,import_last_section_acc = self.compute_vr_class_acc(
        #     vr_class_certainlys, target_vr_class,target_info=target_info,last_vr_class_certainlys=last_vr_class_certainlys,
        #     last_target_vr_class=last_target_vr_class,output_inverse=torch.Tensor(output_inverse).to(self.device))  
        import_vr_acc,import_recall,import_price_acc,import_price_nag,price_class,import_price_result, \
                class_acc,imp_class_acc_cnt,imp_class_acc,output_class_acc,output_imp_class_acc_cnt,output_imp_class_acc \
             = self.compute_real_class_acc(output_inverse=output_inverse,vr_class=vr_class.cpu().numpy(),tar_class=tar_class.cpu().numpy(),
                                           target_vr_class=target_vr_class,target_info=target_info)   
        total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        if self.total_imp_cnt==0:
            self.total_imp_cnt = total_imp_cnt
        else:
            self.total_imp_cnt += total_imp_cnt
                
        self.log("import_vr_acc", import_vr_acc, batch_size=val_batch[0].shape[0], prog_bar=True)    
        self.log("import_recall", import_recall, batch_size=val_batch[0].shape[0], prog_bar=True)   
        # self.log("last_section_slop_acc", last_sec_acc, batch_size=val_batch[0].shape[0], prog_bar=True)  
        self.log("import_price_acc", import_price_acc, batch_size=val_batch[0].shape[0], prog_bar=True)       
        self.log("import_price_nag", import_price_nag, batch_size=val_batch[0].shape[0], prog_bar=True)   
        # self.log("class_acc", class_acc, batch_size=val_batch[0].shape[0], prog_bar=True)   
        # self.log("imp_class_acc_cnt", imp_class_acc_cnt, batch_size=val_batch[0].shape[0], prog_bar=True)  
        # self.log("imp_class_acc", imp_class_acc, batch_size=val_batch[0].shape[0], prog_bar=True)   
        self.log("output_imp_class_acc_cnt", output_imp_class_acc_cnt, batch_size=val_batch[0].shape[0], prog_bar=True)   
        self.log("output_imp_class_acc", output_imp_class_acc, batch_size=val_batch[0].shape[0], prog_bar=True)         
        # self.log("instrument_acc", instrument_acc, batch_size=val_batch[0].shape[0], prog_bar=True)       
        # self.log("instrument_nag", instrument_nag, batch_size=val_batch[0].shape[0], prog_bar=True)          
        # self.log("last_section_acc", last_section_acc, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("pos_acc", pos_acc, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("pos_recall", pos_recall, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("neg_acc", neg_acc, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("neg_recall", neg_recall, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("out_last_fall_acc", out_last_fall_acc, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # self.log("tar_last_raise_rate", tar_last_raise_rate, batch_size=val_batch[0].shape[0], prog_bar=True) 
        # print("import_vr_acc:{},import_recall:{},import_price_acc:{},import_price_nag:{},count:{}".
        #     format(import_vr_acc,import_recall,import_price_acc,import_price_nag,import_index.shape[0]))
        past_target = val_batch[0]
        # self.val_metric_show(output,target,price_class,target_vr_class,output_inverse=output_inverse,vr_class=vr_class,
        #                      target_info=target_info,import_price_result=import_price_result,past_covariate=past_covariate,
        #                     last_target_vr_class=last_target_vr_class,batch_idx=batch_idx)
        
        # 累加结果集，后续统计   
        if self.import_price_result is None:
            self.import_price_result = import_price_result    
        else:
            if import_price_result is not None:
                import_price_result_array = import_price_result.values
                # 修改编号，避免重复
                import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*1000
                import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
                self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)        
        # self._calculate_metrics(output, target, self.val_metrics)
            
        return loss
 
    def filter_batch_by_condition(self,val_batch,filter_conv_index=0,rev_threhold=3,recent_threhold=3):
        """按照已知指标，对结果集的重点关注部分进行初步筛选"""
        
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = val_batch    
        
        recent_data = np.array([item["label_array"][:self.input_chunk_length] for item in target_info])
        # 与近期高点比较，不能差太多
        recent_max = np.max(recent_data,axis=1)
        import_index_bool = (recent_max-recent_data[:,-1])/recent_max<(recent_threhold/100)
        # 当前价格不能处于下降趋势
        # less_ret = np.subtract(recent_data.transpose(1,0),recent_data[:,-1]).transpose(1,0)
        # import_index_bool = import_index_bool & (np.sum(less_ret>0,axis=1)<=3)
        
        # 通过指标筛选(配置参数里指定哪种指标)
        rev_cov = np.array([item["kdj_array"][:self.input_chunk_length] for item in target_info])
        rev_cov = np.array([item["macd_array"][:self.input_chunk_length] for item in target_info])
        rev_cov_max = np.max(rev_cov,axis=1)
        rev_cov_recent = rev_cov[:,-5:]
        # 近期均值大于阈值
        rev_boole = np.mean(rev_cov_recent,axis=1)>rev_threhold
        # 最近数值处于比较高的点
        rev_boole = rev_boole & ((rev_cov_max-rev_cov_recent[:,-1])<=(recent_threhold/100))
        # print("total size:{},import_index_bool size:{},rev_boole size:{}".format(import_index_bool.shape[0],np.sum(import_index_bool),np.sum(rev_boole)))
        import_index_bool = import_index_bool & rev_boole        
        rtn_index = np.where(import_index_bool)[0]
        
        val_batch_filter = [past_target[rtn_index,:,:],past_covariates[rtn_index,:,:],historic_future_covariates[rtn_index,:,:],
                            future_covariates[rtn_index,:,:],static_covariates[rtn_index,:,:],
                            np.array(scaler)[rtn_index].tolist(),target_class[rtn_index,:,:],
                            target[rtn_index,:,:],np.array(target_info)[rtn_index].tolist()]
        return val_batch_filter
    
    def build_vr_class_cer(self,vr_class):
        vr_class_cer = F.softmax(vr_class,dim=-1)
        vr_class_cer = torch.max(vr_class_cer,dim=-1)[1]    
        return vr_class_cer  
        
    def process_val_results(self,results,epoch):
        """按照批次，累加训练结果"""
        
        if epoch not in self.val_results:
            self.val_results[epoch] = results
            self.val_results[epoch]["time"] = 1 
        else:
            self.val_results[epoch]["val_loss"] += results["val_loss"]
            self.val_results[epoch]["val_corr_loss"] += results["val_corr_loss"]
            self.val_results[epoch]["ce_loss"] += results["ce_loss"]
            self.val_results[epoch]["value_diff_loss"] += results["value_diff_loss"]
            self.val_results[epoch]["vr_acc"] += results["vr_acc"]
            self.val_results[epoch]["import_vr_acc"] += results["import_vr_acc"]
            self.val_results[epoch]["time"] += 1         
    
    def compute_slope_acc(self,last_vr_class_certainlys,target_info=None,output_slope=None,target_slope=None):
        """统计反转趋势准确率"""
        
        # 目标上涨幅度计算
        label_array = torch.Tensor([item["label_array"] for item in target_info]).to(self.device)
        target = label_array[:,self.input_chunk_length:]
        last_raise_range = torch.Tensor([ts["last_raise_range"] for ts in target_info])
        last_output_slope = output_slope[:,-1]
        last_target_slope = target_slope[:,-1]
        # 上升趋势统计
        pos_index_bool = (last_output_slope>0.3)
        pos_index = torch.where(pos_index_bool)[0]
        pos_acc_cnt = torch.sum(last_target_slope[pos_index]>0)
        pos_acc = pos_acc_cnt/pos_index.shape[0]
        pos_recall = pos_acc_cnt/torch.sum(last_target_slope>0)

        # 返回准确率，以及上涨下跌占比
        return pos_acc,pos_recall
           
    def get_inverse_data(self,output,target_info=None,single_way=False,scaler=None):
        dataset = global_var.get_value("dataset")
        
        if single_way:
            group_code = target_info["item_rank_code"]
            if dataset.transform_inner:
                scaler_sample = scaler
                pred_inverse = scaler_sample.inverse_transform(output)  
            else:
                scaler_sample = dataset.get_scaler_by_group_code(group_code)  
                pred_inverse = scaler_sample._fitted_params[0].inverse_transform(output) 
            return pred_inverse                          
             
        pred_inverse = []
        for index in range(output.shape[0]):
            output_item = output[index]
            group_code = target_info[index]["item_rank_code"]
            if dataset.transform_inner:
                scaler_sample = scaler[index]
                pred_center_data = scaler_sample.inverse_transform(output_item)   
            else:
                scaler_sample = dataset.get_scaler_by_group_code(group_code)
                pred_center_data = scaler_sample._fitted_params[0].inverse_transform(output_item)     
            pred_inverse.append(pred_center_data)
        return np.stack(pred_inverse)
        
    def compute_real_class_acc(self,label_threhold=0.5,target_info=None,output_inverse=None,vr_class=None,tar_class=None,target_vr_class=None):
        """计算涨跌幅分类准确度"""
        
        target_data = np.array([item["label_array"][self.input_chunk_length-1:] for item in target_info])
        price_data = np.array([item["price_array"][self.input_chunk_length-1:] for item in target_info])
        
        output_label_inverse = output_inverse[:,:,0] 
        # output_second_inverse = output_inverse[:,:,1]
        output_second_inverse = output_inverse[:,:,1]
        # output_third_inverse = output_inverse[:,:,2]
                 
        # 整体需要有上涨幅度
        slope_out_compute = (output_label_inverse[:,-1]  - output_label_inverse[:,0])/np.abs(output_label_inverse[:,0])*100
        output_import_index_bool = slope_out_compute>label_threhold
        # 最后一段上涨
        output_import_index_bool = output_import_index_bool & ((output_label_inverse[:,-1] - output_label_inverse[:,-2])>0)
        # 辅助指标判断
        second_index_bool = (output_second_inverse[:,-1] - output_second_inverse[:,0]) > 0.5
        # 整体上升幅度与振幅差值较小
        second_max = np.max(output_second_inverse,axis=-1)
        second_min = np.min(output_second_inverse,axis=-1)        
        second_index_bool = second_index_bool & (((output_second_inverse[:,-1] - output_second_inverse[:,0])/(second_max - second_min))>0.5)
        # 最后一个值大于0
        second_index_bool = second_index_bool & (output_second_inverse[:,-1]>0)
        # 最后一段上升
        second_index_bool = second_index_bool & (output_second_inverse[:,-1]>output_second_inverse[:,-2])        
        # 直接使用分类
        # third_index_bool = (third_class==CLASS_SIMPLE_VALUE_MAX)
        # 使用knn分类模式判别
        # print("begin knn_clf")
        # predicted_labels = knn_clf.predict(output_inverse)
        # print("begin end")
        # import_index_bool = predicted_labels==3
        # 综合判别
        import_index_bool = output_import_index_bool
        
        # 可信度检验，预测值不能偏离太多
        # import_index_bool = import_index_bool & ((output_inverse[:,0] - recent_data[:,-1])/recent_data[:,-1]<0.1)
        import_index = np.where(import_index_bool)[0]

        # 使用分类判断
        # import_index = (vr_class==CLASS_SIMPLE_VALUE_MAX)
        
        # 重点类别的准确率
        import_acc_count = np.sum(target_vr_class[import_index]==CLASS_SIMPLE_VALUE_MAX)
        import_price_count = np.sum(target_vr_class[import_index]==CLASS_SIMPLE_VALUE_MAX)
        if import_index.shape[0]==0:
            import_acc = torch.tensor(0.0)
        else:
            import_acc = import_acc_count/import_index.shape[0]
         
        # 重点类别的召回率    
        total_imp_cnt = np.sum(target_vr_class==CLASS_SIMPLE_VALUE_MAX)
        if total_imp_cnt!=0:
            import_recall = import_acc_count/total_imp_cnt
        else:
            import_recall = 0

        # 重点类别的价格准确率
        price_class = []
        import_price_result = []
        for i,imp_idx in enumerate(import_index):
            target_info_item = target_info[imp_idx]
            price_array = target_info_item["price_array"][self.input_chunk_length:]
            p_taraget_class = compute_price_class(price_array,mode="first_last")
            import_price_result.append([imp_idx,target_info_item["item_rank_code"],p_taraget_class])
        import_price_result = np.array(import_price_result)        
        price_class = np.array(price_class)
        if import_index.shape[0]==0:
            total_instrument_count = 0
            import_price_acc = 0
            import_price_nag = 0
            instrument_acc = 0
            instrument_nag = 0
        else:
            import_price_acc_index = np.where(import_price_result[:,2]==CLASS_SIMPLE_VALUE_MAX)[0]
            import_price_nag_index = np.where(import_price_result[:,2]==0)[0]
            import_price_acc_array = import_price_result[import_price_acc_index]
            import_price_nag_array = import_price_result[import_price_nag_index]
            import_price_acc_count = import_price_acc_array.shape[0]
            import_price_nag_count = import_price_nag_array.shape[0]
            # 以股票为目标计算准确率
            if import_price_acc_count>0:
                instrument_acc_count = np.unique(import_price_acc_array[:,1]).shape[0]
            else:
                instrument_acc_count = 0
            if import_price_nag_count>0:
                instrument_nag_count = np.unique(import_price_nag_array[:,1]).shape[0]
            else:
                instrument_nag_count = 0              
            total_instrument_count = np.unique(import_price_result[:,1]).shape[0]
            import_price_acc = import_price_acc_count/import_index.shape[0]
            import_price_nag = import_price_nag_count/import_index.shape[0]
            instrument_acc = instrument_acc_count/total_instrument_count
            instrument_nag = instrument_nag_count/total_instrument_count         
        
        if import_price_result.shape[0]==0:
            import_price_result = None
        else:
            import_price_result = pd.DataFrame(import_price_result,columns=["imp_index","instrument","result"])     
        
        # 分类准确率计算
        class_acc,imp_class_acc_cnt,imp_class_acc = self.classify_compute(tar_class, target_vr_class)
        
        output_class_acc,output_imp_class_acc_cnt,output_imp_class_acc = self.classify_compute(vr_class, target_vr_class)
        
        return import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_result, \
                class_acc,imp_class_acc_cnt,imp_class_acc,output_class_acc,output_imp_class_acc_cnt,output_imp_class_acc
    
    def classify_compute(self,pred_class,target_class):
        pred_class = np.argmax(pred_class,axis=-1)
        class_acc = np.sum(pred_class==target_class)/target_class.shape[0]
        imp_class_index = np.where(pred_class==CLASS_SIMPLE_VALUE_MAX)[0]
        if imp_class_index.shape[0]==0:
            return 0,0,0
        imp_class_acc_cnt = np.sum(target_class[imp_class_index]==CLASS_SIMPLE_VALUE_MAX)
        imp_class_acc = imp_class_acc_cnt/imp_class_index.shape[0]
        return class_acc,imp_class_acc_cnt,imp_class_acc
            
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""
        
        (output_value,vr_class,tar_class) = output
        output_value = [output_value_item.squeeze(dim=-1) for output_value_item in output_value]
        (target_real,target_class,scaler,target_info) = target
        future_target_list = []
        target_slope_list = []
        for t in target_info:
            future_target_list.append(t["future_target"])
            target_slope_list.append(t["last_raise_range"])                       
        future_target = torch.Tensor(future_target_list).to(self.device)
        slope_target = torch.Tensor(target_slope_list).to(self.device)
        output_combine = (output_value,vr_class,tar_class)
        
        if self.likelihood:
            # 把似然估计损失叠加到自定义多任务损失里
            loss_like = self.likelihood.compute_loss(output_value, target_real)
            if self.criterion is None:
                return loss_like
            mtl_loss = self.criterion(output_combine, target_real,outer_loss=loss_like)
            return mtl_loss
        else:
            return self.criterion(output_combine, (target_real,target_class,target_info),optimizers_idx=optimizers_idx)

    def custom_histogram_adder(self,batch_idx):
        # iterating through all parameters
        for name,params in self.named_parameters():
            global_step = self.current_epoch*1000 + batch_idx
            global_step = batch_idx
            if params is not None:
                self.logger.experiment.add_histogram(name + "_weights",params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram(name + "_grad",params.grad,global_step)
        
    def configure_optimizers(self):
        optimizers = self.build_dynamic_optimizers()
        
        # 对应优化器，生成多个学习率
        lr_schedulers = []
        for i in range(len(self.past_split)):
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizers[i]
            lr_monitor = lr_sched_kws.pop("monitor", None)
            # # 分类层增加学习率
            # if i==len(self.past_split):
            #     lr_sched_kws["base_lr"] = lr_sched_kws["base_lr"] * 10
            #     lr_sched_kws["max_lr"] = lr_sched_kws["max_lr"] * 10
                
            lr_scheduler = create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": self.lr_freq["interval"],
                "frequency": self.lr_freq["frequency"],              
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            } 
            lr_schedulers.append(lr_scheduler_config)  
        lr_schedulers.append(lr_scheduler_config)  
        return optimizers, lr_schedulers       

    def build_dynamic_optimizers(self):
        """生成多优化器配置"""

        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        optimizers = []
        lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
        base_lr = lr_sched_kws["base_lr"]
        # 针对不同子模型，分别生成优化器
        for i in range(len(self.past_split)):
            avalabel_params = list(map(id, self.sub_models[i].parameters()))
            base_params = filter(lambda p: id(p) in avalabel_params, self.parameters())
            optimizer_kws["params"] = [
                        {'params': base_params},
                        # {'params': self.slope_layer.parameters(), 'lr': base_lr*10},
                        # {'params': self.classify_vr_layer.parameters(), 'lr': base_lr*10}
                        ]
            optimizer = create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)
            optimizers.append(optimizer)
        # 单独定义分类损失优化器   
        optimizer_kws["params"] = [
                    {'params': self.classify_tar_layer.parameters(), 'lr': base_lr*3},
                    # {'params': self.slope_layer.parameters(), 'lr': base_lr*10},
                    # {'params': self.classify_vr_layer.parameters(), 'lr': base_lr*10}
                    ]
        optimizer = create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)
        optimizers.append(optimizer)        
        return optimizers

    def freeze_apply(self,mode=0):
        """ 动态冻结指定参数
           Params:
              mode 冻结参数配置， 0 冻结第一部分 1 冻结第二部分,打开第一部分  2-冻结一二部分，使用分类层
        """
        
        # 如果小于当前模式，则不处理
        if self.freeze_mode>=mode:
            return
        
        print('do mode:{},self.freeze_mode:{}'.format(mode,self.freeze_mode))
        # 修改保存此超参数
        self.freeze_mode = mode
        if mode==0:
            self.criterion.loss_mode = 0    
            # 冻结第一部分模型参数
            for param in self.sub_models[0].parameters():
                param.requires_grad = False    
            self.criterion.loss_mode = 0             
        if mode==1:
            # 冻结第二部分,打开第一部分
            for param in self.sub_models[0].parameters():
                param.requires_grad = True   
            for param in self.sub_models[1].parameters():
                param.requires_grad = False           
            self.criterion.loss_mode = 1
        if mode==2:
            # 冻结一二部分，使用分类层
            for param in self.sub_models[0].parameters():
                param.requires_grad = False   
            for param in self.sub_models[1].parameters():
                param.requires_grad = False              
            self.criterion.loss_mode = 2
                        
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Sequence[TimeSeries]:
        """重载原方法，服务于自定义模式"""
        
        input_data_tuple,scalers, batch_input_series = batch[:-2], batch[-2], batch[-1]
        input_data_tuple = [input_data_tuple[0],input_data_tuple[1],input_data_tuple[2],input_data_tuple[3],input_data_tuple[5]]
        input_past, input_future, input_static = self._process_input_batch(input_data_tuple)
        out_combine = self._produce_predict_output(x=(input_past, input_future, input_static))   
        (output,pred_class,vr_class) = out_combine
        output = output.cpu().numpy()
        # 分别遍历每个系列，进行反向归一化
        batch_predictions_unscale = []
        for i in range(output.shape[0]):
            scaler = scalers[i]
            batch_prediction = output[i]
            batch_prediction_unscale = scaler.inverse_transform(batch_prediction)
            batch_predictions_unscale.append(batch_prediction_unscale)
            
        ts_forecasts = Parallel(n_jobs=self.pred_n_jobs)(
            delayed(_build_forecast_series)(
                [batch_predictions_unscale[batch_idx]],
                input_series,
                vr_class[batch_idx,:,:]
            )
            for batch_idx, input_series in enumerate(batch_input_series)
        )
        return ts_forecasts

    def _get_batch_prediction(
        self, n: int, input_batch: Tuple, roll_size: int
    ) -> torch.Tensor:
        """重载父类方法，以处理定制数据"""

        dim_component = 2
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            future_past_covariates,
            static_covariates,
        ) = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = (
            past_covariates.shape[dim_component] if past_covariates is not None else 0
        )
        n_future_covs = (
            future_covariates.shape[dim_component]
            if future_covariates is not None
            else 0
        )

        input_past, input_future, input_static = self._process_input_batch(
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates[:, :roll_size, :]
                if future_covariates is not None
                else None,
                static_covariates,
            )
        )
        
        # 自定义模型返回的是tuple，包括数值输出，以及分类输出
        out_combine = self._produce_predict_output(x=(input_past, input_future, input_static))
        out = out_combine[0][
            :, self.first_prediction_index :, :
        ]
        out_class = out_combine[1]
        vr_class = out_combine[2]
        
        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = (
                    prediction_length + self.output_chunk_length - n
                )
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]

            # ==========> PAST INPUT <==========
            # roll over input series to contain the latest target and covariates
            input_past = torch.roll(input_past, -roll_size, 1)

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length :, :]

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = (
                    prediction_length - self.input_chunk_length,
                    prediction_length,
                )

            # update past covariates to include next `roll_size` future past covariates elements
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[
                    :, -roll_size:, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[
                    :, :, n_targets : n_targets + n_past_covs
                ] = future_past_covariates[:, left_past:right_past, :]

            # update historic future covariates to include next `roll_size` future covariates elements
            if n_future_covs and self.input_chunk_length >= roll_size:
                input_past[
                    :, -roll_size:, n_targets + n_past_covs :
                ] = future_covariates[:, left_past:right_past, :]
            elif n_future_covs:
                input_past[:, :, n_targets + n_past_covs :] = future_covariates[
                    :, left_past:right_past, :
                ]

            # ==========> FUTURE INPUT <==========
            left_future, right_future = (
                right_past,
                right_past + self.output_chunk_length,
            )
            # update future covariates to include next `roll_size` future covariates elements
            if n_future_covs:
                input_future = future_covariates[:, left_future:right_future, :]

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(
                x=(input_past, input_future, input_static)
            )[:, self.first_prediction_index :, :]

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction,out_class,vr_class
    
    def _produce_predict_output(self, x: Tuple):
        """定制修改了返回值模式，在此需要重载原方法处理数据"""
        
        if self.likelihood:
            output = self(x)
            return self.likelihood.sample(output)
        else:
            output = self(x)
            output = (output[0].squeeze(dim=-1),output[1],output[2])
            return output
            
    def predict_data_eval(self,batch_predictions,batch_input_series):
        """保持scaler方式评估预测数据"""
        
        for batch_idx, input_series in enumerate(batch_input_series):
            for batch_prediction in batch_predictions:
                pred = batch_prediction[batch_idx]
    
    def build_last_batch_index(self,vr_class,target_vr_class,target_info=None):
        size = target_vr_class.shape[0]
        last_batch_index = []
        last_batch_imp_index = []
        last_batch_imp_correct_index = []
        item_codes = []
        # 取得最后一个批次索引
        for s_index in range(size):
            ts = target_info[s_index]
            # 只针对最后一个时间批次进行显示
            if ts["end"]==ts["total_end"]:
                item_codes.append(ts["item_rank_code"])
                last_batch_index.append(s_index) 
                if vr_class[s_index]==CLASS_SIMPLE_VALUE_MAX:
                    last_batch_imp_index.append(s_index)
                    if target_vr_class[s_index]==CLASS_SIMPLE_VALUE_MAX:
                        last_batch_imp_correct_index.append(s_index)
                        
        return last_batch_index,last_batch_imp_index,item_codes
                                
    def val_metric_show(self,output,target,price_class,target_vr_class,output_inverse=None,
                        vr_class=None,past_covariate=None,target_info=None,import_price_result=None,last_target_vr_class=None,batch_idx=0):
        
        if import_price_result is None or import_price_result.shape[0]==0:
            return
        if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
            return 
        
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        names = ["pred","label","price","obv_output","obv_tar","cci_output","cci_tar"]        
        names = ["pred","label","price","obv_output","obv_tar"]          
        result = []
        
        # viz_result_suc.remove_env()
        # viz_result_nor.remove_env()
        # viz_result_fail.remove_env()
              
        res_group = import_price_result.groupby("result")
        target_imp_index = np.where(target_vr_class==3)[0]
        # if target_imp_index.shape[0]>0:
        #     for i in range(15):
        #         rand_index = np.random.randint(0,target_imp_index.shape[0]-1)
        #         s_index = target_imp_index[rand_index]
        #         ts = target_info[s_index]
        #         pred_data = output_inverse[s_index]
        #         pred_center_data = pred_data[:,0]
        #         pred_second_data = pred_data[:,1]            
        #         df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
        #                                 (df_all["instrument_rank"]==ts["item_rank_code"])]   
        #         win = "win_target_{}".format(batch_idx,i)
        #         self.draw_row(pred_center_data, pred_second_data, df_target=df_target, ts=ts, names=names,viz=viz_target,win=win)
                            
        for result,group in res_group:
            r_index = -1
            unique_group = group.drop_duplicates(subset=['instrument'], keep='first')
            for index, row in unique_group.iterrows():
                r_index += 1
                s_index = row["imp_index"]
                ts = target_info[s_index]
                pred_data = output_inverse[s_index]
                pred_center_data = pred_data[:,0]
                pred_second_data = pred_data[:,1]
                # pred_third_data = pred_data[:,2]
                # 可以从全局变量中，通过索引获得实际价格
                df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
                                        (df_all["instrument_rank"]==ts["item_rank_code"])]            
                win = "win_{}_{}".format(batch_idx,r_index)
                if r_index>15:
                    break
                if result==CLASS_SIMPLE_VALUE_MAX:                 
                    viz = viz_result_suc
                elif result==0:                 
                    viz = viz_result_fail   
                else:
                    viz = viz_result_nor  
                self.draw_row(pred_center_data, pred_second_data, df_target=df_target, ts=ts, names=names,viz=viz,win=win)
                      
    def draw_row(self,pred_center_data,pred_second_data,df_target=None,ts=None,names=None,viz=None,win="win"):
        """draw one line"""
        
        # 补充画出前面的label数据
        target_combine_sample = df_target["label"].values
        pad_data = np.array([0 for i in range(self.input_chunk_length)])
        pred_center_data = np.concatenate((pad_data,pred_center_data),axis=-1)
        pred_second_data = np.concatenate((pad_data,pred_second_data),axis=-1)
        # pred_third_data = np.concatenate((pad_data,pred_third_data),axis=-1)
        view_data = np.stack((pred_center_data,target_combine_sample),axis=0).transpose(1,0)
        price_target = df_target["label_ori"].values    
        price_target = np.expand_dims(price_target,axis=0)    
        second_target = df_target["CCI5"].values 
        second_target = np.expand_dims(second_target,axis=-1)  
        # third_target = df_target["KDJ_J"].values 
        # third_target = np.expand_dims(third_target,axis=-1)              
        view_data = np.concatenate((view_data,price_target.transpose(1,0)),axis=1) 
        view_data = np.concatenate((view_data,np.expand_dims(pred_second_data,axis=-1)),axis=1)    
        view_data = np.concatenate((view_data,second_target),axis=1)  
        # view_data = np.concatenate((view_data,np.expand_dims(pred_third_data,axis=-1)),axis=1)    
        # view_data = np.concatenate((view_data,third_target),axis=1)  
        # view_data = view_data[:,3:]
        x_range = np.array([i for i in range(ts["start"],ts["end"])])
        if self.monitor is not None:
            instrument = self.monitor.get_group_code_by_rank(ts["item_rank_code"])
            datetime_range = self.monitor.get_datetime_with_index(instrument,ts["start"],ts["end"])
        else:
            instrument = df_target["instrument"].values[0]
            datetime_range = x_range
        price_array = ts["price_array"][-5:]
        price_class_item = compute_price_class(price_array,mode="first_last")
        # result.append({"instrument":instrument,"datetime":df_target["datetime"].dt.strftime('%Y%m%d').values[0]})
        target_title = "time range:{}/{} code:{},price class:{}".format(
            df_target["datetime"].dt.strftime('%m%d').values[self.input_chunk_length],df_target["datetime"].dt.strftime('%m%d').values[-1],
            instrument,price_class_item)
                  
        viz.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)            
    
    def vr_metric_show(self,target_info=None,last_vr_class_certainlys=None,last_target_vr_class=None):
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        names = ["output","target","price"]
        vr_class_certainlys_imp_index = torch.where(last_vr_class_certainlys==0)[0]
        size = len(vr_class_certainlys_imp_index) if len(vr_class_certainlys_imp_index)<27 else 27
        code_dict = {}
        idx = 0
        date_str = "20230313"
        result = []
        for r_index in range(len(vr_class_certainlys_imp_index)-1):
            s_index = vr_class_certainlys_imp_index[r_index]
            ts = target_info[s_index]
            target_vr_class_sample = last_target_vr_class[s_index].cpu().numpy()
            idx += 1
            if idx>size:
                break
            last_vr_class = last_vr_class_certainlys[s_index]
            last_target_class = last_target_vr_class[s_index]
            # 可以从全局变量中，通过索引获得实际价格
            df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
                                    (df_all["instrument_rank"]==ts["item_rank_code"])]            
            # 补充画出前面的label数据
            target_combine_sample = df_target["label"].values
            view_data = np.expand_dims(target_combine_sample,axis=1) 
            price_array = ts["price_array"][self.input_chunk_length-1:]
            price_class_item = compute_price_class(price_array,mode="fast")        
            price_target = df_target["label_ori"].values    
            price_target = np.expand_dims(price_target,axis=1)        
            view_data = np.concatenate((view_data,price_target),axis=1)
            x_range = np.array([i for i in range(ts["start"],ts["end"])])
            if self.monitor is not None:
                instrument = self.monitor.get_group_code_by_rank(ts["item_rank_code"])
                datetime_range = self.monitor.get_datetime_with_index(instrument,ts["start"],ts["end"])
            else:
                instrument = df_target["instrument"].values[0]
                datetime_range = x_range
            # result.append({"instrument":instrument,"datetime":df_target["datetime"].dt.strftime('%Y%m%d').values[0]})
            win = "win_{}".format(idx)
            target_title = "time range:{}/{} code:{},price class:{},last_vt_class:{}/{}".format(
                df_target["datetime"].dt.strftime('%m%d').values[self.input_chunk_length],df_target["datetime"].dt.strftime('%m%d').values[-1],
                instrument,price_class_item,last_vr_class,last_target_class)
            if price_class_item==CLASS_SIMPLE_VALUE_MAX:
                viz_result_suc.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)   
            elif price_class_item==0:
                viz_result_fail.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)  
            else:
                viz_result_nor.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)
                                         
    def build_scaler_map(self,scalers, batch_input_series,group_column="instrument_rank"):
        scaler_map = {}
        for i in range(len(scalers)):
            series = batch_input_series[i]
            group_col_val = series.static_covariates[group_column].values[0]
            scaler_map[group_col_val] = scalers[i]
        return scaler_map

class TFTExtModel(MixedCovariatesTorchModel):
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
        **kwargs,
    ):
        """重载darts相关类"""
        
        self.mode = mode
        model_kwargs = {key: val for key, val in self.model_params.items()}
        if "devices" in model_kwargs["pl_trainer_kwargs"]:
            self.device = "cuda:" + str(model_kwargs["pl_trainer_kwargs"]["devices"][0])
        else:
            self.device = "cpu"
        self.use_weighted_loss_func = use_weighted_loss_func
            
        if likelihood is None and loss_fn is None:
            # This is the default if no loss information is provided
            model_kwargs["loss_fn"] = None
            model_kwargs["likelihood"] = QuantileRegression()
            
        # 单独定制不确定损失
        if self.use_weighted_loss_func:
            # 定义损失函数种类数量
            params = torch.ones(loss_number, requires_grad=True)
            loss_sigma = torch.nn.Parameter(params)        
            loss_fn = UncertaintyLoss(device=self.device,loss_sigma=loss_sigma)    
            model_kwargs["loss_fn"] = loss_fn 
            model_kwargs["likelihood"] = likelihood
            self.loss_number = loss_number
            
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)
            
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = (
            categorical_embedding_sizes
            if categorical_embedding_sizes is not None
            else {}
        )
        self.add_relative_index = add_relative_index
        self.output_dim: Optional[Tuple[int, int]] = None
        self.norm_type = norm_type
        self.monitor = monitor
        self.past_split = past_split
        self.filter_conv_index = filter_conv_index
    
    def _build_vriable_metas(self,tensors,static_covariates,seq=0):   
        
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]
        conv_defs = [
                "past_target",
                "past_covariate",
                "historic_future_covariate",
            ]          
        input_meta = {}
        for i in range(len(type_names)):
            type_name = type_names[i]
            var_name = variable_names[i]
            tensor = tensors[i]
            # 根据相关设置，决定是否使用单独目标参数
            if hide_target:
                if type_name=="past_target":
                    type_values = ["target_0"]
                elif type_name=="future_target":
                    type_values = ["target_0"]
                else:
                    type_values = [f"{var_name}_{i}" for i in range(tensor.shape[1])]
            else:
                type_values = [f"{var_name}_{i}" for i in range(tensor.shape[1])]
            input_meta[type_name] = type_values
            
        variables_meta = {
            "input": input_meta,
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
           
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in conv_defs:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                        self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index(
                            [i for i in range(static_covariates.shape[1])]
                        )
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                        zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the TFTModel documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )    
        
        return variables_meta,categorical_embedding_sizes  
        
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
        
        return _TFTCusModule(
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
            **self.pl_module_params,
        )      
    
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        val_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):
        """重载原方法"""
        # guarantee that all inputs are either list of `TimeSeries` or `None`
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        val_series = series2seq(val_series)
        val_past_covariates = series2seq(val_past_covariates)
        val_future_covariates = series2seq(val_future_covariates)

        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            past_covariates, future_covariates = self.generate_fit_encodings(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        self._verify_past_future_covariates(
            past_covariates=past_covariates, future_covariates=future_covariates
        )
        self._verify_static_covariates(series[0].static_covariates)

        # Check that dimensions of train and val set match; on first series only
        if val_series is not None:
            if self.encoders.encoding_available:
                (
                    val_past_covariates,
                    val_future_covariates,
                ) = self.generate_fit_encodings(
                    series=val_series,
                    past_covariates=val_past_covariates,
                    future_covariates=val_future_covariates,
                )
            self._verify_past_future_covariates(
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )
            self._verify_static_covariates(val_series[0].static_covariates)

            match = (
                series[0].width == val_series[0].width
                and (past_covariates[0].width if past_covariates is not None else None)
                == (
                    val_past_covariates[0].width
                    if val_past_covariates is not None
                    else None
                )
                and (
                    future_covariates[0].width
                    if future_covariates is not None
                    else None
                )
                == (
                    val_future_covariates[0].width
                    if val_future_covariates is not None
                    else None
                )
            )
            raise_if_not(
                match,
                "The dimensions of the series in the training set "
                "and the validation set do not match.",
            )

        train_dataset = self._build_train_dataset(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=max_samples_per_ts,
            mode="train"
        )

        if val_series is not None:
            val_dataset = self._build_train_dataset(
                target=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
                max_samples_per_ts=max_samples_per_ts,
                mode="valid"
            )
        else:
            val_dataset = None

        # Pro-actively catch length exceptions to display nicer messages
        length_ok = True
        try:
            len(train_dataset)
        except ValueError:
            length_ok = False
        raise_if(
            not length_ok or len(train_dataset) == 0,  # mind the order
            "The train dataset does not contain even one training sample. "
            + "This is likely due to the provided training series being too short. "
            + "This model expect series of length at least {}.".format(
                self.min_train_series_length
            ),
        )
        logger.info(f"Train dataset contains {len(train_dataset)} samples.")

        if isinstance(series, TimeSeries):
            # if only one series is provided, save it for prediction time (including covariates, if available)
            self.training_series = series
            self.static_covariates = series.static_covariates
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
        else:
            self.static_covariates = series[0].static_covariates
            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True
        self._fit_called = True

        return self.fit_from_dataset(
            train_dataset, val_dataset, trainer, verbose, epochs, num_loader_workers
        )
 
    @random_method
    def fit_from_dataset(
        self,
        train_dataset: TrainingDataset,
        val_dataset: Optional[TrainingDataset] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        num_loader_workers: int = 0,
    ):
        """重载父类方法，规避相关数据检查"""
        self._fit_called = True
        # self._verify_train_dataset_type(train_dataset)

        # Pro-actively catch length exceptions to display nicer messages
        train_length_ok, val_length_ok = True, True
        try:
            len(train_dataset)
        except ValueError:
            train_length_ok = False
        if val_dataset is not None:
            try:
                len(val_dataset)
            except ValueError:
                val_length_ok = False

        raise_if(
            not train_length_ok or len(train_dataset) == 0,  # mind the order
            "The provided training time series dataset is too short for obtaining even one training point.",
            logger,
        )
        flag = val_dataset is not None and (not val_length_ok or len(val_dataset) == 0)
        raise_if(
            flag,
            "The provided validation time series dataset is too short for obtaining even one training point.",
            logger,
        )

        train_sample = train_dataset[0]
        if self.model is None:
            # 使用future_target部分(倒数第二列)，进行输出维度判断
            self.train_sample, self.output_dim = train_sample, train_sample[-2].shape[1]
            self._init_model(trainer)
            self.model.monitor = self.monitor
        else:
            # Check existing model has input/output dims matching what's provided in the training set.
            raise_if_not(
                len(train_sample) == len(self.train_sample),
                "The size of the training set samples (tuples) does not match what the model has been "
                "previously trained on. Trained on tuples of length {}, received tuples of length {}.".format(
                    len(self.train_sample), len(train_sample)
                ),
            )
            same_dims = tuple(
                s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in train_sample
            ) == tuple(s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in self.train_sample)
            raise_if_not(
                same_dims,
                "The dimensionality of the series in the training set do not match the dimensionality"
                " of the series the model has previously been trained on. "
                "Model input/output dimensions = {}, provided input/ouptput dimensions = {}".format(
                    tuple(
                        s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in self.train_sample
                    ),
                    tuple(s.shape[1] if (s is not None and isinstance(s, np.ndarray)) else None for s in train_sample),
                ),
            )

        # Setting drop_last to False makes the model see each sample at least once, and guarantee the presence of at
        # least one batch no matter the chosen batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_loader_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._batch_collate_filter,
        )

        # Prepare validation data
        val_loader = (
            None
            if val_dataset is None
            else DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_loader_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=self._batch_collate_filter,
            )
        )

        # if user wants to train the model for more epochs, ignore the n_epochs parameter
        train_num_epochs = epochs if epochs > 0 else self.n_epochs

        # setup trainer
        self._setup_trainer(trainer, verbose, train_num_epochs)

        # TODO: multiple training without loading from checkpoint is not trivial (I believe PyTorch-Lightning is still
        #  working on that, see https://github.com/PyTorchLightning/pytorch-lightning/issues/9636)
        if self.epochs_trained > 0 and not self.load_ckpt_path:
            logger.warning(
                "Attempting to retrain the model without resuming from a checkpoint. This is currently "
                "discouraged. Consider setting `save_checkpoints` to `True` and specifying `model_name` at model "
                f"creation. Then call `model = {self.__class__.__name__}.load_from_checkpoint(model_name, "
                "best=False)`. Finally, train the model with `model.fit(..., epochs=new_epochs)` where "
                "`new_epochs` is the sum of (epochs already trained + some additional epochs)."
            )
        
        # Train model
        if epochs>0 and self.n_epochs>0:
            self._train(train_loader, val_loader)
        return self,train_loader,val_loader
           
    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
        mode="train"
    ) -> CustomSequentialDataset:
        
        return CustomSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self._supports_static_covariates(),
            mode=mode
        )

    def _build_inference_dataset(
        self,
        target: Sequence[TimeSeries],
        n: int,
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
    ) -> CustomInferenceDataset:

        return CustomInferenceDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            use_static_covariates=self._supports_static_covariates(),
        )
        
    def _verify_inference_dataset_type(self, inference_dataset: InferenceDataset):
        """重载方法以规避类型检查"""
        _raise_if_wrong_type(inference_dataset, CustomInferenceDataset)
    
                
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
        # 重点关注价格指数,只对价格涨幅不达标的数据进行增强
        # p_taraget_class = compute_price_class(target_info["price_array"][self.input_chunk_length:])
        # if p_taraget_class not in [0,1]:
        #     return None
        
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
            
        # 重点关注前期走势比较平的
        # slope = slope_classify_compute(focus_target,threhold=2)
        # if slope!=SLOPE_SHAPE_SMOOTH:
        #     return None
        
        # target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
        # target_unscale = self.model.get_inverse_data(target[:,:,0],target_info=target_info,single_way=True).transpose(1,0)

    def _batch_collate_filter(self,ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        
        batch = []
        max_cnt = 0
        adj_max_cnt = 0
        aug_repeat_size = 10
        last_raise_range = []
        
        def data_process(batch,batch_item,target_class,max_cnt=0,adj_max_cnt=0):
            # 训练部分数据增强
            rtn_item = (batch_item[0],batch_item[1],batch_item[2],batch_item[3],batch_item[4],batch_item[5][0],batch_item[6],batch_item[7],batch_item[8]) 
            batch.append(rtn_item) 
            # 训练阶段做增强
            if self.trainer.state.stage==RunningStage.TRAINING:
                for i in range(3):
                    b_rebuild = self.dynamic_build_training_data(batch_item)
                    # 不符合要求则不增强
                    if b_rebuild is None:
                        continue
                    adj_max_cnt += 1
                    batch.append(b_rebuild)   
            return max_cnt,adj_max_cnt
        # 过滤不符合条件的记录
        for b in ori_batch:
            if self.mode=="train":
                future_target = b[-2]
                target_class = b[-3]
            else:
                future_target = b[-1]
            # 如果每天的target数值都相同，则会出现loss的NAN，需要过滤掉
            if not np.all(future_target == future_target[0]):
                # 非训练部分，不需要数据增强，直接转换返回
                if not self.model.training:
                    if self.mode=="predict":
                        batch.append(b)
                    else:
                        # 验证集也使用增强数据
                        # max_cnt,adj_max_cnt = data_process(batch,b,target_class,max_cnt=max_cnt,adj_max_cnt=adj_max_cnt)
                        rtn_item = (b[0],b[1],b[2],b[3],b[4],b[5][0],b[6],b[7],b[8])                        
                        batch.append(rtn_item)
                else:
                    max_cnt,adj_max_cnt = data_process(batch,b,target_class,max_cnt=max_cnt,adj_max_cnt=adj_max_cnt)
        last_raise_range = np.array(last_raise_range)
        ori_rate = max_cnt/len(ori_batch)
        after_rate = adj_max_cnt/len(batch)
        # print("img data cnt/rate:{}/{},after cnt/rate:{}/{}".format(max_cnt,ori_rate,len(batch),after_rate))
        # print("last_raise_range:{}/{}",np.unique(np.array(last_raise_range), return_counts=True))
        aggregated = []
        first_sample = batch[0]
        for i in range(len(first_sample)):
            elem = first_sample[i]
            if isinstance(elem, np.ndarray):
                sample_list = [sample[i] for sample in batch]
                aggregated.append(
                    torch.from_numpy(np.stack(sample_list, axis=0))
                )
            elif isinstance(elem, MinMaxScaler):
                aggregated.append([sample[i] for sample in batch])
            elif isinstance(elem, StockNormalizer):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, TimeSeries):
                aggregated.append([sample[i] for sample in batch])
        return tuple(aggregated)
                                                             
    @staticmethod
    def _supports_static_covariates() -> bool:
        return True         