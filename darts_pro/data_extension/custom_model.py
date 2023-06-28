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
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from cus_utils.encoder_cus import StockNormalizer
from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import compute_price_class,compute_price_class_batch,slope_classify_compute
from cus_utils.metrics import compute_cross_metrics,compute_vr_metrics
import cus_utils.global_var as global_var
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,SLOPE_SHAPE_SHAKE,SLOPE_SHAPE_SMOOTH

import torchmetrics
from torchmetrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import tsaug
from losses.mtl_loss import UncertaintyLoss,MseLoss

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

viz_input = TensorViz(env="data_train")
viz_input_2 = TensorViz(env="data_train_unscale")
viz_input_aug = TensorViz(env="data_train_aug")
viz_input_nag_aug = TensorViz(env="data_train_nag_aug")

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
 
    
class _TFTCusModule(_TFTModule):
    def __init__(
        self,
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
        use_weighted_loss_func=False,
        loss_number=3,
        device="cpu",
        monitor=None,
        **kwargs,
    ):
        
        super().__init__(output_dim,variables_meta,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                full_attention,feed_forward,hidden_continuous_size,categorical_embedding_sizes,dropout,add_relative_index,norm_type,**kwargs)
        
        # 分类模式涵盖前后2段，因此定义2个分类层
        mss_num = 4
        vr_range_num = len(CLASS_SIMPLE_VALUES.keys())
        pred_len = kwargs["output_chunk_length"]
        self.class1_len = int((pred_len-1)/2)
        self.class2_len = pred_len -1 - self.class1_len        
        self.classify_layer_1 = self._construct_classify_layer(self.class1_len,mss_num).to(device)
        self.classify_layer_2 = self._construct_classify_layer(self.class2_len+1,mss_num).to(device)
        self.slope_layer = self._construct_classify_layer(pred_len,pred_len-1).to(device)  
        # 涨跌幅度分类
        self.classify_vr_layer = self._construct_classify_layer(pred_len,vr_range_num).to(device)  
        # mse损失计算                
        self.mean_squared_error = MeanSquaredError().to(device)
        
        # 提前初始化loss计算对象，避免在加载权重的时候出现空指针
        if use_weighted_loss_func and not isinstance(self.criterion,UncertaintyLoss):
            params = torch.ones(loss_number, requires_grad=True)
            loss_sigma = torch.nn.Parameter(params)    
            self.criterion = UncertaintyLoss(loss_sigma=loss_sigma,device=device) 
        
        self.val_results = {}
        
        # 优化器执行频次
        self.lr_freq = {"interval":"step","frequency":88}
               
    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """重载训练方法，加入分类模式"""
        
        out = super().forward(x_in)    
        # 从分位数取得中间数值
        output_sample = out[:,:,0,0]
        
        slope_out = self.slope_layer(output_sample)
        # 涨跌幅度分类
        vr_class = self.classify_vr_layer(output_sample)
        vr_class = torch.unsqueeze(vr_class,1)
        return (out,slope_out,vr_class)
        
    def _construct_classify_layer(self, input_dim, output_dim):
        """使用全连接进行分类数值输出"""
        return nn.Linear(input_dim, output_dim).double()
        # layers = []
        # layers.append(nn.Linear(input_dim, output_dim).double())
        # layers.append(nn.LogSoftmax(dim=-1))
        # return nn.Sequential(*layers)

    def _process_input_batch(
        self, input_batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """重载方法，以适应数据结构变化"""
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
        ) = input_batch
        dim_variable = 2

        x_past = torch.cat(
            [
                tensor
                for tensor in [
                    past_target,
                    past_covariates,
                    historic_future_covariates,
                ]
                if tensor is not None
            ],
            dim=dim_variable,
        )
        return x_past, future_covariates, static_covariates
               
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        
        # app_logger.debug("batch_idx:{}".format(batch_idx))
        # 包括数值数据，以及分类输出
        (output,slope_out,vr_class) = self._produce_train_output(train_batch[:5])
        # 目标数据里包含分类信息
        scaler,target_class,target,target_info = train_batch[5:]
        target_class = target_class[:,:,0]
        target_trend_class = target_class[:,:2]
        target_vr_class = target_class[:,1]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained
        loss,detail_loss = self._compute_loss((output,slope_out,vr_class), (target,target_class,scaler,target_info))
        (value_range_loss,value_diff_loss,corr_loss,mse_loss) = detail_loss
        self.log("base_lr",self.trainer.optimizers[0].param_groups[0]["lr"])
        # self.log("class1_lr",self.trainer.optimizers[0].param_groups[1]["lr"])
        # self.log("class2_lr",self.trainer.optimizers[0].param_groups[2]["lr"])
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.custom_histogram_adder(batch_idx)
        self._calculate_metrics(output, target, self.train_metrics)
        # 走势分类交叉熵损失
        # cross_loss = compute_cross_metrics(out_class, target_trend_class)
        # mse损失
        # self.log("train_mse_loss", mse_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("train_corr_loss", corr_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("train_scope_loss", value_diff_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("train_value_range_loss", value_range_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("train_mse_loss", mse_loss, batch_size=train_batch[0].shape[0], prog_bar=True)        
        return loss
    
    def validation_step(self, val_batch_ori, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        
        # val_batch = self.filter_batch_by_spec_date(val_batch_ori)
        val_batch = val_batch_ori
        (output,slope_out,vr_class) = self._produce_train_output(val_batch[:5])
        _,target_class,target,target_info = val_batch[5:]  
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,1]
        # 全部损失
        loss,detail_loss = self._compute_loss((output,slope_out,vr_class), (target,target_class,None,target_info))
        (value_range_loss,value_diff_loss,corr_loss,mse_loss) = detail_loss
        # 相关系数损失
        # corr_loss = self.compute_ccc_metrics(output, target)
        # 距离损失MSE
        last_sec_acc,last_sec_out_bool = self.compute_scope_metrics(output, target)
        # 走势分类交叉熵损失
        # cross_loss = compute_cross_metrics(out_class, target_trend_class)
        # 总体涨跌幅度分类损失
        # value_range_loss = compute_vr_metrics(vr_class[:,0,:], target_vr_class)         
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_corr_loss", corr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("mse_loss", mse_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_value_range_loss", value_range_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        vr_class_certainlys = self.build_vr_class_cer(vr_class[:,0,:])
        output_inverse = self.get_inverse_data(output.cpu().numpy()[:,:,0,0],target_info=target_info)
        # last_batch_index,last_batch_imp_index,item_codes = self.build_last_batch_index(vr_class_certainlys,target_vr_class,target_info=target_info)
        # 涨跌幅度类别的准确率
        # vr_acc,import_vr_acc,import_recall,import_price_acc,import_price_nag,price_class,\
        #     import_price_acc_withls,import_price_nag_withls = self.compute_vr_class_acc(
        #     vr_class_certainlys, target_vr_class,target_info=target_info,last_sec_index=last_sec_out_bool)  
        import_vr_acc,import_recall,import_price_acc,import_price_nag,price_class,import_index \
             = self.compute_real_class_acc(target, target_vr_class,output_inverse=output_inverse,target_info=target_info)              
        # self.log("vr_acc", vr_acc, batch_size=val_batch[0].shape[0], prog_bar=True)     
        self.log("import_vr_acc", import_vr_acc, batch_size=val_batch[0].shape[0], prog_bar=True)    
        self.log("import_recall", import_recall, batch_size=val_batch[0].shape[0], prog_bar=True)   
        # self.log("last_section_slop_acc", last_sec_acc, batch_size=val_batch[0].shape[0], prog_bar=True)  
        self.log("import_price_acc", import_price_acc, batch_size=val_batch[0].shape[0], prog_bar=True)       
        self.log("import_price_nag", import_price_nag, batch_size=val_batch[0].shape[0], prog_bar=True)   
        print("import_vr_acc:{},import_price_acc:{},import_price_nag:{}".format(import_vr_acc,import_price_acc,import_price_nag))
        past_target = val_batch[0]
        self.val_metric_show(output,target,price_class,vr_class_certainlys,target_vr_class,output_inverse=output_inverse,
                             past_target=past_target,target_info=target_info,import_index=import_index)
        self._calculate_metrics(output, target, self.val_metrics)
        # 记录相关统计数据
        # cr_loss = round(cross_loss.item(),5)
        cr_loss = 0
        # record_results = {"val_loss":round(loss.item(),5),"val_corr_loss":round(corr_loss.item(),5),
        #                       "ce_loss":cr_loss,"value_diff_loss":round(value_diff_loss.item(),5),
        #                       "value_range_loss":round(value_range_loss.item(),5),"vr_acc":round(vr_acc.item(),5),
        #                       "import_vr_acc":round(import_vr_acc.item(),5)}
        # self.process_val_results(record_results,self.current_epoch)
        return loss
    
    def filter_batch_by_spec_date(self,val_batch,date_spec="20230307"):
        df_all = global_var.get_value("dataset").df_all
        df_part = df_all[df_all["datetime"].dt.strftime('%Y%m%d')==date_spec]
        rtn_index = []
        target_info_list = val_batch[-1]
        spec_codes = []
        for i in range(len(target_info_list)):
            target_info = target_info_list[i]
            df_match = df_part[df_part["instrument_rank"]==target_info["item_rank_code"]]
            if df_match.shape[0]==0:
                logger.warning("no data for:{}".format(target_info["item_rank_code"]))
            if df_match["time_idx"].values[0]==(target_info["total_start"]+target_info["future_start"]):
                if target_info["item_rank_code"] not in spec_codes:
                    spec_codes.append(target_info["item_rank_code"])
                    rtn_index.append(i)
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler,target_class,target,target_info) = val_batch
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
    
    def compute_vr_class_acc(self,vr_class,vr_target,target_info=None,last_sec_index=None):
        """计算涨跌幅分类准确度"""

        # 总体准确率
        total_acc = torch.sum(vr_class==vr_target)/vr_target.shape[0]
        # 重点类别的准确率
        import_index = torch.nonzero(vr_class==CLASS_SIMPLE_VALUE_MAX)[:,0]
        import_acc_count = torch.sum(vr_target[import_index]==CLASS_SIMPLE_VALUE_MAX)
        if import_index.shape[0]==0:
            import_acc = torch.tensor(0.0)
        else:
            import_acc = import_acc_count/import_index.shape[0]
         
        # 重点类别的召回率    
        total_imp_cnt = torch.sum(vr_target==CLASS_SIMPLE_VALUE_MAX)
        if total_imp_cnt!=0:
            import_recall = import_acc_count/total_imp_cnt
        else:
            import_recall = 0

        # 重点类别的价格准确率
        price_class = []
        for item in target_info:
            p_taraget_class = compute_price_class(item["price_array"])
            price_class.append(p_taraget_class)
        price_class = torch.Tensor(np.array(price_class)).int()
        import_index = torch.nonzero(vr_class==CLASS_SIMPLE_VALUE_MAX)[:,0]
        import_price_acc_count = torch.sum(price_class[import_index]==CLASS_SIMPLE_VALUE_MAX)
        import_price_nag_count = torch.sum(price_class[import_index]==0)
        if import_index.shape[0]==0:
            import_price_acc = torch.tensor(0.0)
            import_price_nag = torch.tensor(0.0)
        else:
            import_price_acc = import_price_acc_count/import_index.shape[0]
            import_price_nag = import_price_nag_count/import_index.shape[0]
        
        # 介入最后一段上涨下跌预测后的价格准确率
        import_index_withls = torch.where((vr_class==CLASS_SIMPLE_VALUE_MAX) & (last_sec_index))[0]
        import_price_nag_withls_count = torch.sum(price_class[import_index_withls]==0)
        import_price_acc_withls_count = torch.sum(price_class[import_index_withls]==CLASS_SIMPLE_VALUE_MAX)
        if import_index_withls.shape[0]==0:
            import_price_acc_withls = torch.tensor(0.0)
            import_price_nag_withls = torch.tensor(0.0)
        else:
            import_price_acc_withls = import_price_acc_withls_count/import_index_withls.shape[0]
            import_price_nag_withls = import_price_nag_withls_count/import_index_withls.shape[0]
                  
        return total_acc,import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_acc_withls,import_price_nag_withls

    def get_inverse_data(self,output,target_info=None,single_way=False):
        dataset = global_var.get_value("dataset")
        
        if single_way:
            group_code = target_info["item_rank_code"]
            scaler_sample = dataset.get_scaler_by_group_code(group_code)  
            pred_inverse = scaler_sample._fitted_params[0].inverse_transform(output) 
            return pred_inverse                          
             
        pred_inverse = []
        for index in range(output.shape[0]):
            output_item = np.expand_dims(output[index],axis=-1)
            group_code = target_info[index]["item_rank_code"]
            scaler_sample = dataset.get_scaler_by_group_code(group_code)
            pred_center_data = scaler_sample._fitted_params[0].inverse_transform(output_item)     
            pred_inverse.append(pred_center_data[:,0])
        return np.stack(pred_inverse)
        
    def compute_real_class_acc(self,target,vr_target_tensor,target_info=None,output_inverse=None):
        """计算涨跌幅分类准确度"""
        
        # 总体准确率
        slope_out = (output_inverse[:,1:]  - output_inverse[:,:-1])/output_inverse[:,:-1]*100
        # 整体需要有上涨幅度
        import_index_bool = np.sum(slope_out,axis=1)>3
        # 最后一段需要上涨
        import_index_bool = import_index_bool & (slope_out[:,-1]>0)
        recent_data = np.array([item["label_array"][:self.input_chunk_length] for item in target_info])
        # 可信度检验，预测值不能偏离太多
        import_index_bool = import_index_bool & ((output_inverse[:,0] - recent_data[:,-1])/recent_data[:,-1]<0.1)
        # 第一天需要上涨
        import_index_bool = import_index_bool & (slope_out[:,0]>0)
        # 第二天如果下跌，幅度不能超过第一天上涨幅度
        # s1 = slope_out[:,1]>=0
        # s2 = (slope_out[:,1]<0) & (slope_out[:,0]>abs(slope_out[:,1]))
        # import_index_bool = import_index_bool & (s1 | s2)
        # 与近期高点比较，不能差太多
        # recent_max = np.max(recent_data,axis=1)
        # import_index_bool = import_index_bool & (output_inverse[:,0]>recent_max)
        
        import_index = np.where(import_index_bool)[0]
        
        vr_target,raise_range = compute_price_class_batch(target.cpu().numpy()[:,:,0],mode="fast")
        raise_stat = pd.DataFrame(raise_range[import_index])
        app_logger.info("raise_stat:{}".format(raise_stat.describe()))
        # 重点类别的准确率
        import_acc_count = np.sum(vr_target[import_index]==CLASS_SIMPLE_VALUE_MAX)
        if import_index.shape[0]==0:
            import_acc = torch.tensor(0.0)
        else:
            import_acc = import_acc_count/import_index.shape[0]
         
        # 重点类别的召回率    
        total_imp_cnt = np.sum(vr_target==CLASS_SIMPLE_VALUE_MAX)
        if total_imp_cnt!=0:
            import_recall = import_acc_count/total_imp_cnt
        else:
            import_recall = 0

        # 重点类别的价格准确率
        price_class = []
        for item in target_info:
            p_taraget_class = compute_price_class(item["price_array"],mode="fast")
            price_class.append(p_taraget_class)
        price_class = np.array(price_class)
        import_price_acc_count = np.sum(price_class[import_index]==CLASS_SIMPLE_VALUE_MAX)
        import_price_nag_count = np.sum(price_class[import_index]==0)
        if import_index.shape[0]==0:
            import_price_acc = 0
            import_price_nag = 0
        else:
            import_price_acc = import_price_acc_count/import_index.shape[0]
            import_price_nag = import_price_nag_count/import_index.shape[0]

        return import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_index_bool
        
    def compute_corr_metrics(self,output,target):
        num_outputs = output.shape[0]
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        corr_loss = self.pearson(output_sample.transpose(1,0),target.squeeze(-1).transpose(1,0))  
        corr_loss = torch.mean(1 - corr_loss)
        return corr_loss      
 
    def compute_ccc_metrics(self,output,target):
        return self.criterion.ccc_loss_comp(output,target)
       
    def compute_scope_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        compare_target = target[:,:,0]
        last_sec_out_bool = (output_sample[:,-1] - output_sample[:,-2])>0
        last_sec_tar_bool  = (compare_target[:,-1] - compare_target[:,-2])>0
        last_sec_acc_list = (last_sec_out_bool==last_sec_tar_bool)
        last_sec_acc = torch.sum(last_sec_acc_list)/compare_target.shape[0]
        return last_sec_acc,last_sec_out_bool

    def compute_value_diff_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        output_be = output_sample[:,[0,-1]]
        target_be = target[:,[0,-1],0]
        mse_loss = self.mean_squared_error(output_be, target_be)
        return mse_loss 
    
    def _compute_loss(self, output, target):
        """重载父类方法"""
        
        (output_value,slope_out,vr_class) = output
        (target_real,target_class,scaler,target_info) = target
        future_target = torch.Tensor([t["future_target"] for t in target_info]).to(self.device)
        # output_inverse = []
        # for i in range(output_value.shape[0]):
        #     out_inverse = scaler[i].inverse_transform(output_value[i,:,0,0]) 
        #     output_inverse.append(out_inverse)                                             
        output_combine = (output_value.squeeze(dim=-1),slope_out,vr_class)
        
        if self.likelihood:
            # 把似然估计损失叠加到自定义多任务损失里
            loss_like = self.likelihood.compute_loss(output_value, target_real)
            if self.criterion is None:
                return loss_like
            mtl_loss = self.criterion(output_combine, target_real,outer_loss=loss_like)
            return mtl_loss
        else:
            return self.criterion(output_combine, (target_real,future_target,target_class))

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
        """configures optimizers and learning rate schedulers for model optimization."""

        # A utility function to create optimizer and lr scheduler from desired classes
        def _create_from_cls_and_kwargs(cls, kws):
            try:
                return cls(**kws)
            except (TypeError, ValueError) as e:
                raise_log(
                    ValueError(
                        "Error when building the optimizer or learning rate scheduler;"
                        "please check the provided class and arguments"
                        "\nclass: {}"
                        "\narguments (kwargs): {}"
                        "\nerror:\n{}".format(cls, kws, e)
                    ),
                    logger,
                )

        # 对于最后的全连接层，使用高倍数lr
        optimizer_kws = {k: v for k, v in self.optimizer_kwargs.items()}
        ignored_params = list(map(id, self.classify_layer_1.parameters())) + \
            list(map(id, self.classify_layer_2.parameters())) + \
            list(map(id, self.slope_layer.parameters())) + \
            list(map(id, self.classify_vr_layer.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())       
        base_lr = self.optimizer_kwargs["lr"] 
        optimizer_kws["params"] = [
                    {'params': base_params},
                    {'params': self.classify_layer_1.parameters(), 'lr': base_lr*10},
                    {'params': self.classify_layer_2.parameters(), 'lr': base_lr*10},
                    {'params': self.slope_layer.parameters(), 'lr': base_lr*10},
                    {'params': self.classify_vr_layer.parameters(), 'lr': base_lr*10}]

        optimizer = _create_from_cls_and_kwargs(self.optimizer_cls, optimizer_kws)
        
        if self.lr_scheduler_cls is not None:
            lr_sched_kws = {k: v for k, v in self.lr_scheduler_kwargs.items()}
            lr_sched_kws["optimizer"] = optimizer

            # ReduceLROnPlateau requires a metric to "monitor" which must be set separately, most others do not
            lr_monitor = lr_sched_kws.pop("monitor", None)

            lr_scheduler = _create_from_cls_and_kwargs(
                self.lr_scheduler_cls, lr_sched_kws
            )
                    
            return [optimizer], {
                "scheduler": lr_scheduler,
                "interval": self.lr_freq["interval"],
                "frequency": self.lr_freq["frequency"],              
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }
        else:
            return optimizer
             
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
                                
    def val_metric_show(self,output,target,price_class,vr_class,target_vr_class,output_inverse=None,past_target=None,target_info=None,import_index=None):
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        names = ["output","target","price"]
        vr_class_certainlys = vr_class.cpu().numpy()
        vr_class_certainlys_imp_index = np.expand_dims(np.array([i for i in range(len(target_info))]),axis=-1)
        vr_class_certainlys_imp_index = np.where(import_index)[0]
        size = len(vr_class_certainlys_imp_index) if len(vr_class_certainlys_imp_index)<27 else 27
        code_dict = {}
        idx = 0
        date_str = "20230313"
        result = []
        for r_index in range(len(vr_class_certainlys_imp_index)-1):
            s_index = vr_class_certainlys_imp_index[r_index]
            ts = target_info[s_index]
            target_vr_class_sample = target_vr_class[s_index].cpu().numpy()
            vr_class_certainly = vr_class_certainlys[s_index]
            # 检查指定日期的数据
            # if df_all[(df_all["time_idx"]==ts["end"]-1)&(df_all["instrument_rank"]==ts["item_rank_code"])]["datetime"].dt.strftime('%Y%m%d').values[0]!=date_str:
            #     continue         
            idx += 1
            if idx>size:
                break
            pred_center_data = output_inverse[s_index]
            # 可以从全局变量中，通过索引获得实际价格
            df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
                                    (df_all["instrument_rank"]==ts["item_rank_code"])]            
            # 补充画出前面的label数据
            target_combine_sample = df_target["label"].values
            
            pad_data = np.array([0 for i in range(self.input_chunk_length)])
            pred_center_data = np.concatenate((pad_data,pred_center_data),axis=-1)
            view_data = np.stack((pred_center_data,target_combine_sample),axis=0).transpose(1,0)
            price_class_item = price_class[s_index]
            price_target = df_target["label_ori"].values    
            price_target = np.expand_dims(price_target,axis=0)        
            view_data = np.concatenate((view_data,price_target.transpose(1,0)),axis=1)
            x_range = np.array([i for i in range(ts["start"],ts["end"])])
            if self.monitor is not None:
                instrument = self.monitor.get_group_code_by_rank(ts["item_rank_code"])
                datetime_range = self.monitor.get_datetime_with_index(instrument,ts["start"],ts["end"])
            else:
                instrument = df_target["instrument"].values[0]
                datetime_range = x_range
            # result.append({"instrument":instrument,"datetime":df_target["datetime"].dt.strftime('%Y%m%d').values[0]})
            win = "win_{}".format(idx)
            target_title = "time range:{}/{} code:{},price class:{},tar_vr class:{}".format(
                df_target["datetime"].dt.strftime('%Y%m%d').values[self.input_chunk_length],df_target["datetime"].dt.strftime('%Y%m%d').values[-1],
                instrument,price_class_item,target_vr_class_sample)
            viz_input_2.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)   
        # print("pred result:{}".format(result))
             
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

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
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

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
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
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
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

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes
        
        return _TFTCusModule(
            output_dim=self.output_dim,
            variables_meta=variables_meta,
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
                mode="val"
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
        self._verify_train_dataset_type(train_dataset)

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
            collate_fn=self._batch_collate_fn,
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
                collate_fn=self._batch_collate_fn,
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
    
    def dynamic_build_training_data(self,item):
        """使用数据增强，调整训练数据"""
        
        # dc = np.random.random_integers(1000,1000+dynaimc_range*10)/1000
        # dt = np.random.random_integers(1000,1000+dynaimc_range*10)/1000
        past_covariate = item[1]
        future_past_covariate = item[5][1]
        past_target = item[0]
        future_target = item[-2]
        target_info = item[-1]
        scaler = item[5][0]
        
        # 重点关注价格指数,只对价格涨幅不达标的数据进行增强
        # p_taraget_class = compute_price_class(target_info["price_array"])
        # if p_taraget_class not in [0,1]:
        #     return None
        
        target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
        target_unscale = self.model.get_inverse_data(target[:,:,0],target_info=target_info,single_way=True).transpose(1,0)
        
        # 重点关注前期走势比较平的
        # focus_target = target_unscale[self.input_chunk_length-5:self.input_chunk_length]
        # slope = slope_classify_compute(focus_target,threhold=2)
        # if slope!=SLOPE_SHAPE_SMOOTH:
        #     return None
        
        # # 重点关注最后一段上涨的
        # if target_unscale[-1] - target_unscale[-2] <= 0:
        #     return None
                
        # 与近期高点比较，不能差太多
        # recent_max= target_unscale[:self.input_chunk_length].max()
        # if target_unscale[-1]<recent_max and (recent_max-target_unscale[-1])/recent_max>2/100:
        #     return None   
        #
        # # 之前的价格涨幅筛选判断
        # price_arr = target_info["total_price_array"][self.input_chunk_length-self.output_chunk_length:self.input_chunk_length]
        # price_arr_slope = (price_arr[1:] - price_arr[:-1])/price_arr[:-1]
        # # 最近价格连续上涨不可信
        # if price_arr_slope[-1]>0 and price_arr_slope[-2]>0:
        #     return None
        # # 最近一天上涨幅度过高
        # if (price_arr_slope[-1]*100)>6:
        #     return None   
        # # 最后一天需要创出近期新高 
        # if ((target_info["total_price_array"].max()-price_arr[-1])/price_arr[-1])>0.0001:
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
        
        if np.random.randint(0,90)==1:
            # 可视化增强的数据
            index = np.random.randint(0,12)
            win = "win_{}".format(index)
            target_title = "code_{}".format(target_info["item_rank_code"])
            names = ["label","price"]
            price_array = np.concatenate((np.array([0] * (self.input_chunk_length-1)),target_info["price_array"]))
            price_array = np.expand_dims(price_array,axis=-1)
            view_data = np.concatenate((target_unscale,price_array),axis=-1)
            viz_input_aug.viz_matrix_var(view_data,win=win,title=target_title,names=names)        
        return rtn_item
 
    def dynamic_build_nag_training_data(self,item):
        """使用数据增强，调整训练数据"""

        past_covariate = item[1]
        future_past_covariate = item[5][1]
        past_target = item[0]
        future_target = item[-2]
        target_info = item[-1]
        scaler = item[5][0]
        
        # 重点关注价格指数,只对价格跌幅达标的数据进行增强
        p_taraget_class = compute_price_class(target_info["price_array"])
        if p_taraget_class!=0:
            return None
        
        target = np.expand_dims(np.concatenate((past_target,future_target),axis=0),axis=0)
        target_unscale = scaler.inverse_transform(target[0])
        
        # 重点关注前期走势比较平的
        focus_target = target_unscale[self.input_chunk_length-5:self.input_chunk_length]
        slope = slope_classify_compute(focus_target,threhold=2)
        if slope!=SLOPE_SHAPE_SMOOTH:
            return None

                        
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
        
        if np.random.randint(0,90)==1:
            # 可视化增强的数据
            index = np.random.randint(0,12)
            win = "win_{}".format(index)
            target_title = "code_{}".format(target_info["item_rank_code"])
            names = ["label","price"]
            price_array = np.concatenate((np.array([0] * (self.input_chunk_length-1)),target_info["price_array"]))
            price_array = np.expand_dims(price_array,axis=-1)
            view_data = np.concatenate((target_unscale,price_array),axis=-1)
            viz_input_nag_aug.viz_matrix_var(view_data,win=win,title=target_title,names=names)        
        return rtn_item
    
        
    def _batch_collate_fn(self,ori_batch: List[Tuple]) -> Tuple:
        """
        重载方法，调整数据处理模式
        """
        
        batch = []
        max_cnt = 0
        adj_max_cnt = 0
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
                        rtn_item = (b[0],b[1],b[2],b[3],b[4],b[5][0],b[6],b[7],b[8])                        
                        batch.append(rtn_item)
                else:
                    # 训练部分数据增强
                    rtn_item = (b[0],b[1],b[2],b[3],b[4],b[5][0],b[6],b[7],b[8])                    
                    # 增加不平衡类别的数量，随机小幅度调整数据
                    if target_class[1,0]==CLASS_SIMPLE_VALUE_MAX:
                        max_cnt += 1
                        batch.append(rtn_item)
                        adj_max_cnt += 1
                        for i in range(2):
                            b_rebuild = self.dynamic_build_training_data(b)
                            # 不符合要求则不增强
                            if b_rebuild is None:
                                continue
                            # b_rebuild = rtn_item
                            adj_max_cnt += 1
                            batch.append(b_rebuild)
                    elif target_class[1,0]==0:
                        batch.append(rtn_item)
                        # b_rebuild = self.dynamic_build_nag_training_data(b)
                        # # 不符合要求则不增强
                        # if b_rebuild is None:
                        #     continue    
                        # batch.append(b_rebuild)
                    else:
                        # 随机减少大类数量
                        if np.random.randint(0,2)==1:
                            batch.append(rtn_item)
        ori_rate = max_cnt/len(ori_batch)
        after_rate = adj_max_cnt/len(batch)
        # print("img data rate:{},after rate:{}".format(ori_rate,after_rate))
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