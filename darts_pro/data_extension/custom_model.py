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

from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import target_scale
from cus_utils.metrics import compute_cross_metrics,compute_vr_metrics
import cus_utils.global_var as global_var
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC

import torchmetrics
from torchmetrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

from losses.mtl_loss import CorrLoss,UncertaintyLoss

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

viz_input = TensorViz(env="data_train")
viz_input_2 = TensorViz(env="data_train_unscale")

def _build_forecast_series(
     points_preds: Union[np.ndarray, Sequence[np.ndarray]],
     input_series: TimeSeries,
     pred_class: np.ndarray,
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
    pred_mean_class = torch.mean(pred_class,dim=0)
    vr_mean_class = torch.mean(vr_class,dim=0)
    return (TimeSeries.from_times_and_values(
        time_index,
        values,
        freq=input_series.freq_str,
        columns=input_series.columns,
        static_covariates=input_series.static_covariates,
        hierarchy=input_series.hierarchy,
    ),pred_mean_class,vr_mean_class)
     
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
        self.last_classify_layer = self._construct_classify_layer(3,3).to(device)  
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
        
    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """重载训练方法，加入分类模式"""
        
        out = super().forward(x_in)    
        # 从分位数取得中间数值
        output_sample = torch.mean(out[:,:,0,:],dim=-1)
        
        out_classify = self.last_classify_layer(output_sample[:,2:])
        # 涨跌幅度分类
        vr_class = self.classify_vr_layer(output_sample)
        vr_class = torch.unsqueeze(vr_class,1)
        return (out,out_classify,vr_class)
        
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
        
        # 包括数值数据，以及分类输出
        (output,out_class,vr_class) = self._produce_train_output(train_batch[:5])
        # 目标数据里包含分类信息
        _,target_class,target,target_info = train_batch[5:]
        target_class = target_class[:,:,0]
        target_trend_class = target_class[:,:2]
        target_vr_class = target_class[:,1]
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained
        loss = self._compute_loss((output,out_class,vr_class),(target,target_class))
        # self.log("base_lr",self.trainer.optimizers[0].param_groups[0]["lr"])
        # self.log("class1_lr",self.trainer.optimizers[0].param_groups[1]["lr"])
        # self.log("class2_lr",self.trainer.optimizers[0].param_groups[2]["lr"])
        self.log("train_loss", loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.custom_histogram_adder(batch_idx)
        self._calculate_metrics(output, target, self.train_metrics)
        # 相关系数损失
        corr_loss = self.compute_ccc_metrics(output, target)
        # 走势分类交叉熵损失
        # cross_loss = compute_cross_metrics(out_class, target_trend_class)
        # 总体涨跌幅度分类损失
        value_range_loss = compute_vr_metrics(vr_class[:,0,:], target_vr_class) 
        # mse损失
        mse_loss = self.compute_mse_metrics(output, target)   
        # self.log("train_mse_loss", mse_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("train_corr_loss", corr_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("train_cross_loss", cross_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        self.log("train_value_range_loss", value_range_loss, batch_size=train_batch[0].shape[0], prog_bar=True)
        # self.log("train_mse_loss", mse_loss, batch_size=train_batch[0].shape[0], prog_bar=True)        
        return loss
    
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """performs the validation step"""
        
        (output,out_class,vr_class) = self._produce_train_output(val_batch[:5])
        scaler,target_class,target,target_info = val_batch[5:]  
        target_class = target_class[:,:,0]
        target_trend_class = target_class[:,0]
        target_vr_class = target_class[:,1]
        # 全部损失
        loss = self._compute_loss((output,out_class,vr_class), (target,target_class))
        # 相关系数损失
        corr_loss = self.compute_corr_metrics(output, target)
        # 距离损失MSE
        # mse_loss = self.compute_mse_metrics(output, target)
        # 走势分类交叉熵损失
        # cross_loss = compute_cross_metrics(out_class, target_trend_class)
        # 总体涨跌幅度分类损失
        value_range_loss = compute_vr_metrics(vr_class[:,0,:], target_vr_class)         
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_corr_loss", corr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_cross_loss", cross_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        self.log("val_value_range_loss", value_range_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        value_diff_loss = self.compute_value_diff_metrics(output, target)
        mse_loss = self.compute_mse_metrics(output, target)   
        self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_mse_loss", mse_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        vr_class_certainlys = self.build_vr_class_cer(vr_class[:,0,:])
        last_batch_index,last_batch_imp_index,item_codes = self.build_last_batch_index(vr_class_certainlys,target_vr_class,target_info=target_info)
        # 涨跌幅度类别的准确率
        vr_acc,import_vr_acc,import_acc_count,import_recall,import_sec_acc,import_sec_acc_count,import_sec_recall \
             = self.compute_vr_class_acc(vr_class_certainlys, target_vr_class,target_info=target_info,last_batch_index=last_batch_index)  
        self.log("vr_acc", vr_acc, batch_size=val_batch[0].shape[0], prog_bar=True)     
        self.log("import_vr_acc", import_vr_acc, batch_size=val_batch[0].shape[0], prog_bar=True)    
        self.log("import_recall", import_recall, batch_size=val_batch[0].shape[0], prog_bar=True)         
        # self.log("import_sec_acc", import_sec_acc, batch_size=val_batch[0].shape[0], prog_bar=True)    
        # # self.log("import_sec_acc_count", import_sec_acc_count, batch_size=val_batch[0].shape[0], prog_bar=True)     
        # self.log("import_sec_recall", import_sec_recall, batch_size=val_batch[0].shape[0], prog_bar=True)  
        past_target = val_batch[0]
        self.val_metric_show(output,target,out_class,target_trend_class,vr_class_certainlys,
                                 target_vr_class,past_target=past_target,val_batch=val_batch,scaler=scaler,target_info=target_info,
                                 last_batch_index=last_batch_index,item_codes=item_codes)
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
    
    def compute_vr_class_acc(self,vr_class,vr_target,target_info=None,last_batch_index=None):
        """计算涨跌幅分类准确度"""

        # 总体准确率
        total_acc = torch.sum(vr_class==vr_target)/vr_target.shape[0]
        # 重点类别的准确率
        import_index = torch.nonzero(vr_class==CLASS_SIMPLE_VALUE_MAX)[:,0]
        import_sec_index = torch.nonzero(vr_class==CLASS_SIMPLE_VALUE_SEC)[:,0]
        import_acc_count = torch.sum(vr_target[import_index]==CLASS_SIMPLE_VALUE_MAX)
        import_sec_acc_count = torch.sum(vr_target[import_sec_index]==CLASS_SIMPLE_VALUE_SEC)
        if import_index.shape[0]==0:
            import_acc = torch.tensor(0.0)
        else:
            import_acc = import_acc_count/import_index.shape[0]
        if import_sec_index.shape[0]==0:
            import_sec_acc = torch.tensor(0.0)
        else:
            import_sec_acc = import_sec_acc_count/import_sec_index.shape[0]            
        # 重点类别的召回率    
        total_imp_cnt = torch.sum(vr_target==CLASS_SIMPLE_VALUE_MAX)
        total_imp_sec_cnt = torch.sum(vr_target==CLASS_SIMPLE_VALUE_SEC)
        if total_imp_cnt!=0:
            import_recall = import_acc_count/total_imp_cnt
        else:
            import_recall = 0
        if total_imp_sec_cnt!=0:
            import_sec_recall = import_sec_acc_count/total_imp_sec_cnt
        else:
            import_sec_recall = 0            
        # 最后一个批次的准确率统计
        # last_vr_target =  vr_target[last_batch_index]
        # last_vr_class = vr_class[last_batch_index]
        # last_import_index = torch.nonzero(last_vr_class==CLASS_SIMPLE_VALUE_MAX)[:,0]
        # last_import_acc_count = torch.sum(last_vr_target[last_import_index]==CLASS_SIMPLE_VALUE_MAX)
        # if last_import_index.shape[0]==0:
        #     last_import_acc = torch.tensor(0.0)
        # else:
        #     last_import_acc = last_import_acc_count/last_import_index.shape[0]
        # print("last_import_acc:{},last_import_index cnt:{}".format(last_import_acc,last_import_index.shape[0]))   
                 
        return total_acc,import_acc, import_acc_count,import_recall,import_sec_acc,import_sec_acc_count,import_sec_recall
        
       
    def compute_corr_metrics(self,output,target):
        num_outputs = output.shape[0]
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs).to(self.device)
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        corr_loss = self.pearson(output_sample.transpose(1,0),target.squeeze(-1).transpose(1,0))  
        corr_loss = torch.mean(1 - corr_loss)
        return corr_loss      
 
    def compute_ccc_metrics(self,output,target):
        return self.criterion.ccc_loss_comp(output,target)
       
    def compute_mse_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        mse_loss = self.mean_squared_error(output_sample, target[:,:,0])
        return mse_loss 

    def compute_value_diff_metrics(self,output,target):
        output_sample = torch.mean(output[:,:,0,:],dim=-1)
        output_be = output_sample[:,[0,-1]]
        target_be = target[:,[0,-1],0]
        mse_loss = self.mean_squared_error(output_be, target_be)
        return mse_loss 
    
    def _compute_loss(self, output, target):
        """重载父类方法"""
        
        (output_value,out_classify,vr_class) = output
        output_combine = (output_value.squeeze(dim=-1),out_classify,vr_class)
        
        if self.likelihood:
            # 把似然估计损失叠加到自定义多任务损失里
            loss_like = self.likelihood.compute_loss(output_value, target[1])
            if self.criterion is None:
                return loss_like
            mtl_loss = self.criterion(output_combine, target,outer_loss=loss_like)
            return mtl_loss
        else:
            return self.criterion(output_combine, target)

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
            list(map(id, self.last_classify_layer.parameters())) + \
            list(map(id, self.classify_vr_layer.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())       
        base_lr = self.optimizer_kwargs["lr"] 
        optimizer_kws["params"] = [
                    {'params': base_params},
                    {'params': self.classify_layer_1.parameters(), 'lr': base_lr*10},
                    {'params': self.classify_layer_2.parameters(), 'lr': base_lr*10},
                    {'params': self.last_classify_layer.parameters(), 'lr': base_lr*10},
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
                "monitor": lr_monitor if lr_monitor is not None else "val_loss",
            }
        else:
            return optimizer
             
    def predict_step(
        self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Sequence[TimeSeries]:
        """重载原方法，服务于自定义模式"""
        
        input_data_tuple,scalers, batch_input_series = batch[:-2], batch[-2], batch[-1]
        # scaler_map = self.build_scaler_map(scalers, batch_input_series)
        # number of individual series to be predicted in current batch
        num_series = input_data_tuple[0].shape[0]

        # number of times the input tensor should be tiled to produce predictions for multiple samples
        # this variable is larger than 1 only if the batch_size is at least twice as large as the number
        # of individual time series being predicted in current batch (`num_series`)
        batch_sample_size = min(
            max(self.pred_batch_size // num_series, 1), self.pred_num_samples
        )

        # counts number of produced prediction samples for every series to be predicted in current batch
        sample_count = 0

        # repeat prediction procedure for every needed sample
        batch_predictions = []
        while sample_count < self.pred_num_samples:

            # make sure we don't produce too many samples
            if sample_count + batch_sample_size > self.pred_num_samples:
                batch_sample_size = self.pred_num_samples - sample_count

            # stack multiple copies of the tensors to produce probabilistic forecasts
            input_data_tuple_samples = self._sample_tiling(
                input_data_tuple, batch_sample_size
            )

            # get predictions for 1 whole batch (can include predictions of multiple series
            # and for multiple samples if a probabilistic forecast is produced)
            batch_prediction,batch_pred_class,batch_vr_class = self._get_batch_prediction(
                self.pred_n, input_data_tuple_samples, self.pred_roll_size
            )

            # reshape from 3d tensor (num_series x batch_sample_size, ...)
            # into 4d tensor (batch_sample_size, num_series, ...), where dim 0 represents the samples
            out_shape = batch_prediction.shape
            batch_prediction = batch_prediction.reshape(
                (
                    batch_sample_size,
                    num_series,
                )
                + out_shape[1:]
            )
            # 还需要处理分类数据的shape
            out_class_shape = batch_pred_class.shape
            batch_pred_class = batch_pred_class.reshape(
                (
                    batch_sample_size,
                    num_series,
                )
                + out_class_shape[1:]
            )
            vr_class_shape = batch_vr_class.shape
            batch_vr_class = batch_vr_class.reshape(
                (
                    batch_sample_size,
                    num_series,
                )
                + vr_class_shape[1:]
            )            
            # save all predictions and update the `sample_count` variable
            batch_predictions.append(batch_prediction)
            sample_count += batch_sample_size

        # concatenate the batch of samples, to form self.pred_num_samples samples
        batch_predictions = torch.cat(batch_predictions, dim=0)
        batch_predictions = batch_predictions.cpu().detach().numpy()
        # self.predict_data_eval(batch_predictions,batch_input_series)
        
        batch_predictions_unscale = np.zeros([batch_predictions.shape[0],
                    batch_predictions.shape[1],batch_predictions.shape[2],batch_predictions.shape[3]])
        # 分别遍历每个系列，进行反向归一化
        for i in range(batch_predictions.shape[1]):
            scaler = scalers[i]
            for j in range(batch_predictions.shape[0]):
                batch_prediction = batch_predictions[j,i,:,:]
                batch_prediction_unscale = scaler.inverse_transform(batch_prediction)
                batch_predictions_unscale[j,i,:,:] = batch_prediction_unscale
            
        ts_forecasts = Parallel(n_jobs=self.pred_n_jobs)(
            delayed(_build_forecast_series)(
                [batch_prediction[batch_idx] for batch_prediction in batch_predictions_unscale],
                input_series,
                batch_pred_class[:,batch_idx,:],
                batch_vr_class[:,batch_idx,:,:]
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
                                
    def val_metric_show(self,output,target,out_class,target_class,vr_class,target_vr_class,past_target=None,val_batch=None,
                        scaler=None,target_info=None,last_batch_index=None,item_codes=None):
        
        names = ["output","target","price"]
        vr_class_certainlys = vr_class.cpu().numpy()
        vr_class_certainlys_imp_index = np.argwhere(vr_class_certainlys==CLASS_SIMPLE_VALUE_MAX)
        size = len(vr_class_certainlys_imp_index) if len(vr_class_certainlys_imp_index)<9 else 9
        # print("item_codes:",item_codes)
        loop_range = range(size)      
        # loop_range = last_batch_index
        df_all = global_var.get_value("dataset").df_all
        code_dict = {}
        idx = 0
        range_index = np.random.random_integers(0, vr_class_certainlys_imp_index.shape[0]-1,size)
        for idx,r_index in enumerate(range_index):
            s_index = vr_class_certainlys_imp_index[r_index,0]
            ts = target_info[s_index]
            scaler_sample = scaler[s_index]
            target_sample = target[s_index,:,:].cpu().numpy()
            target_sample = target_sample[:,0]
            past_target_sample = past_target[s_index,:,0].cpu().numpy()
            output_sample = output[s_index,:,0,:].cpu().numpy()
            target_class_sample = target_class[s_index].cpu().numpy()
            target_vr_class_sample = target_vr_class[s_index].cpu().numpy()
            vr_class_certainly = vr_class_certainlys[s_index]
            if vr_class_certainly!=CLASS_SIMPLE_VALUE_MAX:
                continue           
            output_class_sample = out_class[s_index].cpu().numpy()
            output_class_index = np.argmax(output_class_sample, axis=-1)
            pred_center_data_ori = get_np_center_value(output_sample)
            # 数值部分图形
            target_title = "rank:{},tar class:{},out class:{},tar_vr class:{}&{}".format(
                ts["item_rank_code"],target_class_sample,output_class_index,target_vr_class_sample,vr_class_certainly)
            # 补充画出前面的label数据
            target_combine_sample = np.concatenate((past_target_sample,target_sample),axis=-1)
            pad_data = np.array([0 for i in range(past_target_sample.shape[0])])
            pred_center_data = np.concatenate((pad_data,pred_center_data_ori),axis=-1)
            view_data = np.stack((pred_center_data,target_combine_sample),axis=0).transpose(1,0)
            win = "win_{}".format(idx)
            viz_input.viz_matrix_var(view_data,win=win,title=target_title,names=names)     
            # 针对分类部分，画出相应的缩放折线      
            pred_center_data = scaler_sample.inverse_transform(np.expand_dims(pred_center_data_ori,axis=0))
            pred_center_data = np.concatenate((pad_data,pred_center_data[0]),axis=-1)
            pred_center_data = np.expand_dims(pred_center_data,axis=0)
            target_sample = scaler_sample.inverse_transform(np.expand_dims(target_sample,axis=0))
            past_target_sample = scaler_sample.inverse_transform(np.expand_dims(past_target_sample,axis=0))
            target_combine_sample = np.concatenate((past_target_sample,target_sample),axis=-1)
            # 可以从全局变量中，通过索引获得实际价格
            price_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
                                    (df_all["instrument_rank"]==ts["item_rank_code"])]["label_ori"].values    
            price_target = np.expand_dims(price_target,axis=0)        
            view_data = np.concatenate((pred_center_data,target_combine_sample,price_target),axis=0).transpose(1,0)
            x_range = np.array([i for i in range(ts["start"],ts["end"])])
            if self.monitor is not None:
                instrument = self.monitor.get_group_code_by_rank(ts["item_rank_code"])
                datetime_range = self.monitor.get_datetime_with_index(instrument,ts["start"],ts["end"])
            else:
                instrument = ts["item_rank_code"]
                datetime_range = x_range
            target_title = "time range:{}/{} code:{},tar class:{},out class:{},tar_vr class:{}&{}".format(
                datetime_range[0],datetime_range[-1],instrument,target_class_sample,output_class_index,target_vr_class_sample,vr_class_certainly)
            viz_input_2.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=x_range)   
               
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
    
    def dynamic_build_training_data(self,item,dynaimc_range=3):
        d = np.random.random_integers(1000,1000+dynaimc_range*10)/1000
        past_covariate = item[1]
        future_target = item[-2]
        rtn_item = (item[0],past_covariate,item[2],item[3],item[4],item[5],item[6],future_target,item[8])
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
                if self.mode!="train":
                    batch.append(b)
                else:
                    # 增加不平衡类别的数量，随机小幅度调整数据
                    if target_class[1,0]==CLASS_SIMPLE_VALUE_MAX or target_class[1,0]==0:
                        max_cnt += 1
                        for i in range(1):
                            adj_max_cnt += 2
                            b_rebuild = self.dynamic_build_training_data(b)
                            batch.append(b_rebuild)
                    else:
                        # 随机减少大类数量
                        if np.random.randint(0,2)==1:
                            batch.append(b)
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