from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import _generate_new_dates
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
from darts.utils.utils import seq2series, series2seq

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import torch
import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
import joblib
from sklearn.preprocessing import MinMaxScaler

from .series_data_utils import StatDataAssis
from cus_utils.encoder_cus import StockNormalizer,unverse_transform_slope_value
from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import compute_price_class,normalization,pairwise_compare,comp_max_and_rate
from cus_utils.metrics import compute_cross_metrics,compute_vr_metrics
import cus_utils.global_var as global_var
from tft.class_define import CLASS_SIMPLE_VALUES,CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUE_SEC,SLOPE_SHAPE_SMOOTH,CLASS_LAST_VALUE_MAX
from darts_pro.data_extension.custom_tcn_model import LSTMReg,TargetDataReg
from darts_pro.data_extension.custom_base_model import BaseMixModule
from losses.mtl_loss import TripletLoss,UncertaintyLoss

viz_result_suc = TensorViz(env="train_result_suc")
viz_result_fail = TensorViz(env="train_result_fail")
viz_result_nor = TensorViz(env="train_result_nor")
viz_input_aug = TensorViz(env="data_train_aug")
viz_target = TensorViz(env="data_target")

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
    
    
class _CusModule(BaseMixModule):
    
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
        train_sample=None,
        model_type="tft",
        **kwargs,
    ):
        super().__init__(
        output_dim,
        variables_meta_array,
        num_static_components,
        hidden_size,
        lstm_layers,
        num_attention_heads,
        full_attention,
        feed_forward,
        hidden_continuous_size,
        categorical_embedding_sizes,
        dropout,
        add_relative_index,
        norm_type,
        use_weighted_loss_func=use_weighted_loss_func,
        past_split=past_split,
        filter_conv_index=filter_conv_index,
        loss_number=loss_number,
        device=device,
        train_sample=train_sample,
        model_type=model_type,
        **kwargs)
        
 
    def forward(
        self, x_in: Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
        future_target,
        scaler,
        past_target=None,
        target_info=None,
        optimizer_idx=-1
    ) -> torch.Tensor:
        """重载训练方法，加入分类模式"""
        out_total = []
        out_class_total = []
        batch_size = x_in[1].shape[0]
        # 分别单独运行模型
        for i,m in enumerate(self.sub_models):
            # 根据优化器编号匹配计算
            if optimizer_idx==i or optimizer_idx>=len(self.sub_models) or optimizer_idx==-1:
                # 根据配置，不同的模型使用不同的过去协变量
                past_convs_item = x_in[0][i]
                x_in_item = (past_convs_item,x_in[1],x_in[2])
                out = m(x_in_item)
                vr_layer = self.vr_layers[i]
                out_class = vr_layer(out[:,:,0,0])
                # out_class = torch.ones([batch_size,self.output_chunk_length,1]).to(self.device)
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
 
    def _construct_classify_layer(self, input_dim, output_dim=4,hidden_dim=64,device=None):
        """使用全连接进行分类数值输出
          Params
            layer_num： 层数
            input_dim： 序列长度
            output_dim： 类别数
        """
        
        class_layer = nn.Linear(input_dim,output_dim)
        class_layer = class_layer.cuda(device)
        return class_layer

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """performs the training step"""
        
        train_batch = self.filter_batch_by_condition(train_batch,filter_conv_index=self.filter_conv_index)
        loss,detail_loss,output = self.training_step_real(train_batch, batch_idx) 
        return loss
    
    def training_step_real(self, train_batch, batch_idx) -> torch.Tensor:
        """包括第一及第二部分数值数据,以及分类数据"""

        # 收集目标数据用于分类
        future_target = train_batch[-2]
        # 目标数据里包含分类信息
        scaler_tuple,target_class,target,target_info = train_batch[5:]       
        scaler = [s[0] for s in scaler_tuple] 
        input_batch = self._process_input_batch(train_batch[:5])
        target_class = target_class[:,:,0]     
        # 给criterion对象设置epoch数量。用于动态loss策略
        if self.criterion is not None:
            self.criterion.epoch = self.epochs_trained   
        total_loss = torch.tensor(0.0).to(self.device)
        ce_loss = None
        for i in range(len(self.past_split)):
            y_transform = None 
            (output,vr_class,tar_class) = self(input_batch,future_target,scaler,past_target=train_batch[0],optimizer_idx=i,target_info=target_info)
            loss,detail_loss = self._compute_loss((output,vr_class,tar_class), (target,target_class,target_info,y_transform),optimizers_idx=i)
            (corr_loss_combine,triplet_loss_combine,extend_value) = detail_loss 
            self.log("train_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=train_batch[0].shape[0], prog_bar=False)  
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
        self.log("lr0",self.trainer.optimizers[1].param_groups[0]["lr"], batch_size=train_batch[0].shape[0], prog_bar=True)                
        # 手动维护global_step变量  
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()
        return total_loss,detail_loss,output

    def build_focus_data(self,output_ori,past_target_ori,target_info=None,scalers=None):
        # 根据序列预测输出结果，整合并形成二次特征

        if self.trainer.state.stage==RunningStage.TRAINING:
            output = output_ori.detach().cpu().numpy()
            past_target = past_target_ori.detach().cpu().numpy()
        else:
            output = output_ori.cpu().numpy()
            past_target = past_target_ori.cpu().numpy()
        
        combine_data = np.concatenate((past_target,output),axis=1)
            
        # output = np.stack([scalers[i].inverse_transform(output[i]) for i in range(len(scalers))])  
        # past_target = np.stack([scalers[i].inverse_transform(past_target[i]) for i in range(len(scalers))])
        #
        # first_target = past_target[:,:,0]
        # first_output = output[:,:,0]
        # f1_range_score = (first_output[:,-1]-first_output[:,0])/first_output[:,0]
        # f2_mean_score = np.mean(first_output,axis=1)
        #
        # second_target = past_target[:,:,1]
        # second_output = output[:,:,1]
        # s1_range_score = (second_output[:,-1]-second_output[:,0])/second_output[:,0]
        # second_max = np.max(second_output,axis=-1)
        # second_min = np.min(second_output,axis=-1)     
        # s2_max_min_score = (second_output[:,-1] - second_output[:,0])/(second_max - second_min)   
        #
        # third_target = past_target[:,:,2]
        # third_output = output[:,:,2]            
        # t1_min_score = (third_output[:,0] - np.min(third_target,axis=1)) / np.min(third_target,axis=1)
        # t2_min_score = (third_output[:,-1] - np.min(third_target,axis=1)) / np.min(third_target,axis=1)
        # t3_range_score = (third_output[:,-1]-third_output[:,0])/third_output[:,0]
        #
        # total_score = np.stack([f1_range_score,f2_mean_score,s1_range_score,s2_max_min_score,t1_min_score,t2_min_score,t3_range_score],axis=-1)
        # total_score = normalization(total_score,mode="numpy")
        return torch.Tensor(combine_data).to(self.device)
        
    def on_train_epoch_start(self):
        self.loss_data = []
           
    def on_train_epoch_end(self):
        self.custom_histogram_adder()
        
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
            total_cnt = ins_unique.values[:,1].sum()
            for i in range(4):
                cnt_values = ins_unique[ins_unique.index==i].values
                if cnt_values.shape[0]==0:
                    cnt = 0
                else:
                    cnt = cnt_values[0,1]
                rate = cnt/total_cnt
                # print("cnt:{} with score:{},total_cnt:{},rate:{}".format(cnt,i,total_cnt,rate))
                self.log("score_{} rate".format(i), rate, prog_bar=True) 
            self.log("total cnt", total_cnt, prog_bar=True)  
        self.log("total_imp_cnt", self.total_imp_cnt, prog_bar=True)  
        
    def validation_step(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""
        
        loss,detail_loss,output = self.validation_step_real(val_batch, batch_idx)
        return loss
                                 
    def validation_step_real(self, val_batch, batch_idx) -> torch.Tensor:
        """训练验证部分"""

        input_batch = self._process_input_batch(val_batch[:5])
        # 收集目标数据用于分类
        scaler_tuple,target_class,future_target,target_info = val_batch[5:]  
        scaler = [s[0] for s in scaler_tuple]
        (output,vr_class,vr_class_list) = self(input_batch,future_target,scaler,past_target=val_batch[0],target_info=target_info,optimizer_idx=-1)
        
        raise_range_batch = np.expand_dims(np.array([ts["raise_range"] for ts in target_info]),axis=-1)
        y_transform = MinMaxScaler().fit_transform(raise_range_batch)    
        y_transform = raise_range_batch  
        y_transform = torch.Tensor(y_transform).to(self.device)  
              
        past_target = val_batch[0]
        past_covariate = val_batch[1]
        target_class = target_class[:,:,0]
        target_vr_class = target_class[:,0].cpu().numpy()
        output_combine = [output_item[:,:,0,0] for output_item in output]
        output_combine = torch.stack(output_combine,dim=2).cpu().numpy()
        output_inverse = self.get_inverse_data(output_combine,target_info=target_info,scaler=scaler)
        whole_target = np.concatenate((past_target.cpu().numpy(),future_target.cpu().numpy()),axis=1)
        target_inverse = self.get_inverse_data(whole_target,target_info=target_info,scaler=scaler)
        # 全部损失
        loss,detail_loss = self._compute_loss((output,vr_class,vr_class_list), (future_target,target_class,target_info,y_transform),optimizers_idx=-1)
        (corr_loss_combine,triplet_loss_combine,extend_value) = detail_loss
        self.log("val_loss", loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        for i in range(len(corr_loss_combine)):
            self.log("val_corr_loss_{}".format(i), corr_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_triplet_loss_{}".format(i), triplet_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("classify_loss_{}".format(i), classify_loss_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
            # self.log("val_acc_{}".format(i), corr_acc_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("val_ce_loss", ce_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("value_diff_loss", value_diff_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("last_vr_loss", last_vr_loss, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        # 涨跌幅度类别的准确率
        vr_class_sf = [self.build_vr_class_cer(vc).cpu().numpy() for vc in vr_class_list]
        self.metric_data(output_combine,future_target.cpu().numpy(),target_vr_class)
        import_price_result = self.compute_real_class_acc(output_inverse=output_inverse,target_vr_class=target_vr_class,
                    vr_class=vr_class_sf,output_data=output_combine,target_info=target_info,target_inverse=target_inverse)   
        total_imp_cnt = np.where(target_vr_class==3)[0].shape[0]
        if self.total_imp_cnt==0:
            self.total_imp_cnt = total_imp_cnt
        else:
            self.total_imp_cnt += total_imp_cnt
        
        past_target = val_batch[0]
        self.val_metric_show(output,future_target,target_vr_class,output_inverse=output_inverse,vr_class=vr_class,
                             target_inverse=target_inverse,target_info=target_info,import_price_result=import_price_result,past_covariate=past_covariate,
                            batch_idx=batch_idx)
        
        # # 累加结果集，后续统计   
        if self.import_price_result is None:
            self.import_price_result = import_price_result    
        else:
            if import_price_result is not None:
                import_price_result_array = import_price_result.values
                # 修改编号，避免重复
                import_price_result_array[:,0] = import_price_result_array[:,0] + batch_idx*1000
                import_price_result_array = np.concatenate((self.import_price_result.values,import_price_result_array))
                self.import_price_result = pd.DataFrame(import_price_result_array,columns=self.import_price_result.columns)        
        
        # for i in range(3):
        #     self.log("triplet acc_{}".format(i), corr_acc_combine[i], batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("output_imp_class_acc_cnt", output_imp_class_acc_cnt, batch_size=val_batch[0].shape[0], prog_bar=True)
        # self.log("output_imp_class_acc", output_imp_class_acc, batch_size=val_batch[0].shape[0], prog_bar=True)
        
        return loss,detail_loss,output
 
    
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
    
    def build_import_index(self,output_inverse=None,vr_class=None,target_inverse=None,output_data=None):  
        """生成涨幅达标的预测数据下标"""
        
        output_label_inverse = output_inverse[:,:,0] 
        # output_second_inverse = output_inverse[:,:,1]
        output_second_inverse = output_inverse[:,:,1]
        output_third_inverse = output_inverse[:,:,2]
                 
        # output_import_index_bool = slope_out_compute>label_threhold
        # target_size = output_inverse.shape[-1]
        # combine_bool = np.sum(output_import_index_bool,axis=1)>=(target_size-1)
        # import_index = np.where(combine_bool)[0]
        
        # 价格指标整体需要有上涨幅度
        range_compute = (output_inverse[:,-1,:]  - output_inverse[:,0,:])/np.abs(output_inverse[:,0,:])*100
        # slope_out_compute = np.sum(output_label_inverse,axis=-1)
        first_index_bool = range_compute>10
        first_his_tar = target_inverse[:,:self.input_chunk_length,0]
        # macd数值在0以上
        first_index_bool = (np.mean(output_label_inverse,axis=1)>0)
        # macd历史均值在0以上
        # first_index_bool = first_index_bool & (np.mean(first_his_tar,axis=1)>0)
        # 接近或超过前期高点
        # first_max = np.max(first_his_tar,axis=1)
        # first_index_bool = first_index_bool & (((output_label_inverse[:,-1]-first_max)>0) | ((np.abs(output_label_inverse[:,-1]-first_max))/first_max<0.2))
        
        # 辅助指标判断
        second_his_tar = target_inverse[:,:self.input_chunk_length,1]
        # 上涨趋势
        # peak_list = [find_peaks(second_his_tar[i], height=2,distance=5)[0] for i in range(second_his_tar.shape[0])]
        # peak_data_list = [second_his_tar[i,peak_list[i]] for i in range(second_his_tar.shape[0])]
        # peak_data_list = np.array([peak_data_item[-1] for peak_data_item in peak_data_list])
        # # 结束值超过最近一个波峰
        # second_index_bool = ((output_second_inverse[:,-1]-peak_data_list)/peak_data_list*100>1) 
        # 超过一定的涨幅
        second_index_bool = (((output_second_inverse[:,-1]-output_second_inverse[:,0])/output_second_inverse[:,0]*100)>5)
        # 整体上升幅度与振幅差值较小
        second_max = np.max(output_second_inverse,axis=-1)
        second_min = np.min(output_second_inverse,axis=-1)  
        second_index_bool = second_index_bool & (((output_second_inverse[:,-1] - output_second_inverse[:,0])/(second_max - second_min))>0.3)
        
        # qtlu指标下降趋势
        third_his_tar = target_inverse[:,:self.input_chunk_length,2]
        # 不能超过前期高点
        third_index_bool = (output_third_inverse[:,-1] - np.max(third_his_tar,axis=1)<0)
        # 历史峰值计算
        peak_list = [find_peaks(third_his_tar[i], height=0.5,distance=5)[0] for i in range(third_his_tar.shape[0])]
        peak_data_list = [third_his_tar[i,peak_list[i]] for i in range(third_his_tar.shape[0])]
        peak_data_list = np.array([peak_data_item[-1] for peak_data_item in peak_data_list])
        # 不超过最近峰值
        third_index_bool = third_index_bool & (output_third_inverse[:,-1]<peak_data_list)
        # 不能超过前期低点太多
        third_index_bool = (output_third_inverse[:,-1] - np.min(third_his_tar,axis=1)) < 0.015
        third_index_bool = third_index_bool & ((output_third_inverse[:,0] - np.min(third_his_tar,axis=1)) < 0.02)        
        # 涨幅小或者下跌
        third_index_bool = ((output_third_inverse[:,-1] - output_third_inverse[:,0]) < 0.01)  
        # 最后一段上涨
        # third_index_bool = third_index_bool & ((output_third_inverse[:,-1] - output_third_inverse[:,-2])>0)        
        # third_index_bool = third_index_bool & (((output_third_inverse[:,-1] - output_third_inverse[:,0])/(third_max - third_min))>0.5)
        import_index_bool = third_index_bool & second_index_bool
        import_index = np.where(import_index_bool)[0]
        
        import_index = np.where((vr_class[0]>=2) & (vr_class[1]>=2))[0]
        # 直接使用分类
        # third_index_bool = (third_class==CLASS_SIMPLE_VALUE_MAX)
        # 使用knn分类模式判别
        # print("begin knn_clf")
        # predicted_labels = knn_clf.predict(output_inverse)
        # print("begin end")
        # import_index_bool = predicted_labels==3
        
        # 综合判别
        # second_index = np.argsort(-range_compute[:,1],axis=0)
        # second_index = second_index[:100]
        # # second_index = np.where(range_compute[:,1]>0)
        # third_index = np.argsort(range_compute[:,2],axis=0)
        # third_index = third_index[:100]
        # # third_index = np.where(range_compute[:,2]<0)
        # # import_index = second_index 
        # import_index = np.intersect1d(third_index,second_index)
        #
        # clu_data = global_var.get_value("imp_clu_data")
        # clu_imp_idx = []
        # 通过计算与目标值的距离比较，来决定是否入选
        # import_index = torch.argsort(torch.sum(vr_class,dim=1),descending=True)[:50].cpu().numpy()
        return import_index        
    
    def compare_with_clu(self,output_data,target_data):
        
        comp_data = pairwise_compare(torch.Tensor(target_data).to(self.device),torch.Tensor(output_data).to(self.device),
                                     distance_func=self.criterion.ccc_distance)
        distance_mean = torch.mean(comp_data,dim=0)
        return distance_mean
    
    def collect_result(self,import_index,target_vr_class=None,target_info=None): 
          
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
            ts = target_info[imp_idx]
            price_array = ts["price_array"][self.input_chunk_length:]
            p_taraget_class = compute_price_class(price_array,mode="first_last")
            import_price_result.append([imp_idx,ts["item_rank_code"],p_taraget_class])
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
        
        return import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_result
    
               
    def compute_real_class_acc(self,target_info=None,output_inverse=None,output_data=None,target_vr_class=None,vr_class=None,target_inverse=None):
        """计算涨跌幅分类准确度"""
        
        # 使用分类判断s
        import_index = self.build_import_index(output_inverse=output_inverse,vr_class=vr_class,
                             target_inverse=target_inverse,output_data=output_data)
        import_acc, import_recall,import_price_acc,import_price_nag,price_class,import_price_result = \
            self.collect_result(import_index, target_vr_class, target_info)
        
        return import_price_result
    
    def metric_data(self,output,target,target_class):
        """数据准确性检验"""
        
        import_index = np.where(target_class==CLASS_SIMPLE_VALUE_MAX)[0]
        neg_index = np.where(target_class==0)[0]
        output_data = output[import_index]
        anchor_data = target[import_index]
        neg_data = target[neg_index]
        size = neg_data.shape[0] if neg_data.shape[0]<output_data.shape[0] else output_data.shape[0]
        
        # for i in range(3):
        #     loss_comp = UncertaintyLoss().corr_loss_comp(torch.Tensor(output_data[:,:,i]), torch.Tensor(anchor_data[:,:,i]))
        #     loss_neg_comp = UncertaintyLoss().corr_loss_comp(torch.Tensor(output_data[:size,:,i]), torch.Tensor(neg_data[:size,:,i]))
        #     print("loss_comp_{}:{},loss_neg_comp:{}".format(i,loss_comp,loss_neg_comp))
            
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""
        
        (output_value,vr_class,tar_class) = output
        output_value = [output_value_item.squeeze(dim=-1) for output_value_item in output_value]
        output_combine = (output_value,vr_class,tar_class)
        
        return self.criterion(output_combine, target,optimizers_idx=optimizers_idx)

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.classify_vr_layer.named_parameters():
            global_step = self.current_epoch
            if params is not None:
                self.logger.experiment.add_histogram(name + "_weights",params,global_step)
            if params.grad is not None:
                self.logger.experiment.add_histogram(name + "_grad",params.grad,global_step)
        

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
                                
    def val_metric_show(self,output,target,target_vr_class,output_inverse=None,target_inverse=None,
                        vr_class=None,past_covariate=None,target_info=None,import_price_result=None,batch_idx=0):
        
        if import_price_result is None or import_price_result.shape[0]==0:
            return
        # if self.trainer.state.stage==RunningStage.SANITY_CHECKING:
        #     return 
        
        dataset = global_var.get_value("dataset")
        df_all = dataset.df_all
        names = ["pred","label","price","obv_output","obv_tar","cci_output","cci_tar"]        
        names = ["price","macd_output","macd","rank_output","rank","qtlu_output","qtlu"]          
        result = []
        
        # viz_result_suc.remove_env()
        # viz_result_nor.remove_env()
        # viz_result_fail.remove_env()
              
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
        
        total_index = 0                    
        for result,group in res_group:
            r_index = -1
            unique_group = group.drop_duplicates(subset=['instrument'], keep='first')
            for index, row in unique_group.iterrows():
                r_index += 1
                total_index += 1
                s_index = row["imp_index"]
                ts = target_info[s_index]
                pred_data = output_inverse[s_index]
                pred_center_data = pred_data[:,0]
                pred_second_data = pred_data[:,1]
                target_item = target_inverse[s_index]
                pred_third_data = pred_data[:,2]
                # 可以从全局变量中，通过索引获得实际价格
                # df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
                #                         (df_all["instrument_rank"]==ts["item_rank_code"])]            
                win = "win{}_{}".format(ts["instrument"],ts["future_start"])
                if r_index>15:
                    break
                if result==CLASS_SIMPLE_VALUE_MAX:                 
                    viz = viz_result_suc
                elif result==0:                 
                    viz = viz_result_fail   
                else:
                    viz = viz_result_nor  
                # self.draw_row(pred_center_data, pred_second_data, pred_third_data,ts=ts,target_item=target_item, names=names,viz=viz,win=win)
                      
    def draw_row(self,pred_center_data,pred_second_data,pred_third_data,target_item=None,ts=None,names=None,viz=None,win="win"):
        """draw one line"""
        
        names = ['MACD','QTLUMA5','RANKMA5','MACD_OUTPUT','RANKMA5_OUTPUT','QTLUMA5_OUTPUT']
        
        price_array = ts["price_array"]
        # 补充画出前面的label数据
        pad_data = np.array([0 for i in range(self.input_chunk_length)])
        pred_first_data = np.concatenate((pad_data,pred_center_data),axis=-1)
        pred_second_data = np.concatenate((pad_data,pred_second_data),axis=-1)
        pred_third_data = np.concatenate((pad_data,pred_third_data),axis=-1)
        view_data = np.expand_dims(price_array,axis=-1)
        first_target = target_item[:,0]    
        first_target = np.expand_dims(first_target,axis=-1)    
        second_target = target_item[:,1] 
        second_target = np.expand_dims(second_target,axis=-1)  
        third_target = target_item[:,2] 
        third_target = np.expand_dims(third_target,axis=-1)    
        view_data = np.concatenate((view_data,first_target),axis=1) 
        view_data = np.concatenate((view_data,second_target),axis=1)
        view_data = np.concatenate((view_data,third_target),axis=1)
        view_data = np.concatenate((view_data,np.expand_dims(pred_first_data,axis=-1)),axis=1)           
        view_data = np.concatenate((view_data,np.expand_dims(pred_second_data,axis=-1)),axis=1)    
        view_data = np.concatenate((view_data,np.expand_dims(pred_third_data,axis=-1)),axis=1)    
        view_data = view_data[:,1:]
        # np.save("custom/data/asis/view_data/{}.npy".format(win),view_data)
        # view_data = np.concatenate((view_data[:,2:3],view_data[:,3:4],view_data[:,5:6],view_data[:,6:7]),axis=1)
        x_range = np.array([i for i in range(ts["start"],ts["end"])])
        if self.monitor is not None:
            instrument = self.monitor.get_group_code_by_rank(ts["item_rank_code"])
            datetime_range = self.monitor.get_datetime_with_index(instrument,ts["start"],ts["end"])
        else:
            instrument = ts["instrument"]
            datetime_range = x_range
        price_class_item = compute_price_class(price_array[self.input_chunk_length:],mode="first_last")
        # result.append({"instrument":instrument,"datetime":df_target["datetime"].dt.strftime('%Y%m%d').values[0]})
        target_title = "time range:{}/{} code:{},price class:{}".format(
            ts["future_start"],ts["future_end"],instrument,price_class_item)
                  
        viz.viz_matrix_var(view_data,win=win,title=target_title,names=names,x_range=datetime_range)   
    
                            
    def build_scaler_map(self,scalers, batch_input_series,group_column="instrument_rank"):
        scaler_map = {}
        for i in range(len(scalers)):
            series = batch_input_series[i]
            group_col_val = series.static_covariates[group_column].values[0]
            scaler_map[group_col_val] = scalers[i]
        return scaler_map

class _TFTModuleBatch(_CusModule):
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
        train_sample=None,
        **kwargs,
    ):
        super().__init__(output_dim,variables_meta_array,num_static_components,hidden_size,lstm_layers,num_attention_heads,
                                    full_attention,feed_forward,hidden_continuous_size,
                                    categorical_embedding_sizes,dropout,add_relative_index,norm_type,past_split=past_split,
                                    use_weighted_loss_func=use_weighted_loss_func,train_sample=train_sample,
                                    device=device,**kwargs)  
        self.lr_freq = {"interval":"epoch","frequency":1}
        
        self.train_filepath = "{}/train_output_batch.pickel".format(batch_file_path)
        self.valid_filepath = "{}/valid_output_batch.pickel".format(batch_file_path)
        # 使用具备梯度额参数作为聚类簇心，供聚类损失使用,形状为:类别数*预测时间步长
        self.cluster_center = [torch.nn.Parameter(torch.rand(len(CLASS_SIMPLE_VALUES.keys()), 
                                    kwargs["output_chunk_length"]).to(device)) for i in range(len(past_split))]
    
    def on_train_start(self): 
        super().on_train_start()
        self.train_output_flag = False
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
      
    
    def _compute_loss(self, output, target,optimizers_idx=0):
        """重载父类方法"""
        
        (output_value,vr_class,tar_class) = output
        output_value = [output_value_item.squeeze(dim=-1) for output_value_item in output_value]
        output_combine = (output_value,vr_class,tar_class)
        # 添加聚类簇心参数
        return self.criterion(output_combine, target,cluster_centers=self.cluster_center,optimizers_idx=optimizers_idx)  
        
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