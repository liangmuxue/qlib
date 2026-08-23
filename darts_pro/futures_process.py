# 使用darts架构的TFT模型，定制化numpy数据集模式

from __future__ import division
from __future__ import print_function

import datetime
import os
import numpy as np
from collections import Counter
import pandas as pd
import pickle
import copy
import math
from qlib.utils import get_or_create_path
from qlib.log import get_module_logger
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import shap
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer.states import RunningStage
            
from darts.metrics import mape
from darts.models import TFTModel
from darts import TimeSeries, concatenate
from torchmetrics import (
    PearsonCorrCoef,
    MetricCollection,
)
from qlib.utils import flatten_dict, get_callable_kwargs, init_instance_by_config
from qlib.contrib.model.pytorch_utils import count_parameters
from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP

from cus_utils.tensor_viz import TensorViz
from darts_pro.data_extension.futures_model import FuturesModel,FuturesIndustryModel
from darts_pro.tft_futures_dataset import TFTFuturesDataset

from cus_utils.common_compute import compute_price_class
import cus_utils.global_var as global_var
from cus_utils.db_accessor import DbAccessor
from trader.utils.date_util import get_tradedays_dur,get_tradedays,get_next_month
from .tft_process_dataframe import TftDataframeModel 
from darts_pro.data_extension.series_data_utils import StatDataAssis
from sklearn.preprocessing import MinMaxScaler
from darts_pro.data_extension.futures_industry_dataset import FuturesIndustryDataset
from tft.class_define import CLASS_SIMPLE_VALUES,get_simple_class
from captum.attr import LayerGradientShap,GradientShap,IntegratedGradients

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from cus_utils.log_util import AppLogger
logger = AppLogger()

class FuturesProcessModel(TftDataframeModel):

    def fit(
        self,
        dataset: TFTFuturesDataset,
    ):
        self.init_env(dataset)
        
        if self.type.startswith("fit_futures_togather"):
            self.fit_futures_togather(dataset)
            return   
        if self.type.startswith("fit_futures_industry"):
            self.fit_futures_industry(dataset)
            return        
        if self.type.startswith("fit_futures_bidi"):
            self.fit_futures_bidi(dataset)
            return    
        if self.type.startswith("fit_futures_trans"):
            self.fit_futures_trans(dataset)
            return         
        if self.type.startswith("fit_futures_tcn"):
            self.fit_futures_tcn(dataset)
            return                  
        if self.type.startswith("pred_futures_industry"):
            self.fit_futures_industry(dataset)
            return            
        if self.type.startswith("pred_futures_togather"):
            self.fit_futures_togather(dataset)
            return     
        if self.type.startswith("pred_futures_bidi"):
            self.fit_futures_bidi(dataset)
            return     
        if self.type.startswith("pred_futures_tcn"):
            self.fit_futures_tcn(dataset)
            return    
        if self.type.startswith("pred_futures_trans"):
            self.fit_futures_trans(dataset)
            return                      
        if self.type.startswith("analysis_model"):
            self.analysis_model(dataset)   
            return     
        if self.type=="predict":
            self.predict(dataset)   
            return   
        if self.type=="predict_indus_and_detail":
            self.predict_indus_and_detail(dataset)   
            return       
        if self.type.startswith("build_val_result"):
            self.build_val_result(dataset)
            return    
        if self.type.startswith("batch_pred_bidi"):
            self.batch_pred_bidi(dataset)
            return                                      
        print("Do Nothing")

    def init_env(self,dataset):
        
        self.dataset = dataset
        dataset.provider_file = os.path.join(self.optargs["provider_uri"],"instruments",self.optargs["market"]+".txt")
        global_var.set_value("dataset", dataset)
        viz_data = TensorViz(env="viz_data")
        viz_result = TensorViz(env="viz_result")
        viz_result_detail = TensorViz(env="viz_result_detail")
        viz_result_ext = TensorViz(env="viz_result_ext")
        global_var.set_value("viz_data",viz_data)
        global_var.set_value("viz_result",viz_result)
        global_var.set_value("viz_result_detail",viz_result_detail)
        global_var.set_value("viz_result_ext",viz_result_ext)
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)

    def fit_futures_trans(
        self,
        dataset: TFTFuturesDataset,
    ):
        
        # torch.set_float32_matmul_precision('medium')
        
        ori_model,build_data,outer_params = self.prepare_model_env(dataset)
        train_data = build_data['train']
        val_data = build_data['val']
        test_data = build_data['test']
        
        train_series_transformed,past_convariates_train,future_convariates_train = train_data
        val_series_transformed,past_convariates_val,future_convariates_val = val_data
        test_series_transformed,past_convariates_test,future_convariates_test = test_data        
        
        if self.type=="pred_futures_trans":  
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model,train_loader,val_loader,test_loader = \
            ori_model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                    test_series=test_series_transformed,test_past_covariates=past_convariates_test,test_future_covariates=future_convariates_test,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=False)  
            ori_model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            ori_model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            ori_model.model.set_outer_params(outer_params) 
            trainer.validate(model=model,dataloaders=val_loader)
            loss_result = ori_model.model.loss_result
            # print("ins_total loss:{}".format(loss_result['ins_total']))
        elif self.type=="fit_futures_trans":
            # 设置外部hook，每训练一个轮次后，进行测试
            
            class ExternalTrainHook(Callback):
                """外部训练生命周期钩子，完全脱离LightningModule"""
                def __init__(self,trainer,test_loader,epoch_num=0,log_folder=None,caller=None,train_loader=None,val_loader=None):
                    self.caller = caller
                    self.trainer = trainer
                    self.test_loader = test_loader
                    self.begin = False
                    logger = pl_loggers.TensorBoardLogger(save_dir=log_folder, default_hp_metric=False,name="", version="val_logs")
                    trainer.logger = logger     
                    self.epoch_num = epoch_num     
                    self.train_loader = train_loader
                    self.val_loader = val_loader

                def clone_pl_model(self,src_model) -> pl.LightningModule:
                    # 用原超参新建实例
                    new_model = type(src_model)(**src_model.hparams)
                    # 复制网络权重
                    new_model.load_state_dict(src_model.state_dict())
                    new_model.set_outer_params(outer_params) 
                    # 同步设备
                    new_model = new_model.to(src_model.device)
                    # 对于内部nn模型，直接使用deepcopy
                    new_model.sub_models = copy.deepcopy(src_model.sub_models)
                    # 单独设置外部维护参数
                    new_model.fur_scale = src_model.fur_scale        
                                
                    return new_model
          
                def on_validation_end(self, trainer, pl_module):
                    tb = self.trainer.logger.experiment
                    model = self.clone_pl_model(pl_module)
                    model.set_outer_params({'outer_call':True}) 
                    
                    self.trainer.validate(model=model,dataloaders=self.test_loader)
                    loss_result = model.loss_result
                    tb.add_scalar("epoch", trainer.current_epoch, trainer.current_epoch)  
                    for key in loss_result:
                        if self.caller.optargs["target_mode"][0]==3 and key.startswith("cate_main"):
                            continue                           
                        print("{} loss:{}".format(key,loss_result[key]))
                        tb.add_scalar(key, loss_result[key], trainer.current_epoch)   
                    rate_total = model.rate_total                               
                    for key in rate_total.columns:
                        if self.caller.optargs["target_mode"][0]==2:
                            if key.startswith("dist"):
                                continue
                            if key.startswith("trend"):
                                continue             
                            if key=='win_rate' or key=='yield_rate' or key=='total_cnt':
                                continue   
                        else:
                            if key.startswith("trend") or key=='total_cnt' or key.startswith("dist"):
                                continue      
                            if key.startswith("val_"):
                                continue          
                        item = rate_total[key].values[0]
                        tb.add_scalar("rate/{}".format(key), item, trainer.current_epoch)
                        if key=='cate_yield':
                            print("cate_yield:{}".format(item))
                    if self.caller.optargs["target_mode"][0]==5:
                        print("anno_yield:",model.anno_yield)
                        tb.add_scalar("rate/anno_yield", model.anno_yield, trainer.current_epoch)
                    
                    # 根据归因数据，动态调整未来协变量缩放参数
                    new_fur_scale = pl_module.sub_models[0].trans_model_decoder.fur_scale
                    if (trainer.current_epoch%6)==0 and trainer.current_epoch>1:
                        # 调用归因分析方法，取得归因结果
                        # ori_model.model = model
                        # model_env = (ori_model,build_data,outer_params,self.train_loader,self.val_loader)
                        model = self.clone_pl_model(pl_module)
                        rtn_data = self.caller.ind_analysis(self.train_loader.dataset,self.val_loader.dataset,pl_module=model,train_loader=self.train_loader,val_loader=self.val_loader)
                        # 重点关注过去业务协变量和未来时间协变量的权重关系，只看验证集
                        past_convs_weights = rtn_data['past_convs'][1]
                        future_single_emb_weights = rtn_data['future_single_emb'][1]
                        # 未来时间协变量的归因权重不能超出过去业务协变量的一定比例,如果超出，则调整缩放参数
                        scale_threhold = [8,10]
                        scale_value = past_convs_weights/future_single_emb_weights
                        if scale_value < scale_threhold[0]:
                            new_fur_scale = new_fur_scale * scale_value /scale_threhold[0]
                        if scale_value > scale_threhold[1]:
                            new_fur_scale = new_fur_scale * scale_value /scale_threhold[1]                           
                        pl_module.sub_models[0].trans_model_decoder.set_fur_scale(new_fur_scale) 
                        pl_module.set_net_parasms({'fur_scale':new_fur_scale})
                        
                    print("model.fur_scale:{}".format(new_fur_scale))
                    tb.add_scalar('fur_scale', new_fur_scale, trainer.current_epoch)                         
                        
            trainer,model_inner,train_loader,val_loader,_ = \
            ori_model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                    test_series=None,test_past_covariates=None,test_future_covariates=None,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=True)   
            trainer_test,_,_,_,test_loader = \
            ori_model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                    test_series=test_series_transformed,test_past_covariates=past_convariates_test,test_future_covariates=future_convariates_test,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=True)  
            
            log_folder = os.path.join(self.optargs["work_dir"],self.optargs["model_name"])
            hook = ExternalTrainHook(trainer_test,test_loader,log_folder=log_folder,epoch_num=trainer.current_epoch,caller=self,train_loader=train_loader,val_loader=val_loader)
            trainer.callbacks.append(hook)
            ori_model.train(trainer,model_inner,train_loader,val_loader)
    
    def prepare_model_env(self,dataset):
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
        
        # 生成tft时间序列数据集,包括目标数据、协变量等
        global_var.set_value("trend_threhold",self.optargs["trend_threhold"])
        train_data,val_data,test_data = dataset.build_series_data()
        train_series_transformed,past_convariates_train,future_convariates_train = train_data
        val_series_transformed,past_convariates_val,future_convariates_val = val_data
        test_series_transformed,past_convariates_test,future_convariates_test = test_data
        global_var.set_value("load_ass_data",False)
        global_var.set_value("save_ass_data",False)  
            
        # 使用股票代码数量作为embbding长度
        emb_size = dataset.get_emb_size()
        # emb_size = 500
        load_weight = self.optargs["load_weight"]
        # map_location = torch.device("cpu")
        device = self._build_device()
        
        outer_params = {'pred_weights':self.optargs["pred_weights"],'mode':self.type,'use_pcgrad':self.optargs['use_pcgrad'],
                        'top_num':self.optargs['loss_top_num'],'pred_top_num':self.optargs['pred_top_num'],
                        'opt_size':self.optargs['opt_size'],'candidate_inverse':self.optargs['candidate_inverse'],'pred_mode':self.optargs['pred_mode'],
                        'trend_threhold':self.optargs['trend_threhold'],'max_epochs':self.n_epochs,
                        'pred_index_data_path':self.kwargs["pred_index_data_path"],'load_index_data':self.kwargs["load_index_data"],
                        'pred_cate_data_path':self.kwargs["pred_cate_data_path"],'load_cate_data':self.kwargs["load_cate_data"]
                        }
        if load_weight:
            best_weight = self.optargs["best_weight"]    
            ori_model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
            self.rebuild_model_params(ori_model,model_name=self.optargs["model_name"])  
            ori_model.model.set_outer_params(outer_params) 
        else:
            ori_model = self._build_model(dataset,emb_size=emb_size,use_model_name=True,mode=1) 
        ori_model.mode = self.type 
        ori_model.set_outer_params(outer_params)    
        
        return ori_model,{'train':train_data,'val':val_data,'test':test_data},outer_params
         
    def rebuild_model_params(self,model,model_name=None):
        
        model.model.model_name = model_name
        model.batch_size = self.batch_size     
        model.model.mode = self.type 
        model.model.step_mode = self.optargs["step_mode"]        
        
        
        gpu_params = self.gpus
        model.trainer_params['devices'] = gpu_params
        model.trainer_params['gpus'] = len(gpu_params)
        
    def _build_device(self):
        
        gpu_params = self.gpus
        cudas = ",".join([str(x) for x in gpu_params])
        device = "cuda:" + cudas
        return device
    
    def _build_model(self,dataset,emb_size=1000,use_model_name=True,mode=0,callbacks=None):
        """生成模型"""
        
        log_every_n_steps = self.kwargs["log_every_n_steps"]
        optimizer_cls = torch.optim.Adam
        # optimizer_cls = torch.optim.SGD
        scheduler_config = self.kwargs["scheduler_config"]
        optimizer_kwargs = self.kwargs["optimizer_kwargs"]
        
        # scheduler = torch.optim.lr_scheduler.CyclicLR
        scheduler = torch.optim.lr_scheduler.LinearLR
        categorical_embedding_sizes = {"dayofweek": 5,dataset.get_group_rank_column(): emb_size}
        # categorical_embedding_sizes = None    
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        past_split = self.optargs["past_split"] 
        target_mode = self.optargs["target_mode"] 
        scale_mode = self.optargs["scale_mode"] 
        filter_conv_index = self.optargs["filter_conv_index"] 
        model_name = self.optargs["model_name"]
        model_type = self.optargs["model_type"]
        if not use_model_name:
            model_name = None
        gpu_params = self.gpus
        gpus_size = len(gpu_params)
        # 自定义回调函数
        lightning_callbacks = []
        if callbacks is not None:
            lightning_callbacks.append(callbacks)          
                   
        pl_trainer_kwargs = {"accelerator": "cpu","log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks}    
        pl_trainer_kwargs = {"accelerator": "gpu","gpus":gpus_size, "strategy":"ddp", "devices": gpu_params,"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks}               
        if mode in [0,1,2]:  
            my_model = FuturesModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=self.optargs["forecast_horizon"],
                    cut_len=self.optargs["cut_len"],
                    hidden_size=64,
                    lstm_layers=1,
                    num_attention_heads=4,
                    dropout=self.optargs["dropout"],
                    batch_size=self.batch_size,
                    n_epochs=self.n_epochs,
                    add_relative_index=True,
                    add_encoders=None,
                    categorical_embedding_sizes=categorical_embedding_sizes,
                    # likelihood=QuantileRegression(
                    #     quantiles=quantiles
                    # ), 
                    likelihood=None,
                    # loss_fn=torch.nn.MSELoss(),
                    use_weighted_loss_func=True,
                    loss_number=4,
                    # torch_metrics=metric_collection,
                    random_state=42,
                    model_name=model_name,
                    force_reset=True,
                    log_tensorboard=True,
                    save_checkpoints=True,
                    past_split=past_split,
                    target_mode=target_mode,
                    scale_mode=scale_mode,
                    filter_conv_index=filter_conv_index,
                    work_dir=self.optargs["work_dir"],
                    lr_scheduler_cls=scheduler,
                    lr_scheduler_kwargs=scheduler_config,
                    optimizer_cls=optimizer_cls,
                    optimizer_kwargs=optimizer_kwargs,
                    model_type=model_type,
                    opt_size=self.optargs["opt_size"],
                    pl_trainer_kwargs=pl_trainer_kwargs,
                    pred_top_num=self.optargs["pred_top_num"],
                    task_weights=self.optargs["task_weights"],
                    main_task_seq=self.optargs["main_task_seq"],
                    grad_limits=self.optargs["grad_limits"],
                    pred_weights=self.optargs["pred_weights"],
                    # pl_trainer_kwargs={"log_every_n_steps":log_every_n_steps,"callbacks": lightning_callbacks},
                )
            my_model.act_model_type = mode
        return my_model
                    
    def analysis_model(
        self,
        dataset: TFTFuturesDataset,
        model_env=None
    ):
        """分析特征重要性"""
        
        if model_env is not None:
            # 从参数中直接取得数据环境和模型
            model_ori,build_data,outer_params,train_loader,val_loader = model_env
        else:
            # 生成数据环境和模型
            model_ori,build_data,outer_params = self.prepare_model_env(dataset)
            train_data = build_data['train']
            val_data = build_data['val']
            test_data = build_data['test']
            train_series_transformed,past_convariates_train,future_convariates_train = train_data
            val_series_transformed,past_convariates_val,future_convariates_val = val_data
            test_series_transformed,past_convariates_test,future_convariates_test = test_data 
            
            
            trainer,model,train_loader,val_loader,_ = \
            model_ori.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                     max_samples_per_ts=None,trainer=None,epochs=self.n_epochs,verbose=True,num_loader_workers=8,seperate_mode=False)  
            model_ori.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            model_ori.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            model_ori.model.set_outer_params(outer_params) 

        model_ori.model.eval()
        real_model = model_ori.model.sub_models[0].cuda().float()
        real_model.eval()        # SHAP 必须 eval
        
        for p in real_model.parameters():
            p.requires_grad = False  # 关闭梯度，提速防报错
        
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        # 背景数据：用少量训练集（10~100 个样本，算基准）
        cnt = 100
        past_covariate_total = []
        static_covariate_total = []
        historic_future_covariates_total = []
        # 背景数据：用少量训练集
        background,_ = self.form_input_data(train_dataset,pl_module=model_ori.model,data_loader=train_loader,sampler_cnt=5,mode='train')
        # 待解释数据：选测试集一小部分
        to_explain,layer_recorder = self.form_input_data(val_dataset,pl_module=model_ori.model,data_loader=val_loader,sampler_cnt=3,mode='val')
        # Model Layer compute analysis
        sample_num = 50
        rtn_data = None
        # self.model_layer_analysis(real_model, background, to_explain,sample_num=sample_num)
        # self.viz_shape_value(to_explain,sample_num=sample_num)
        # rtn_data = self.ind_layer_analysis(real_model, background, to_explain)
        rtn_data = self.ind_analysis(train_loader.dataset,val_loader.dataset, pl_module=model_ori.model, train_loader=train_loader,val_loader=val_loader)
        # self.check_sampler_output(real_model, background)

        return rtn_data   
         
    
    def ind_analysis(self,train_dataset,val_dataset,pl_module=None,train_loader=None,val_loader=None):
        
        real_model = pl_module.sub_models[0].cuda().float()
        background,_ = self.form_input_data(train_dataset,pl_module=pl_module,data_loader=train_loader,sampler_cnt=5,mode='train')
        to_explain,layer_recorder = self.form_input_data(val_dataset,pl_module=pl_module,data_loader=val_loader,sampler_cnt=3,mode='val')
        rtn_data = self.ind_layer_analysis(real_model, background, to_explain)
        
        return rtn_data
    
    def check_sampler_output(self,model,input):
        
        import darts_pro.act_model.union_transformer as ut
        ut.set_global(True)
        
        with torch.no_grad():
            input_1 = [item[:10] for item in input]
            input_2 = [item[10:20] for item in input]
            out1 = model(*input_1) 
            out2 = model(*input_2)
            out = model(*input) 
            print("out2 mean:{},std:{}".format(out[2][0].mean(),out[2][0].std()))
            print("out3 mean 2:{},std:{}".format(out[3][0].mean(),out[3][0].std()))
    
    def viz_shape_value(self,to_explain,sample_num=10):
        
        analysis_no = 2
        analysis_task_name = 'taskTotal'
        analysis_no = 3
        analysis_task_name = 'taskMain'        
        loaded_shap_values = {}
        with open("custom/data/asis/shap_values_train_{}.pkl".format(analysis_no), "rb") as f:
            loaded_shap_values['train'] = pickle.load(f)
        with open("custom/data/asis/shap_values_val_{}.pkl".format(analysis_no), "rb") as f:
            loaded_shap_values['val'] = pickle.load(f)        
        # ---------- 可视化（重要性 + 影响方向） ----------
        # 特征重要性图（最常用）
        print("\n=== import trend_list_main ===")
        title_list = ['static_covs','past_convs','his_future_emb','future_single_emb']
        sum_sv_combine = []
        sum_explain_combine = []
        exp_past_cov_combine = []
        sv_past_cov_combine = []
        for ana_type in ['train','val']:
            sum_sv = []
            sum_explain = []
            
            for i in range(len(loaded_shap_values[ana_type])):
                title = analysis_task_name + "_" + ana_type + "_" + title_list[i]
                print("\n=== do {} {} ===".format(ana_type,title))
                sv = loaded_shap_values[ana_type][i]
                to_explain_item = to_explain[i].cpu().numpy()
                to_explain_item = to_explain_item[:sample_num]
                # 合并shap数值的非特征维度,包含所有输出元素的shap取值
                if i==0:
                    sv_2d = sv.sum(axis=(1,3)) 
                    # 合并待解释数据的非特征维度
                    to_explain_item = to_explain_item.mean(axis=(1))
                elif i==1:
                    sv_2d = sv.sum(axis=(1,2,4)) 
                    to_explain_item = to_explain_item.mean(axis=(1,2))
                    sv_past_cov_combine.append(sv_2d)
                    exp_past_cov_combine.append(to_explain_item)
                elif i==2:
                    sv_2d = sv.sum(axis=(1,2,4)) 
                    to_explain_item = to_explain_item.mean(axis=(1,2))
                else:
                    sv_2d = sv.sum(axis=(1,3)) 
                    to_explain_item = to_explain_item.mean(axis=(1))
                sum_sv.append(sv_2d.sum(axis=-1))
                sum_explain.append(to_explain_item.sum(axis=-1))
                names = [f'num_{i}' for i in range(sv.shape[2])]
                # Draw
                plt.figure(figsize=(10,6))
                shap.summary_plot(
                    sv_2d, to_explain_item,
                    feature_names=names,
                    plot_type="bar", # 蜂群删掉这行
                    show=False,      # 必须False
                )
                # 加标题
                plt.title(title, fontsize=13, pad=12)
                fig = plt.gcf()
                save_path = "custom/data/asis/view_data/{}/{}.png".format(analysis_task_name,title)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig) 
            # 对不同输入进行总体比较   
            sum_sv = np.stack(sum_sv).transpose(1,0)
            sum_explain = np.stack(sum_explain).transpose(1,0)
            sum_sv_combine.append(sum_sv)
            sum_explain_combine.append(sum_explain)
            names = title_list
            plt.figure(figsize=(10,6))
            shap.summary_plot(
                sum_sv, sum_explain,
                feature_names=names,
                plot_type="bar", # 蜂群删掉这行
                show=False,      # 必须False
            )
            title = analysis_task_name + "_" + ana_type + "_all"
            plt.title(title, fontsize=13, pad=12)
            fig = plt.gcf()
            save_path = "custom/data/asis/view_data/{}/{}.png".format(analysis_task_name,title)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
        # # 绘制依赖图
        # plt.figure(figsize=(10,6))
        # for i, ana_type in enumerate(['train','val']):
        #     c = '#1f77b4' if ana_type=='train' else '#ff7f0e'
        #     sum_sv = sum_sv_combine[i]
        #     sum_explain = sum_explain_combine[i]
        #     plt.scatter(sum_explain, sum_sv,c=c, alpha=0.5, s=12, label="train and test set")            
        # fig = plt.gcf()
        # save_path = "custom/data/asis/view_data/{}/dep_tt.png".format(analysis_task_name)
        # fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close(fig)     
        # # Depandance
        # plt.figure(figsize=(10,6))      
        # columns = ['RSI5','SUMD5','CORR5','RSQR5','RVI','MACD','RSV5','WVMA5','STD5','OPEN_COM','KDJ_K','QTLUMA5']
        # sum_explain_train = exp_past_cov_combine[0] # pd.DataFrame(exp_past_cov_combine[0],columns=columns)
        # sum_explain_val = exp_past_cov_combine[1] # pd.DataFrame(exp_past_cov_combine[1],columns=columns)
        # sum_sv_train = sv_past_cov_combine[0] # pd.DataFrame(sv_past_cov_combine[0],columns=columns)
        # sum_sv_val = sv_past_cov_combine[1] # pd.DataFrame(sv_past_cov_combine[1],columns=columns)        
        # # shap.dependence_plot(columns[1], sum_sv_train,sum_explain_train,feature_names=columns, show=False, title="train dep")
        # shap.dependence_plot(columns[1], sum_sv_val,sum_explain_val, feature_names=columns, show=False, title="val dep")            
        # fig = plt.gcf()
        # save_path = "custom/data/asis/view_data/{}/dep_plot_val.png".format(analysis_task_name)
        # fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.close(fig)                
        

    def ind_layer_analysis(self,model_ori,background_input,test_input):

        class ShapWrapper(torch.nn.Module):
            """专门包装 PL 多输出模型，让 SHAP 只接收一个输出"""
            def __init__(self, pl_model, output_index=0,output_no=2):
                super().__init__()
                self.model = pl_model
                self.output_index = output_index  # 选择看第几个输出
                self.layer = None
                self.output_no = output_no
                
            def set_cur_epoch(self,cur_epoch):
                self.cur_epoch = cur_epoch
            def set_max_epochs(self,max_epochs):
                self.max_epochs = max_epochs
                                
            def set_layer(self,layer=None):
                self.layer = layer
            
            def forward(self, *inputs):
                # 原模型返回 (out1, out2, out3)
                if self.layer is not None:
                    outputs = self.forward_to_layer(inputs)
                else:
                    outputs = self.forward_to_end(inputs)
                return outputs
            
            def top_select(self,x):
                x = self.model.top_selector[0](x,output_index=self.output_index)
                # 对于多输出，需要指定计算某个输出值
                # x = x[self.output_no]
                return x
                              
            def forward_to_end(self,x_array):
                x = self.model(*x_array,current_epoch=self.cur_epoch,max_epochs=self.max_epochs)   
                # print("output is:{}".format(x[self.output_index][0].mean()))
                if self.output_index==1:
                    return self.transfer_features(x[self.output_index][0])
                return x[self.output_index][0]
            
            def transfer_features(self,fea_out):
                monitor_keys = ["abpfi","cdifi","hsjs","nffi","yzyl"]
                output = torch.cat([fea_out[monitor_key] for monitor_key in monitor_keys],dim=-1)
                return output
            
            def forward_to_layer(self,x_array):
                if self.layer=='model.trans_model_encoder':
                    x = self.model.trans_model_encoder(*x_array,current_epoch=self.cur_epoch,max_epochs=self.max_epochs)   
                    x = x[0]
                    x = x.reshape([x.shape[0],-1])         
                if self.layer=='model.trans_model_decoder':
                    x = self.model.trans_model_encoder(*x_array,current_epoch=self.cur_epoch,max_epochs=self.max_epochs)   
                    x = self.model.trans_model_decoder(x[0],x_array[3],x[1])   
                    x = x.reshape([x.shape[0],-1])                          
                if self.layer=='model.top_selector.0':
                    x = self.model.trans_model_encoder(*x_array,current_epoch=self.cur_epoch,max_epochs=self.max_epochs)   
                    x = self.model.trans_model_decoder(x[0],x_array[3],x[1])   
                    x = x.reshape(x.shape[0],x.shape[1],-1)
                    x = self.top_select(x)
                    x = x.reshape([x.shape[0],-1])
                return x
        
        output_indexs = [2,3]
        # output_indexs = [2]
        # output_indexs = [1]
        rtn_data = {}
        for output_index in output_indexs:
            model = ShapWrapper(model_ori,output_index=output_index)         
            x = tuple([item[:50] for item in test_input])
            baselines = tuple([xi[:50] for xi in background_input])
            x_train = tuple([xi[50:100] for xi in background_input])
            current_epoch = 130
            max_epochs = 180
            model.set_cur_epoch(current_epoch)
            model.set_max_epochs(max_epochs)
            # current_epoch = torch.tensor(current_epoch)
            # y_baseline = model(*baselines)
            # y_pred = model(*x)
            # print("y_baseline:{},y_pred:{}".format(y_baseline.mean(),y_pred.mean()))
            input_names = ['static_covs','past_convs','his_future_emb','future_single_emb']
            # 对每个输入分别归因
            # gs = GradientShap(model)
            # for i in range(len(baselines)):
            #     baselines_item = tuple([baselines[j] if j==i else x[j] for j in range(len(baselines))])
            #     attr_inp_combine = []
            #     for k in range(sample_num):
            #         # baselines_item_single = tuple([item[k:k+1] for item in baselines_item])
            #         x_single = tuple([item[k:k+1] for item in x])
            #         attr_inp,delta = gs.attribute(x_single, baselines_item, n_samples=200, stdevs=0.1,target=0,return_convergence_delta=True)
            #         attr_inp_combine.append(attr_inp[i])  
            #     attr_inp_combine = torch.cat(attr_inp_combine)
            #     print("input {} attr_inp shape:{},mean:{}".format(input_names[i],attr_inp_combine.shape, attr_inp_combine.abs().mean().item()))       
    
            # 训练接和验证集的归因差异排查
            ig = IntegratedGradients(model)
            for i in range(len(baselines)):
                train_importance_total = []
                val_importance_total = []            
                train_attr, delta_train = ig.attribute(x_train, target=0, return_convergence_delta=True,internal_batch_size=32)
                val_attr, delta_val = ig.attribute(x, target=0, return_convergence_delta=True,internal_batch_size=32)
                train_attr_final = torch.nan_to_num(train_attr[i]) 
                train_importance = np.abs(train_attr_final.detach().cpu().numpy())
                train_importance_mean = train_importance.flatten()
                train_importance_mean = train_importance_mean[train_importance_mean!=0]
                train_importance_mean = np.abs(train_importance).mean()
                train_importance_detail = []
                for j in range(train_attr_final.shape[-1]):
                    train_importance_detail_item = train_attr_final[...,j]
                    train_importance_detail_item = train_importance_detail_item[train_importance_detail_item!=0].abs().detach().cpu().numpy().mean()
                    train_importance_detail.append(train_importance_detail_item)
    
                val_attr_final = torch.nan_to_num(val_attr[i])    
                val_importance = np.abs(val_attr_final[i].detach().cpu().numpy())                       
                # val_importance = np.abs(val_attr[i].detach().cpu().numpy())           
                val_importance_mean = val_importance.mean()
                val_importance_detail = val_importance.mean(tuple(range(val_importance.ndim - 1)))
                if output_index==2:
                    rtn_data[input_names[i]] = [train_importance_mean,val_importance_mean]
                print("output_index_{} input {} train_importance mean:{},val_importance mean:{}".format(output_index,input_names[i],
                                            train_importance_mean, val_importance_mean)) 
                # if i==1:
                #     print("input {} train_importance detail:{},val_importance detail:{}".format(input_names[i],train_importance_detail, val_importance_detail)) 
 
        focus_layers = ['model.trans_model_encoder','model.trans_model_decoder','model.top_selector.0']  
        train_contribs = {}
        val_contribs = {}       
        def concat_input(tule_input):
            static_covs,past_convs_item, his_future_covs = tule_input
            x_concat = torch.cat([static_covs.unsqueeze(-2).repeat(1,1,past_convs_item.shape[2],1),past_convs_item,his_future_covs],dim=-1)
            return x_concat
        # ========== 核心1：分层独立归一 + 绝对值统计（推荐主方案） ==========
        def layer_wise_normalize(attr_tensor):
            """
            单图层独立Min-Max归一化，不跨层共享极值
            attr_tensor: 任意维度归因张量
            """
            attr = attr_tensor.detach().cpu().numpy()
            # 每层/每个样本单独归一，保留层内分布
            min_val = np.min(attr, axis=tuple(range(1, attr.ndim)), keepdims=True)
            max_val = np.max(attr, axis=tuple(range(1, attr.ndim)), keepdims=True)
            norm_attr = (attr - min_val) / (max_val - min_val + 1e-8)
            return norm_attr
        
        def calc_total_contribution(attr_tensor):
            """
            计算层总有效贡献：绝对值均值（消除正负抵消、维度影响）
            """
            attr_abs = torch.abs(attr_tensor)
            # 全局平均，统一不同维度层的统计口径
            total_contrib = torch.mean(attr_abs).item()
            return total_contrib
        
        # ========== 核心2：深度加权系数（平滑校正） ==========
        def get_depth_weight(layer_depth, total_depth):
            """
            深度越靠近输出（depth越大），权重小幅下调，平滑抵消梯度偏置
            layer_depth: 当前层相对深度 (Transformer=1, MLP=2)
            total_depth: 网络总层数
            """
            # 系数范围 0.7 ~ 1.0，线性平滑，不极端压制/抬高
            return 1.0 - 0.3 * (layer_depth - 1) / (total_depth - 1)   
             
        # for name, layer in model.named_modules():
        #     if name in focus_layers:
        #         model.set_layer(name)
        #         lgs = LayerGradientShap(model, layer)
        #         attr_inp_combine = []
        #         attr_inp_combine_after = []
        #         attr_inp_combine_train = []
        #         attr_inp_combine_train_after = []
        #         for k in range(sample_num):
        #             # baselines_item_single = tuple([item[k:k+1] for item in baselines_item])
        #             x_single = tuple([item[k:k+1] for item in x])
        #             x_single_train = tuple([item[k:k+1] for item in x_train])
        #             # attr_inp = lgs.attribute(x_single, baselines=baselines, n_samples=50,target=0,attribute_to_layer_input=True)
        #             attr_inp_after = lgs.attribute(x_single, baselines=baselines, n_samples=50,target=0,attribute_to_layer_input=False)
        #             # attr_inp_train = lgs.attribute(x_single_train, baselines=baselines, n_samples=50,target=0,attribute_to_layer_input=True)
        #             attr_inp_train_after = lgs.attribute(x_single_train, baselines=baselines, n_samples=50,target=0,attribute_to_layer_input=False)
        #             if name=='model.trans_model_encoder':
        #                 # attr_inp_combine.append(attr_inp[1])
        #                 # attr_inp_combine_train.append(attr_inp_train[1])
        #                 for h in range(x_single[0].shape[1]):
        #                     size = h * x_single[0].shape[2]
        #                     attr_inp_after = lgs.attribute(x_single, baselines=baselines, n_samples=50,target=size,attribute_to_layer_input=False)
        #                     attr_inp_train_after = lgs.attribute(x_single_train, baselines=baselines, n_samples=50,target=size,attribute_to_layer_input=False)
        #                     attr_inp_after_final = torch.nan_to_num(attr_inp_after[0]) 
        #                     attr_inp_train_after_final = torch.nan_to_num(attr_inp_train_after[0])  
        #                     attr_inp_combine_after.append(attr_inp_after_final)
        #                     attr_inp_combine_train_after.append(attr_inp_train_after_final)
        #             if name=='model.trans_model_decoder':
        #                 # attr_inp_combine.append(attr_inp[1])
        #                 # attr_inp_combine_train.append(attr_inp_train[1])
        #                 for h in range(x_single[0].shape[1]):
        #                     size = h * x_single[0].shape[2]
        #                     attr_inp_after = lgs.attribute(x_single, baselines=baselines, n_samples=50,target=size,attribute_to_layer_input=False)
        #                     attr_inp_train_after = lgs.attribute(x_single_train, baselines=baselines, n_samples=50,target=size,attribute_to_layer_input=False)
        #                     attr_inp_after_final = torch.nan_to_num(attr_inp_after[0]) 
        #                     attr_inp_train_after_final = torch.nan_to_num(attr_inp_train_after[0])      
        #                     attr_inp_combine_after.append(attr_inp_after_final)
        #                     attr_inp_combine_train_after.append(attr_inp_train_after_final)                      
        #             if name=='model.top_selector.0':
        #                 # attr_inp_combine.append(attr_inp[0])
        #                 # attr_inp_combine_train.append(attr_inp_train[0])   
        #                 if output_index==2:
        #                     r = 7
        #                 else:
        #                     r = 4
        #                 for h in range(r):   
        #                     attr_inp_after = lgs.attribute(x_single, baselines=baselines, n_samples=50,target=h,attribute_to_layer_input=False)
        #                     attr_inp_train_after = lgs.attribute(x_single_train, baselines=baselines, n_samples=50,target=h,attribute_to_layer_input=False)
        #                     attr_inp_after_final = torch.nan_to_num(attr_inp_after[0]) 
        #                     attr_inp_train_after_final = torch.nan_to_num(attr_inp_train_after[0])      
        #                     attr_inp_combine_after.append(attr_inp_after_final)
        #                     attr_inp_combine_train_after.append(attr_inp_train_after_final)                                           
        #         # attr_inp_combine = torch.cat(attr_inp_combine)    
        #         # attr_inp_combine_train = torch.cat(attr_inp_combine_train) 
        #         attr_inp_combine_after = torch.cat(attr_inp_combine_after)    
        #         attr_inp_combine_train_after = torch.cat(attr_inp_combine_train_after) 
        #         # train_contribs[name] = attr_inp_combine_train.abs().mean().item()
        #         # val_contribs[name] = attr_inp_combine.abs().mean().item()
        #         # train_contrib_value = torch.norm(attr_inp_combine_train_after,dim=-1).mean().item()
        #         # val_contrib_value = torch.norm(attr_inp_combine_after,dim=-1).mean().item()
        #         attr_inp_combine_train_after = attr_inp_combine_train_after[attr_inp_combine_train_after!=0]
        #         attr_inp_combine_after = attr_inp_combine_after[attr_inp_combine_after!=0]
        #         train_contrib_total = calc_total_contribution(attr_inp_combine_train_after)
        #         # train_contrib_layer = np.mean(np.abs(layer_wise_normalize(attr_inp_combine_train_after)))
        #         val_contrib_total = calc_total_contribution(attr_inp_combine_after)
        #         # val_contrib_layer = np.mean(np.abs(layer_wise_normalize(attr_inp_combine_after)))
        #         train_contribs[name+"_after_total"] = train_contrib_total
        #         val_contribs[name+"_after_total"] = val_contrib_total
        # print("train_contribs:",train_contribs)
        # print("val_contribs:",val_contribs)
        
        return rtn_data
                        
    def model_layer_analysis(self,model,background_input,test_input,sample_num=10):
        """Model Layer compute analysis"""
        
        # 每个输入展平后的总长度
        dim1 = 34 * 7
        dim2 = 34 * 28 * 12
        dim3 = 34 * 28 * 20
        dim4 = 34 * 20
        total_dim = dim1 + dim2 + dim3 + dim4
        
        def flatten_inputs(x_arr):
            """把多个不同形状输入 → 展平 → 拼接成一个长向量"""
            x_arr_flat = []
            for x in x_arr:
                x_arr_flat.append(x.flatten(1))
            return torch.cat(x_arr_flat, dim=1).cpu().numpy()
        
        def split_restore_inputs(X_flat):
            """把长向量 → 拆分 → 恢复成模型需要的多输入形状"""
            X_tensor = torch.tensor(X_flat, dtype=torch.float32)
            
            x1_flat = X_tensor[:, :dim1]
            x2_flat = X_tensor[:, dim1:dim1+dim2]
            x3_flat = X_tensor[:, dim1+dim2:dim1+dim2+dim3]
            x4_flat = X_tensor[:, dim1+dim2+dim3:]
            
            # 恢复原形状
            x1 = x1_flat.view(-1, 34,7).cuda()
            x2 = x2_flat.view(-1, 34, 28,12).cuda()
            x3 = x3_flat.view(-1, 34, 28,20).cuda()
            x4 = x4_flat.view(-1, 34, 20).cuda()
            return x1, x2, x3, x4

        class ShapWrapper(torch.nn.Module):
            """专门包装 PL 多输出模型，让 SHAP 只接收一个输出"""
            def __init__(self, pl_model, output_index=0,output_no=2):
                super().__init__()
                self.model = pl_model
                self.output_index = output_index  # 选择看第几个输出
                self.layer = None
                self.output_no = output_no
            
            def set_layer(self,layer=None):
                self.layer = layer
            
            def forward(self, *inputs):
                # 原模型返回 (out1, out2, out3)
                # outputs = self.forward_to_layer(inputs)
                outputs = self.forward_to_end(inputs)
                return outputs

            # 前向，拿到该层输出,注意：这里要搭一个“到这一层为止”的 forward
            def forward_to_layer(self,x_array):
                if self.layer is model.trans_model:
                    x = model.trans_model(*x_array)   
                    x = x[0][1]
                    x = x.reshape([x.shape[0],-1])         
                if self.layer is model.top_selector[0]:
                    x = model.trans_model(*x_array)   
                    x = x[0][1]  
                    x = x.reshape(x.shape[0],x.shape[1],-1)
                    x = model.top_selector[0](x,None)
                    # 对于多输出，需要指定计算某个输出值
                    x = x[self.output_no]
                    x = x.reshape([x.shape[0],-1])
                return x
                                
            def forward_to_end(self,x_array):
                x = model.trans_model(*x_array)   
                x = x[0][1]  
                x = x.reshape(x.shape[0],x.shape[1],-1)
                x = model.top_selector[0](x,None)
                # 对于多输出，需要指定计算某个输出值
                x = x[self.output_no]
                x = x.reshape([x.shape[0],-1])*1000000
                return x
                       
        layers_to_explain = [model.trans_model, model.top_selector[0]]
        
        # 4. 逐层算 SHAP
        shap_per_layer = []
        layer_outputs = []

        # ==============================================
        # 包装函数：给 KernelExplainer 用（输出=中间层）
        # ==============================================
        # def predict_middle(X_flat):
        #     """输入：展平向量；输出：模型中间层结果"""
        #     x_tuple = split_restore_inputs(X_flat)
        #
        #     with torch.no_grad():
        #         model(*x_tuple)  # 前向传播
        #
        #     # 返回【中间层结果】给 SHAP
        #     return model.middle.numpy()
        
        X_test = [item[:sample_num] for item in test_input]
        X_train = [item[sample_num:2*sample_num] for item in background_input]
        backround = [item[:sample_num] for item in background_input]
        
        for i,out_no in enumerate([2,3]):
               
            wrapped_model = ShapWrapper(model, output_no=out_no)     
            # wrapped_model.set_layer(layer)
            
            # GradientExplainer：指定 (model, layer)
            # explainer = shap.GradientExplainer(
            #     (wrapped_model, layer),   # 关键：绑定特定层
            #     background_input
            # )
            # explainer = shap.KernelExplainer(forward_to_layer, background)
            explainer = shap.GradientExplainer(wrapped_model, backround)
            # 分别计算训练集的SHAP和验证集的SHAP
            sv_val = explainer.shap_values(X_test, nsamples=10)     
            sv_train = explainer.shap_values(X_train, nsamples=10)     
            # shap_per_layer.append(sv)   
            with open("custom/data/asis/shap_values_val_{}.pkl".format(out_no), "wb") as f:
                pickle.dump(sv_val, f)
            with open("custom/data/asis/shap_values_train_{}.pkl".format(out_no), "wb") as f:
                pickle.dump(sv_train, f)                
            print("out_{}  ok".format(out_no))

        # for i, (out, sv) in enumerate(zip(layer_outputs, shap_per_layer)):
        #     print(f"Layer {i+1} | out.shape={out.shape} | shap.shape={sv.shape}")
            # 看 sv 是否全接近 0 → 这一层“不干活”
                
    def form_input_data(self,fur_dataset,sampler_cnt=3,data_loader=None,pl_module=None,mode='train'):
        
        static_covs_total,past_convs_item_total,his_future_emb_total,future_emb_total,future_single_emb_total = [],[],[],[],[]
        future_date = []
        
        for i,data in enumerate(data_loader):
            (
                past_target,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                past_future_covariates,
                future_target,
                target_class,
                price_targets,
                past_future_round_targets,
                index_round_targets,
                future_week_info,
                target_info
            ) = data           
            inp = (past_target, future_target, past_covariates, historic_future_covariates, future_covariates, 
                   static_covariates, past_future_covariates, price_targets, past_future_round_targets, index_round_targets,target_class,target_info)    
            future_date.append([item[0]['future_start_datetime'] for item in target_info])
            pl_module = pl_module.cuda()       
            input_batch = pl_module._process_input_batch(inp,mode=mode)
            input_batch_transform = []
            for item in input_batch:
                if isinstance(item,list):
                    item = [it.float().cuda() for it in item]  
                else:
                    item = item.float().cuda()
                input_batch_transform.append(item)
            pl_module.eval()
            with torch.no_grad():
                ouput = pl_module.forward(input_batch_transform)    
            out_total,input_final,_ = ouput
            layer_recorder = out_total[0][-1]
            static_covs,past_convs_item,his_future_emb,future_single_emb = input_final
            static_covs_total.append(static_covs)
            past_convs_item_total.append(past_convs_item)
            his_future_emb_total.append(his_future_emb)
            future_single_emb_total.append(future_single_emb)
            if i>=sampler_cnt-1:
                break
        
        future_date = np.array(future_date)
        
        static_covs_total = torch.cat(static_covs_total,dim=0).float().cuda()
        past_convs_item_total = torch.cat(past_convs_item_total,dim=0).float().cuda()
        his_future_emb_total = torch.cat(his_future_emb_total,dim=0).float().cuda()
        # future_emb_total = torch.cat(future_emb_total,dim=0).float()
        future_single_emb_total = torch.cat(future_single_emb_total,dim=0).float().cuda()
        
        input_final = [static_covs_total,past_convs_item_total,his_future_emb_total,future_single_emb_total]
           
        return input_final,layer_recorder
    
    @staticmethod           
    def _batch_collate_fn(batch):
        """批次整合"""
        
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
            elif isinstance(elem, tuple):
                aggregated.append([sample[i] for sample in batch])                
            elif isinstance(elem, Dict):
                aggregated.append([sample[i] for sample in batch])                
            elif elem is None:
                aggregated.append(None)                
            elif isinstance(elem, List):
                aggregated.append([sample[i] for sample in batch])
            else:
                print("no match for:",elem.dtype)
        return tuple(aggregated)   

    def build_val_result(self,dataset=None):
        """生成验证结果,并整合保存"""
        
        self.fit_futures_industry(dataset)
        
    def build_pred_result(self,pred_date,dataset=None):    
        """根据预测区间参数进行预测，pred_range为二元数组，数组元素类型为date"""        

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]   
        
        if dataset is None:
            dataset = self.dataset
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        expand_length = 2 *(input_chunk_length + output_chunk_length)
         
        # 根据日期范围逐日进行预测，得到预测结果   
        start_date = pred_date
        # 同时需要延长集合时间
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,-1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        begin_day = valid_range[0]
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train":[total_range[0],prev_day],"valid":[begin_day,prev_day]}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 记录实际截止日期对应的序列编号最大值，后续与模拟数据进行区分
        time_idx_mapping = dataset.df_all.groupby("instrument")["time_idx"].max()
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length,begin_date=int(start_date)) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 再次生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)   
        # 给每个品种序列放入实际最大编号   
        for series in val_series_transformed:
            time_idx = time_idx_mapping[time_idx_mapping.index==series.instrument_code].values[0]
            series.last_time_idx = time_idx
        
        device = self._build_device()  
        best_weight = self.optargs["best_weight"]    
        model = FuturesModel.load_from_checkpoint(self.optargs["model_name"],work_dir=self.optargs["work_dir"],device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
        model_name = self.optargs["model_name"]  
        self.rebuild_model_params(model,model_name=model_name)  
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        
        # 进行推理及预测，先fit再predict
        model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)               
        
        # 通过植入属性的方式，设置预测品种个数
        if 'pred_top_num' in self.optargs:
            model.model.pred_top_num = self.optargs['pred_top_num']
        else:
            model.model.pred_top_num = 2
        # 进行预测           
        pred_result = model.predict(series=val_series_transformed,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))
        
        return pred_result        
                             
    def predict(self, dataset,pred_range=None):
        """根据预测区间参数进行预测并进行评估，pred_range为二元数组，数组元素类型为date"""

        if pred_range is None:
            pred_range = dataset.kwargs["segments"]["test"] 
            
        start_date = pred_range[0]
        end_date = pred_range[1]
        trade_dates = np.array(get_tradedays(start_date,end_date)).astype(np.int)
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        pred_result_list = {}
        for pred_date in trade_dates:
            pred_result = self.build_pred_result(str(pred_date),dataset=dataset)
            pred_result_list[pred_date] = pred_result[pred_date]
            
        # 对预测结果进行评估
        pred_dates = np.array(list(pred_result_list.keys())).astype(np.int)
        # 取得实际需要的日期结果数据
        match_dates = np.intersect1d(trade_dates,pred_dates)
        pred_result_target = {}
        # 生成真实数据，以进行评估
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        last_day = get_tradedays_dur(end_date,3*output_chunk_length)      
        segments = {"train":[total_range[0],last_day],"valid":[valid_range[0],last_day]}  
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        df_target = dataset.df_all
        import_price_result = []
        for key in match_dates:
            pred_result_target[key] = pred_result_list[key]
            target_class_list = []
            for index,row in pred_result_list[key].iterrows():
                instrument = row['instrument']
                trend = row['top_flag']
                item_cur_idx = df_target[(df_target['instrument']==instrument)&(df_target['datetime_number']==key)]['time_idx'].values[0]
                df_item = df_target[(df_target['instrument']==instrument)&(df_target['time_idx']>=(item_cur_idx-1))]
                price_list = df_item['CLOSE'].values
                diff_range = (price_list[output_chunk_length] - price_list[0])/price_list[0]
                p_taraget_class = get_simple_class(diff_range)  
                if trend==0:
                    p_taraget_class = [3,2,1,0][p_taraget_class] 
                    diff_range = -diff_range
                target_class_list.append(p_taraget_class)
                import_price_result.append([key,instrument,trend,p_taraget_class,diff_range])
        import_price_result = np.array(import_price_result)
        import_price_result = pd.DataFrame(import_price_result,
            columns=["date","instrument","trend","result","yield_rate"])
        import_price_result['trend'] = import_price_result['trend'].astype(int)
        import_price_result['result'] = import_price_result['result'].astype(int)
        import_price_result['yield_rate'] = import_price_result['yield_rate'].astype(float)
        
        print("total yield:{}".format(import_price_result["yield_rate"].sum()))
        return import_price_result          

    def build_pred_result_2step(self,pred_date,dataset=None):    
        """使用二阶段模式，根据预测区间参数进行预测。pred_range为二元数组，数组元素类型为date"""        

        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]   
        if dataset is None:
            dataset = self.dataset

        step1_keep_col = dataset.col_def['step1_keep_col']
        step2_keep_col = dataset.col_def['step2_keep_col']
        target_mode_step1 = self.optargs["target_mode_step1"]  
        target_mode_step2 = self.optargs["target_mode_step2"]  
        scale_mode_step1 = self.optargs["scale_mode_step1"]  
        scale_mode_step2 = self.optargs["scale_mode_step2"]  
                                
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        expand_length = 2 *(input_chunk_length + output_chunk_length)
         
        # 根据日期范围逐日进行预测，得到预测结果   
        start_date = pred_date
        # 同时需要延长集合时间
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        prev_day = get_tradedays_dur(start_date,-1)
        last_day = get_tradedays_dur(start_date,3*output_chunk_length)
        # 以当天为数据时间终点
        total_range[1] = prev_day
        begin_day = valid_range[0]
        valid_range[1] = prev_day 
        # 生成未扩展的真实数据
        segments = {"train":[total_range[0],prev_day],"valid":[begin_day,prev_day]}
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        # 记录实际截止日期对应的序列编号最大值，后续与模拟数据进行区分
        time_idx_mapping = dataset.df_all.groupby("instrument")["time_idx"].max()
        # 为了和训练阶段保持一致处理，需要补充模拟数据
        df_expands = dataset.expand_mock_df(dataset.df_all,expand_length=expand_length) 
        # 生成模拟数据后重置日期区间,以生成足够日期范围的val_series_transformed
        segments = {"train":[total_range[0],last_day],"valid":[begin_day,last_day]}  
        # 再次生成序列数据            
        train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = \
            dataset.build_series_data_with_segments(segments,outer_df=df_expands)   
        # 给每个品种序列放入实际最大编号   
        for series in val_series_transformed:
            time_idx = time_idx_mapping[time_idx_mapping.index==series.instrument_code].values[0]
            series.last_time_idx = time_idx
        
        device = self._build_device()
        best_weight = self.optargs["best_weight"]  
        # 首先使用总体模型生成总体趋势结果
        model_name_step1 = self.optargs["model_name_step1"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step1,work_dir=self.optargs["work_dir"],device=device,
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.rebuild_model_params(model,model_name=model_name_step1)  
        model.model.step_mode = 1
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        
        # 2个阶段需要不同的目标值，在这里通过多配置的模式，并且动态改变val_series序列的列值来实现
        model.scale_mode = scale_mode_step1
        model.model.scale_mode = scale_mode_step1
        model.target_mode = target_mode_step1
        model.model.target_mode = target_mode_step1        
        target_column = np.array(dataset.col_def['target_column'])
        real_target_column = target_column[np.array(step1_keep_col)]
        rm_cols = np.setdiff1d(target_column, real_target_column, assume_unique=False).tolist()
        val_series_transformed_new = []
        for ser in val_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            val_series_transformed_new.append(ser_new)
        train_series_transformed_new = []
        for ser in train_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            train_series_transformed_new.append(ser_new)            
            
        model.fit(train_series_transformed_new, future_covariates=future_convariates, val_series=val_series_transformed_new,
                val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)   
        
        # 先fit再赋值，则以当前数据集生成的mapping为准
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings                     
        model.predict(series=val_series_transformed_new,past_covariates=past_convariates,future_covariates=future_convariates,
                                                    batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))        
        # 再用第二阶段模型生成实际品种结果
        model_name_step2 = self.optargs["model_name_step2"]  
        model = FuturesIndustryModel.load_from_checkpoint(model_name_step2,work_dir=self.optargs["work_dir"],device=device,
                                                         best=best_weight,batch_file_path=self.batch_file_path)
        self.rebuild_model_params(model,model_name=model_name_step2)  
        model.model.step_mode = 2
        model.batch_size = self.batch_size     
        model.mode = "predict"
        model.model.mode = "predict"
        
        # 2个阶段需要不同的目标值，在这里通过多配置的模式，并且动态改变val_series序列的列值来实现
        model.scale_mode = scale_mode_step2
        model.model.scale_mode = scale_mode_step2
        model.target_mode = target_mode_step2
        model.model.target_mode = target_mode_step2           
        target_column = np.array(dataset.col_def['target_column'])
        real_target_column = target_column[np.array(step2_keep_col)]
        rm_cols = np.setdiff1d(target_column, real_target_column, assume_unique=False).tolist()
        val_series_transformed_new = []
        for ser in val_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            val_series_transformed_new.append(ser_new)
        train_series_transformed_new = []
        for ser in train_series_transformed:
            ser_new = ser.drop_columns(rm_cols)
            ser_new.instrument_code = ser.instrument_code
            train_series_transformed_new.append(ser_new)                       
        # For pred step2 result
        model.model.inter_rs_filepath = self.optargs["inter_rs_filepath"]
        # 进行推理及预测，先fit再predict
        model.fit(train_series_transformed_new, future_covariates=future_convariates, val_series=val_series_transformed_new,
                 val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                 max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=6)    
        # 先fit再赋值，则以当前数据集生成的mapping为准           
        model.model.train_sw_ins_mappings = model.train_sw_ins_mappings
        model.model.valid_sw_ins_mappings = model.valid_sw_ins_mappings          
        pred_result = model.predict(series=val_series_transformed_new,past_covariates=past_convariates,future_covariates=future_convariates,
                                            batch_size=self.batch_size,num_loader_workers=0,pred_date_begin=int(pred_date))
        
        return pred_result 
    
    def predict_indus_and_detail(self, dataset,pred_range=None):
        """使用二阶段模型，根据预测区间参数进行预测并进行评估，pred_range为二元数组，数组元素类型为date"""

        if pred_range is None:
            pred_range = dataset.kwargs["segments"]["test"] 
            
        start_date = pred_range[0]
        end_date = pred_range[1]
        trade_dates = np.array(get_tradedays(start_date,end_date)).astype(np.int)
        
        input_chunk_length = self.optargs["wave_period"] - self.optargs["forecast_horizon"]
        output_chunk_length = self.optargs["forecast_horizon"]
        pred_result_list = {}
        for pred_date in trade_dates:
            pred_result = self.build_pred_result_2step(str(pred_date),dataset=dataset)
            pred_result_list[pred_date] = pred_result[pred_date]
            
        # 对预测结果进行评估
        pred_dates = np.array(list(pred_result_list.keys())).astype(np.int)
        # 取得实际需要的日期结果数据
        match_dates = np.intersect1d(trade_dates,pred_dates)
        pred_result_target = {}
        # 生成真实数据，以进行评估
        total_range = dataset.segments["train"]
        valid_range = dataset.segments["valid"]    
        # 扩充起止时间，以进行数据集预测匹配
        last_day = get_tradedays_dur(end_date,3*output_chunk_length)      
        segments = {"train":[total_range[0],last_day],"valid":[valid_range[0],last_day]}  
        dataset.build_series_data_with_segments(segments,no_series_data=True,val_ds_filter=False,fill_future=True)
        df_target = dataset.df_all
        import_price_result = []
        for key in match_dates:
            pred_result_target[key] = pred_result_list[key]
            target_class_list = []
            for index,row in pred_result_list[key].iterrows():
                instrument = row['instrument']
                trend = row['top_flag']
                item_cur_idx = df_target[(df_target['instrument']==instrument)&(df_target['datetime_number']==key)]['time_idx'].values[0]
                df_item = df_target[(df_target['instrument']==instrument)&(df_target['time_idx']>=(item_cur_idx-1))]
                price_list = df_item['CLOSE'].values
                diff_range = (price_list[output_chunk_length] - price_list[0])/price_list[0]
                p_taraget_class = get_simple_class(diff_range)  
                if trend==0:
                    p_taraget_class = [3,2,1,0][p_taraget_class] 
                    diff_range = -diff_range
                target_class_list.append(p_taraget_class)
                import_price_result.append([key,instrument,trend,p_taraget_class,diff_range])
        import_price_result = np.array(import_price_result)
        import_price_result = pd.DataFrame(import_price_result,
            columns=["date","instrument","trend","result","yield_rate"])
        import_price_result['trend'] = import_price_result['trend'].astype(int)
        import_price_result['result'] = import_price_result['result'].astype(int)
        import_price_result['yield_rate'] = import_price_result['yield_rate'].astype(float)
        
        print("total yield:{}".format(import_price_result["yield_rate"].sum()))
        return import_price_result    
           
    def batch_pred_bidi(self,dataset):   
        """批量推理，并生成整合后结果"""
        
        self.pred_data_path = self.kwargs["pred_data_path"]
        self.batch_file_path = self.kwargs["batch_file_path"]
        self.load_dataset_file = self.kwargs["load_dataset_file"]
        self.save_dataset_file = self.kwargs["save_dataset_file"]      
        if not os.path.exists(self.batch_file_path):
            os.mkdir(self.batch_file_path)
        # 生成tft时间序列数据集,包括目标数据、协变量等
        model_path = self.optargs["model_path"]
        entries = os.listdir(model_path)
        subdirectories = []
        for entry in entries:
            full_path = os.path.join(model_path, entry)
            if os.path.isdir(full_path):
                subdirectories.append(entry)
        
        total_results = []
        for sub_path in subdirectories:
            model_name = "step"
            # 拼接对应的模型目录
            work_dir = os.path.join(model_path,sub_path)
            train_series_transformed,val_series_transformed,series_total,past_convariates,future_convariates = dataset.build_series_data()
            global_var.set_value("load_ass_data",False)
            global_var.set_value("save_ass_data",False)  
            device = self._build_device()
            best_weight = self.optargs["best_weight"]    
            # 获取模型父路径，并遍历分别获得每个推理需要使用的模型
            model = FuturesModel.load_from_checkpoint(model_name,work_dir=work_dir,device=device,
                                                             best=best_weight,batch_file_path=self.batch_file_path,map_location=None)
            self.rebuild_model_params(model,model_name=model_name)  
            model.batch_size = self.batch_size     
            model.mode = "pred_result"
            model.model.mode = "pred_result"             
            # 预测模式下，通过设置epochs为0来达到不进行训练的目的，并直接执行validate
            trainer,model_real,train_loader,val_loader = model.fit(train_series_transformed, future_covariates=future_convariates, val_series=val_series_transformed,
                     val_future_covariates=future_convariates,past_covariates=past_convariates,val_past_covariates=past_convariates,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0)
            model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            model.model.train_sw_ins_mappings = train_loader.dataset.sw_ins_mappings
            trainer.validate(model=model_real,dataloaders=val_loader)    
            # 推理后保留当前子模型下一个月份的数据，作为实际结果数据
            mode_month_date = datetime.datetime.strptime(str(sub_path), "%Y%m")  
            next_month = get_next_month(mode_month_date).strftime("%Y%m")
            result_view_file_path = model.model.result_view_file_path
            coll_result = pd.read_csv(result_view_file_path)
            coll_result = coll_result[coll_result['date'].astype(str).str[:6]==next_month]
            total_results.append(coll_result)
        total_results = pd.concat(total_results)
        total_results = total_results[['date','pred_trend', 'top_index', 'instrument', 'diff_range']] 
        save_path = os.path.join(self.optargs["work_dir"],"total_coll_results.csv")
        total_results.to_csv(save_path,index=False)
