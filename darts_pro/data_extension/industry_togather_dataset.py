from darts.utils.data.sequential_dataset import (
    SplitCovariatesTrainingDataset,
)

import pickle
import os
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from numba.core.types import none
import torch
from typing import Optional, Sequence, Tuple, Union
from sklearn.preprocessing import MinMaxScaler

from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset,DualCovariatesSequentialDataset
from darts.utils.data.inference_dataset import InferenceDataset,PastCovariatesInferenceDataset,DualCovariatesInferenceDataset
from darts.utils.data.shifted_dataset import GenericShiftedDataset,MixedCovariatesTrainingDataset
from darts.utils.data.utils import CovariateType
from darts.logging import raise_if_not
from darts import TimeSeries
from cus_utils.common_compute import normalization,slope_last_classify_compute
from tft.class_define import CLASS_VALUES,get_simple_class,get_complex_class

from cus_utils.db_accessor import DbAccessor
import cus_utils.global_var as global_var
from cus_utils.encoder_cus import transform_slope_value
from tushare.stock.indictor import kdj
from .custom_dataset import CusGenericShiftedDataset
from .industry_mapping_util import IndustryMappingUtil
from cus_utils.log_util import AppLogger
from torch._jit_internal import ignore
logger = AppLogger()


class IndustryTogatherDataset(CusGenericShiftedDataset):
    """基于日期对齐并按照行业分类的滚动数据集,以行业分类为主进行整合"""
    
    def __init__(
        self,
        target_series,
        covariates=None,
        future_covariates=None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        target_num=0,
        shift: int = 1,
        shift_covariates: bool = False,
        max_samples_per_ts: Optional[int] = None,
        covariate_type: CovariateType = CovariateType.NONE,
        use_static_covariates: bool = True,
        load_ass_data=False,
        ass_sw_ins_mappings=None,
        mode="train"
    ):  
        """沿用父类初始化方法，保留股票和行业分类的映射"""
        
        self.dbaccessor = DbAccessor({})
        self.future_covariates = future_covariates
        
        super().__init__(target_series,covariates,input_chunk_length,output_chunk_length,shift,shift_covariates,
                         max_samples_per_ts,covariate_type,use_static_covariates,load_ass_data=load_ass_data,mode=mode)
        
        # 取得DataFrame数据，统一评估日期
        dataset = global_var.get_value("dataset")   
        if mode=="train":
            df_data_total = dataset.df_train          
        else:
            df_data_total = dataset.df_val 
        
        sw_indus = self.get_sw_industry()            
        # 只使用在行业分类索引中的目标数据
        df_data = df_data_total[df_data_total["instrument"].isin(sw_indus[:,0])]
        df_data["instrument"] = df_data["instrument"].astype(np.int32)
        g_col = dataset.get_group_rank_column()
        group_column = dataset.get_group_column()
        df_data[g_col] = df_data[group_column].rank(method='dense',ascending=True).astype("int")  
        # 映射target_series对应的索引位置
        codes = df_data[group_column].unique()
        combine_codes = IndustryMappingUtil.assc_series_and_codes(codes, target_series, dataset=dataset)
        self.keep_index = combine_codes[:,1]
        # 通过最小最大日期，生成所有交易日期列表
        datetime_col = dataset.get_datetime_index_column()   
        time_column = dataset.get_time_column()   
        date_list = df_data[datetime_col].unique()      
        date_list = np.sort(date_list)
        # 由于使用future_datetime方式对齐，因此从前面截取input序列长度,后面截取output序列长度用于预测长度预留
        date_list = date_list[input_chunk_length:-output_chunk_length]        
        self.date_list = date_list
        # 取得行业分类数量
        rank_num_max = df_data[g_col].unique().shape[0]
        date_mappings = np.ones([date_list.shape[0],rank_num_max])*(-1)
        # 遍历行业分类数据集，填入对应序号
        for index,date in enumerate(date_list):
            instrument_rank_nums = df_data[df_data[datetime_col]==date][[g_col,time_column]].values
            instrument_rank_nums[:,0] = instrument_rank_nums[:,0] - 1
            # 通过索引映射到指定数据上，而缺失的值仍保持-1的初始值
            date_mappings[index,instrument_rank_nums[:,0]] = instrument_rank_nums[:,1]     
        self.date_mappings = date_mappings.astype(np.int32)
        self.sw_ins_mappings = None

        # 预先定义数据形状
        self.total_instrument_num = rank_num_max
        self.target_num = target_num
        self.past_target_shape = [self.total_instrument_num,input_chunk_length,target_num]
        self.future_target_shape = [self.total_instrument_num,output_chunk_length,target_num]
        self.covariates_shape = [self.total_instrument_num,input_chunk_length,covariates[0].n_components]
        self.covariates_future_shape = [self.total_instrument_num,output_chunk_length,covariates[0].n_components]
        self.future_covariates_shape = [self.total_instrument_num,output_chunk_length,future_covariates[0].n_components]
        self.historic_future_covariates_shape = [self.total_instrument_num,input_chunk_length,future_covariates[0].n_components]
        self.static_covariate_shape = [self.total_instrument_num,target_series[0].static_covariates_values(copy=False).shape[-1]]
        self.last_targets_shape = [self.total_instrument_num,target_num]
        # Fake Shape
        self.sw_indus_shape = [rank_num_max,input_chunk_length+output_chunk_length]  
                
    def get_sw_industry(self):       
        """取得申万行业分类列表"""
        
        # 只使用指定行业分类数据
        sql = "select code,level from sw_industry where level=1 order by code asc"
        result_rows = self.dbaccessor.do_query(sql)    
        results = []
        for row in result_rows:
            code = row[0][:-3]
            results.append([code,row[1]])
        return np.array(results)
    
    def __len__(self):
        return self.date_list.shape[0]
            
    def __getitem__(
        self, idx
    ):
        """以日期为单位,整合行业分类数据"""
        
        date_mapping = self.date_mappings[idx]
        past_target_total = np.zeros(self.past_target_shape)
        past_covariate_total = np.zeros(self.covariates_shape)
        target_info_total = [None for _ in range(date_mapping.shape[0])]
        future_target_total = np.zeros(self.future_target_shape)
        static_covariate_total = np.zeros(self.static_covariate_shape)
        covariate_future_total = np.zeros(self.covariates_future_shape)
        future_covariates_total = np.zeros(self.future_covariates_shape)
        historic_future_covariates_total = np.zeros(self.historic_future_covariates_shape)
        target_class_total = np.ones(self.total_instrument_num)*-1
        target_class_total = target_class_total.astype(np.int8)
        raise_range_total = np.zeros(self.total_instrument_num)
        prev_raise_range_total = np.zeros(self.total_instrument_num)
        rank_data = np.zeros([self.total_instrument_num,2])
        
        ########### 生成行业数据 #########
        sw_date_mapping = self.date_mappings[idx]
        sw_indus_targets_total = np.ones(self.sw_indus_shape)*(-1)
        for index,ser_idx in enumerate(sw_date_mapping):
            if ser_idx==-1:
                continue            
            # 取得原序列索引进行series取数
            ori_index = self.keep_index[index]
            code = ori_index + 1
            target_series = self.target_series[ori_index]
            # 如果最后的序列号与当前序列号的差值不足序列输出长度，则忽略
            offset_end = target_series.time_index[-1] + 1 - ser_idx
            if offset_end<self.output_chunk_length:
                continue
            # 如果开始的序列号与当前序列号的差值不足序列输入长度，则忽略
            offset_begin = ser_idx - target_series.time_index[0]
            if offset_begin<self.input_chunk_length:
                continue          
            # if self.mode!="train":
            #     print("gggs")  
            # 目标序列
            target_vals = target_series.random_component_values(copy=False)[...,:self.target_num]
            # 对应的起始索引号属于future_datetime,因此减去序列输入长度，即为计算序列起始索引
            past_start = ser_idx - self.input_chunk_length
            # 需要从整体偏移量中减去起始偏移量，以得到并使用相对偏移量
            past_start = past_start - target_series.time_index[0]
            # 后续索引计算都以past_start为基准
            past_end = past_start + self.input_chunk_length
            future_start = past_end
            future_end = future_start + self.output_chunk_length
            covariate_start = past_start
            covariate_end = past_end

            # 预测目标，包括过去和未来数据
            future_target = target_vals[future_start:future_end]
            past_target = target_vals[past_start:past_end]
            # rank数值就是当前索引加1
            code = ori_index + 1
            try:
                instrument = self.ass_data[code][0]
            except Exception:
                print("eee")
            price_array = self.ass_data[code][2][past_start:future_end]
            datetime_array = self.ass_data[code][3][past_start:future_end]
            # 记录预测未来第一天的关联日期，用于后续数据对齐
            future_start_datetime = self.ass_data[code][3][past_end]
            
            # 辅助数据索引数据还需要加上偏移量，以恢复到原索引
            past_start_real = past_start+target_series.time_index[0]
            future_start_real = future_start+target_series.time_index[0]
            future_end_real = future_end+target_series.time_index[0]
            # 辅助数据索引数据还需要加上偏移量，以恢复到原索引
            target_info = {"item_rank_code":code,"instrument":instrument,"past_start":past_start,"past_end":past_end,
                               "future_start_datetime":future_start_datetime,"future_start":future_start_real,"future_end":future_end_real,
                               "price_array":price_array,"datetime_array":datetime_array,
                               "total_start":target_series.time_index.start,"total_end":target_series.time_index.stop}
            
            # 过去协变量序列数据
            covariate_series = self.covariates[ori_index] 
            raise_if_not(
                covariate_end <= len(covariate_series),
                f"The dataset contains covariates "
                f"that don't extend far enough into the future. ({idx}-th sample)",
            )
    
            covariate_total = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end + self.output_chunk_length
            ]
            # 过去协变量的过去数值以及未来数值
            covariate = covariate_total[:self.input_chunk_length]
            covariate_future = covariate_total[self.input_chunk_length:]
            raise_if_not(
                len(covariate)
                == (
                    self.output_chunk_length
                    if self.shift_covariates
                    else self.input_chunk_length
                ),
                f"The dataset contains covariates "
                f"whose time axis doesn't allow to obtain the input (or output) chunk relative to the "
                f"target series.",
            )
            # 静态协变量
            static_covariate = target_series.static_covariates_values(copy=False)
            # 未来协变量的过去值和未来值
            f_conv_values = self.future_covariates[ori_index].random_component_values(copy=False)
            future_covariate = f_conv_values[future_start:future_end]
            historic_future_covariate = f_conv_values[past_start:past_end]
            # 填充到总体数据中
            past_target_total[index] = past_target
            past_covariate_total[index] = covariate
            target_info_total[index] = target_info
            future_target_total[index] = future_target
            static_covariate_total[index] = static_covariate
            covariate_future_total[index] = covariate_future
            future_covariates_total[index] = future_covariate
            historic_future_covariates_total[index] = historic_future_covariate

            # 缩短未来计算周期，使用倒数第三天计算涨跌幅
            raise_range = (price_array[-3] - price_array[0])/price_array[0]*100
            p_target_class = get_simple_class(raise_range)
            target_class_total[index] = p_target_class
            
        ######### 分别对目标值和协变量，在总体范围层面进行归一化 #########
        
        real_index = np.where(target_class_total>=0)[0]
        real_past_target = past_target_total[real_index]
        real_future_target = future_target_total[real_index]
        # 目标值需要共用scaler         
        past_target_scale = []
        future_target_scale = []
        # flatten处理，在全范围进行归一化
        real_past_target_reshape = real_past_target.reshape(-1,real_past_target.shape[-1])
        real_future_reshape = real_future_target.reshape(-1,real_future_target.shape[-1])
        target_scaler = MinMaxScaler()
        target_scaler.fit(real_past_target_reshape)
        scale_data_past = target_scaler.transform(real_past_target_reshape)
        scale_data_past = scale_data_past.reshape(real_past_target.shape)
        scale_data_future = target_scaler.transform(real_future_reshape)
        scale_data_future = scale_data_future.reshape(real_future_target.shape)
        past_target_total[real_index] = scale_data_past
        future_target_total[real_index] = scale_data_future
        
        # 衡量未来值3个交易日内涨跌差值
        last_targets_total = future_target_total[:,2,:] - future_target_total[:,0,:]
        last_targets_total = MinMaxScaler().fit_transform(last_targets_total)
        
        # 过去协变量的归一化处理，整体范围进行
        real_past_conv = past_covariate_total[real_index]
        real_past_conv_reshape = real_past_conv.reshape(-1,real_past_conv.shape[-1])
        past_conv_scale = MinMaxScaler().fit_transform(real_past_conv_reshape)
        past_conv_scale = past_conv_scale.reshape(real_past_conv.shape)
        past_covariate_total[real_index] = past_conv_scale
        # 未来协变量和静态协变量已经归一化过了，不需要在此进行  
        
        # last_targets_total Shape is : [insturment_number,2,split_number]
        return past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,last_targets_total,sw_indus_targets_total,target_info_total                      
        

