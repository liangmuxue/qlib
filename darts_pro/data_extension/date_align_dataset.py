from darts.utils.data.sequential_dataset import (
    SplitCovariatesTrainingDataset,
)

import pickle
import os
from typing import Optional, List, Tuple, Union
import numpy as np
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

import cus_utils.global_var as global_var
from cus_utils.encoder_cus import transform_slope_value
from tushare.stock.indictor import kdj
from .custom_dataset import CusGenericShiftedDataset

from cus_utils.log_util import AppLogger
logger = AppLogger()

class DateShiftedDataset(CusGenericShiftedDataset):
    """基于日期对齐的滚动数据集"""
    
    def __init__(
        self,
        target_series,
        covariates=None,
        future_covariates=None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        shift_covariates: bool = False,
        max_samples_per_ts: Optional[int] = None,
        covariate_type: CovariateType = CovariateType.NONE,
        use_static_covariates: bool = True,
        load_ass_data=False,
        mode="train"
    ):
        """数据初始化，基于日期进行数据对齐"""
        
        super().__init__(target_series,covariates,input_chunk_length,output_chunk_length,shift,shift_covariates,
                         max_samples_per_ts,covariate_type,use_static_covariates,load_ass_data=load_ass_data,mode=mode)
        self.future_covariates = future_covariates
        
        # 取得DataFrame数据，统一评估日期
        dataset = global_var.get_value("dataset")
        if mode=="train":
            df_data = dataset.df_train          
        else:
            df_data = dataset.df_val        
        
        datetime_col = dataset.get_datetime_index_column()
        g_col = dataset.get_group_rank_column()
        time_column = dataset.get_time_column()
        
        # 通过最小最大日期，生成所有交易日期列表
        date_list = df_data[datetime_col].unique()
        date_list = np.sort(date_list)
        cut_len = input_chunk_length + output_chunk_length
        # 由于使用future_datetime方式对齐，因此从前面截取input序列长度,后面截取output序列长度用于预测长度预留
        date_list = date_list[input_chunk_length:-output_chunk_length]
        # 根据排序号进行日期分组映射数据的初始化
        rank_num_max = len(target_series)
        date_mappings = np.ones([date_list.shape[0],rank_num_max])*(-1)
        # 循环每个日期，并查询填充对应的股票索引信息
        logger.info("process date_list begin,mode:{}".format(mode))
        for index,date in enumerate(date_list):
            instrument_rank_nums = df_data[df_data[datetime_col]==date][[g_col,time_column]].values
            instrument_rank_nums = instrument_rank_nums[np.argsort(instrument_rank_nums[:,0])]
            # 由于rank编号从1开始，因此在这里进行修正
            instrument_rank_nums[:,0] = instrument_rank_nums[:,0] - 1
            # 通过索引映射到指定数据上，而缺失的值仍保持-1的初始值
            date_mappings[index,instrument_rank_nums[:,0]] = instrument_rank_nums[:,1]
        logger.info("process date_list end,mode:{}".format(mode))
        self.date_mappings = date_mappings.astype(np.int16)
        self.date_list = date_list
        # 预先定义数据形状
        self.total_instrument_num = rank_num_max
        self.past_target_shape = [self.total_instrument_num,input_chunk_length,target_series[0].n_components]
        self.future_target_shape = [self.total_instrument_num,output_chunk_length,target_series[0].n_components]
        self.covariates_shape = [self.total_instrument_num,input_chunk_length,covariates[0].n_components]
        self.covariates_future_shape = [self.total_instrument_num,output_chunk_length,covariates[0].n_components]
        self.future_covariates_shape = [self.total_instrument_num,output_chunk_length,future_covariates[0].n_components]
        self.historic_future_covariates_shape = [self.total_instrument_num,input_chunk_length,future_covariates[0].n_components]
        self.static_covariate_shape = [self.total_instrument_num,target_series[0].static_covariates_values(copy=False).shape[-1]]
        self.last_targets_shape = [self.total_instrument_num,target_series[0].n_components]
        
    def __getitem__(
        self, idx
    ):
        """以日期为单位,整合当日所有股票数据"""
        
        # logger.info("__getitem__ in,index:{}".format(idx))
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
        last_targets_total = np.zeros(self.last_targets_shape)
        
        for index,ser_idx in enumerate(date_mapping):
            # 如果当日没有记录则相关变量保持为0或空
            if ser_idx==-1:
                continue
            target_series = self.target_series[index]
            # 如果最后的序列号与当前序列号的差值不足序列输出长度，则忽略
            offset_end = target_series.time_index[-1] + 1 - ser_idx
            if offset_end<self.output_chunk_length:
                continue
            # 如果开始的序列号与当前序列号的差值不足序列输入长度，则忽略
            offset_begin = ser_idx - target_series.time_index[0]
            if offset_begin<self.input_chunk_length:
                continue            
            # 目标序列
            target_vals = target_series.random_component_values(copy=False)
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
            code = index + 1
            instrument = self.ass_data[code][0]
            label_array = self.ass_data[code][1][past_start:future_end]
            price_array = self.ass_data[code][2][past_start:future_end]
            datetime_array = self.ass_data[code][3][past_start:future_end]
            # 记录预测未来第一天的关联日期，用于后续数据对齐
            future_start_datetime = self.ass_data[code][3][past_end]
            
            # 辅助数据索引数据还需要加上偏移量，以恢复到原索引
            past_start_real = past_start+target_series.time_index[0]
            future_start_real = future_start+target_series.time_index[0]
            future_end_real = future_end+target_series.time_index[0]
            # 存储辅助指标
            kdj_k,kdj_d,kdj_j,rsi_20,rsi_5,macd_diff,macd_dea = self.ass_data[code][4:]
            kdj_k = kdj_k[past_start_real:future_end_real]
            kdj_d = kdj_d[past_start_real:future_end_real]
            kdj_j = kdj_j[past_start_real:future_end_real]        
            rsi_20 = rsi_20[past_start_real:future_end_real]
            rsi_5 = rsi_5[past_start_real:future_end_real]
            macd_diff = macd_diff[past_start_real:future_end_real]  
            macd_dea = macd_dea[past_start_real:future_end_real]
            # 辅助数据索引数据还需要加上偏移量，以恢复到原索引
            target_info = {"item_rank_code":code,"instrument":instrument,"past_start":past_start,"past_end":past_end,
                               "future_start_datetime":future_start_datetime,"future_start":future_start_real,"future_end":future_end_real,
                               "price_array":price_array,"datetime_array":datetime_array,
                               "kdj_k":kdj_k,"kdj_d":kdj_d,"kdj_j":kdj_j,"rsi_20":rsi_20,"rsi_5":rsi_5,"macd_diff":macd_diff,"macd_dea":macd_dea,
                               "total_start":target_series.time_index.start,"total_end":target_series.time_index.stop}
            
            covariate_series = self.covariates[index] 
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
            f_conv_values = self.future_covariates[index].random_component_values(copy=False)
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
            
            # 计算涨跌幅类别
            price_array = target_info["price_array"][self.input_chunk_length-1:]
            raise_range = (price_array[-1] - price_array[0])/price_array[0]*100
            p_target_class = get_simple_class(raise_range)
            target_class_total[index] = p_target_class

        # 分别对目标值和协变量，以日期为单位实现整体归一化
        # 忽略没有数值的部分
        real_index = np.where(target_class_total>=0)[0]
        # 最后一段涨幅的归一化处理
        real_future_target = future_target_total[real_index]
        real_past_target = past_target_total[real_index]
        last_target = real_future_target[:,-1,:] - real_future_target[:,-2,:]
        last_targets_total[real_index] = MinMaxScaler().fit_transform(last_target.reshape(-1, last_target.shape[-1])).reshape(last_target.shape)  
        # 目标值需要共用scaler         
        target_scaler = MinMaxScaler()
        past_target_total[real_index] = target_scaler.fit_transform(real_past_target.reshape(-1, real_past_target.shape[-1])).reshape(real_past_target.shape)
        future_target_total[real_index] = target_scaler.transform(real_future_target.reshape(-1, real_future_target.shape[-1])).reshape(real_future_target.shape)
        # 过去协变量的归一化处理
        real_past_conv = past_covariate_total[real_index]
        past_covariate_total[real_index] = MinMaxScaler().fit_transform(
            real_past_conv.reshape(-1, real_past_conv.shape[-1])).reshape(real_past_conv.shape)
        # 未来协变量和静态协变量已经归一化过了，不需要在此进行  
                             
        return past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,last_targets_total,target_info_total

    def __len__(self):
        return self.date_list.shape[0]
        
        
