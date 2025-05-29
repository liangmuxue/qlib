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
from sklearn.preprocessing import MinMaxScaler,StandardScaler

from darts.utils.data.shifted_dataset import GenericShiftedDataset,MixedCovariatesTrainingDataset
from darts.utils.data.utils import CovariateType
from darts.logging import raise_if_not
from darts import TimeSeries
from cus_utils.common_compute import normalization_except_outlier,interquartile_range
from tft.class_define import CLASS_VALUES,get_simple_class,get_complex_class

from cus_utils.db_accessor import DbAccessor
import cus_utils.global_var as global_var
from cus_utils.encoder_cus import rolling_norm
from tushare.stock.indictor import kdj
from .custom_dataset import CusGenericShiftedDataset
from .industry_mapping_util import FuturesMappingUtil
from cus_utils.log_util import AppLogger
from torch._jit_internal import ignore
logger = AppLogger()


class FuturesIndustryDataset(GenericShiftedDataset):
    """期货数据整合行业板块数据，形成多重数据架构"""
    
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
        scale_mode=None,
        cut_len=2,
        mode="train"
    ):  
        """兼顾期货品种和板块指数的映射"""

        super().__init__(target_series,covariates,input_chunk_length,output_chunk_length,shift,shift_covariates,
                         max_samples_per_ts,covariate_type,use_static_covariates)
                
        self.dbaccessor = DbAccessor({})
        self.future_covariates = future_covariates

        # 取得DataFrame数据，统一评估日期
        dataset = global_var.get_value("dataset")   
        # 生成目标索引，以及行业分类目标索引        
        fur_ins_mappings = self.build_accord_instrument_mapping(target_series,dataset=dataset)        
        self.sw_ins_mappings = fur_ins_mappings
        self.indus_index = FuturesMappingUtil.get_industry_data_index(self.sw_ins_mappings)
        self.indus_rel_index = FuturesMappingUtil.get_industry_rel_index(self.sw_ins_mappings)
        self.ins_in_indus_index = FuturesMappingUtil.get_industry_instrument(self.sw_ins_mappings,without_main=False)
        self.instrument_index = FuturesMappingUtil.get_instrument_index(self.sw_ins_mappings)
        self.ins_codes = FuturesMappingUtil.get_combine_industry_instrument(self.sw_ins_mappings)[:,-1]
        self.main_index = FuturesMappingUtil.get_main_index(self.sw_ins_mappings)
        self.main_index_rel = FuturesMappingUtil.get_main_index_in_indus(self.sw_ins_mappings)
        
        if mode=="train":
            df_data_total = dataset.df_train          
        else:
            df_data_total = dataset.df_val 
        
        instrument_df = self.get_variety_list_with_indus()
        # 包含品种编码和行业板块编码
        codes_query = np.concatenate([instrument_df["code"].values,instrument_df["indus_code"].values,['ZS_ALL']])
        # 只使用在数据表中的目标数据
        df_data = df_data_total[df_data_total["instrument"].isin(codes_query)]        
        # df_data = df_data_total[df_data_total["instrument"].isin(self.ins_codes)]
        df_data = df_data.sort_values(by=["instrument","datetime_number"],ascending=True)
        g_col = dataset.get_group_rank_column()
        group_column = dataset.get_group_column()
        # 重新生成编号
        df_data[g_col] = df_data[group_column].rank(method='dense',ascending=True).astype("int")  
        self.df_data = df_data
        # 通过最小最大日期，生成所有交易日期列表
        datetime_col = dataset.get_datetime_index_column()   
        time_column = dataset.get_time_column()   
        date_list = df_data[datetime_col].unique()      
        date_list = np.sort(date_list)
        # 由于使用future_datetime方式对齐，因此从前面截取input序列长度,后面截取output序列长度用于预测长度预留
        date_list = date_list[input_chunk_length:-output_chunk_length]        
        self.date_list = date_list
        # 取得行业和品种分类数量
        rank_num_max = df_data['instrument'].unique().shape[0]
        date_mappings = np.ones([date_list.shape[0],rank_num_max])*(-1)
        # 遍历行业分类数据集，填入对应序号
        for index,date in enumerate(date_list):
            instrument_rank_nums = df_data[df_data[datetime_col]==date][[g_col,time_column]].values
            instrument_rank_nums[:,0] = instrument_rank_nums[:,0] - 1
            # 通过索引映射到指定数据上，而缺失的值仍保持-1的初始值
            date_mappings[index,instrument_rank_nums[:,0]] = instrument_rank_nums[:,1]     
        self.date_mappings = date_mappings.astype(np.int32)

        # 预先定义数据形状
        self.total_instrument_num = rank_num_max 
        self.keep_index = [i for i in range(self.total_instrument_num)] # 实际索引对照
        self.target_num = target_num
        self.past_target_shape = [self.total_instrument_num,input_chunk_length,target_num]
        self.future_target_shape = [self.total_instrument_num,output_chunk_length,target_num]
        self.covariates_shape = [self.total_instrument_num,input_chunk_length,covariates[0].n_components]
        self.covariates_future_shape = [self.total_instrument_num,output_chunk_length,covariates[0].n_components]
        self.future_covariates_shape = [self.total_instrument_num,output_chunk_length,future_covariates[0].n_components]
        self.historic_future_covariates_shape = [self.total_instrument_num,input_chunk_length,future_covariates[0].n_components]
        self.static_covariate_shape = [self.total_instrument_num,target_series[0].static_covariates_values(copy=False).shape[-1]]
        # Fake Shape
        self.sw_indus_shape = [rank_num_max,input_chunk_length+output_chunk_length]  
        
        # 创建整体目标涨跌评估数据,不进行归一化
        self.cut_len = cut_len
        self.total_target_vals = self.build_total_tar_scale_data(target_series,scale_mode=scale_mode,cut_len=self.output_chunk_length)
        # indus_target_vals = []
        # for k,instruments in enumerate(self.ins_in_indus_index):
        #     target_series_ins = target_series[instruments]
        #     indus_target_vals.append(self.build_total_tar_scale_data(target_series_ins,scale_mode=scale_mode,cut_len=self.output_chunk_length))

        self.mode = mode
        # 目标归一化模式： 0:单位目标归一化，round目标不归一化 2:单位目标归一化，round目标使用百分比幅度，以及未来值二次归一化 3:单位目标忽略，round目标使用百分比幅度,未来值一次归一化 
        self.scale_mode = scale_mode
        # 组建辅助数据
        load_ass_data = global_var.get_value("load_ass_data")
        if load_ass_data:
            # 如果配置为加载模式，则从全局变量透传数据
            self.ass_data = global_var.get_value("ass_data_{}".format(mode))
        else:
            # 直接从数据集中创建
            self.ass_data = {}
            for series in self.target_series:
                code = int(series.static_covariates["instrument_rank"].values[0])
                instrument = global_var.get_value("dataset").get_group_code_by_rank(code)
                price_array = df_data[(df_data["time_idx"]>=series.time_index.start)&(df_data["time_idx"]<series.time_index.stop)
                                    &(df_data["instrument_rank"]==code)]["label_ori"].values
                datetime_array = df_data[(df_data["time_idx"]>=series.time_index.start)&(df_data["time_idx"]<series.time_index.stop)
                                    &(df_data["instrument_rank"]==code)]["datetime_number"].values                                
                diff_range = df_data[(df_data["time_idx"]>=series.time_index.start)&(df_data["time_idx"]<series.time_index.stop)
                                    &(df_data["instrument_rank"]==code)]["diff_range"].values         
                rsv_diff = df_data[(df_data["time_idx"]>=series.time_index.start)&(df_data["time_idx"]<series.time_index.stop)
                                    &(df_data["instrument_rank"]==code)]["rsv_diff"].values                                           
                self.ass_data[code] = (instrument,diff_range,price_array,datetime_array,rsv_diff)
            # 保存到本地
            save_ass_data = global_var.get_value("save_ass_data")
            if save_ass_data:
                # 保存辅助数据
                ass_data_path = os.path.join(global_var.get_value("ass_data_path"),"ass_data_{}.pkl".format(mode))
                with open(ass_data_path, "wb") as fout:
                    pickle.dump(self.ass_data, fout) 
                                    
    def get_variety_list_with_indus(self):       
        """取得期货品种列表,包含行业板块"""
        
        sql = "select v.code,upper(concat('zs_',i.code)) as indus_code,v.name as name from trading_variety v left join futures_industry i on v.industry_id = i.id" \
            " order by v.code asc"
        result_rows = self.dbaccessor.do_query(sql)    
        columns = ["code","indus_code","name"]
        result_rows = [[row[i] for i in range(len(row))]  for row in result_rows]
        result_rows = np.array(result_rows)
        result = pd.DataFrame(result_rows,columns=columns)
        return result

    def build_accord_instrument_mapping(self,target_series,dataset=None):
        """筛选出符合条件的品种，以及分类映射关系"""

        # 取得相关期货品种以及对应行业板块
        instrument_df = self.get_variety_list_with_indus()     
        fur_indus = self.get_fur_industry()
        fur_indus_df = pd.DataFrame(fur_indus,columns=["code","name"])
        # 插入整体指标数据
        fur_indus_df = fur_indus_df.append(pd.DataFrame(np.array([['ZS_all','综合']]),columns=["code","name"]),ignore_index=True)
        # 生成继承映射关系对象
        sw_ins_mappings = FuturesMappingUtil.build_accord_mapping(target_series, fur_indus_df, instrument_df, dataset=dataset)
        return sw_ins_mappings
    
    def get_fur_industry(self):       
        """取得板块分类"""
        
        sql = "select upper(concat('ZS_',code)),name from futures_industry where delete_flag=0 order by code asc"
        result_rows = self.dbaccessor.do_query(sql)    
        results = [[row[0],row[1]] for row in result_rows]
        return np.array(results)
    
    def get_round_norm_mm(self,target_series):
        """取得目标差分归一化的参考最大最小值"""

        total_target_vals = []
        mm_values = np.zeros([len(target_series),2,self.target_num])
        index_range = []
        begin_index = 0
        
        for i in range(len(target_series)):
            ts = target_series[i]
            
            for j in range(self.target_num):
                target_vals = ts.random_component_values(copy=False)[...,j]
                _,round_vals = rolling_norm(target_vals,self.input_chunk_length+self.output_chunk_length,self.output_chunk_length)
                mm_values[i,:,j] = np.array([round_vals.min(),round_vals.max()])
        
        return mm_values
    
    def build_total_tar_scale_data(self,target_series,cut_len=1,scale_mode=None):
        """创建整体目标涨跌评估数据的归一化数据,整体归一化"""
        
        # 拆分分离出行业数值以及整体指数数值
        target_series_ins = [target_series[index] for index in self.instrument_index]
        target_series_indus = [target_series[index] for index in self.indus_index]
        target_series_main = [target_series[self.main_index]]
        # 具体计算
        combine_values_ins = self.build_total_tar_scale_data_inner(target_series_ins, cut_len, scale_mode)
        combine_values_indus = self.build_total_tar_scale_data_inner(target_series_indus, cut_len, scale_mode)
        combine_values_main = self.build_total_tar_scale_data_inner(target_series_main, cut_len, scale_mode)
        # 根据索引映射合并回原数组
        combine_values_rebuild = []
        for k in range(len(target_series)):
            if k in self.instrument_index:
                ins_index = np.where(self.instrument_index==k)[0][0]
                combine_values_rebuild.append(combine_values_ins[ins_index])        
            elif k in self.indus_index:
                indus_index = np.where(self.indus_index==k)[0][0]
                combine_values_rebuild.append(combine_values_indus[indus_index])   
            elif k==self.main_index:
                combine_values_rebuild.append(combine_values_main[0])   
            else:
                combine_values_rebuild.append(None)
        return combine_values_rebuild
                        
    def build_total_tar_scale_data_inner(self,target_series,cut_len=1,scale_mode=None):
        """创建整体目标涨跌评估数据的归一化数据,整体归一化"""
        
        eps = 1e-5
        total_target_vals = []
        combine_values = []
        range_values = []
        index_range = []
        begin_index = 0
        for i in range(len(target_series)):
            ts = target_series[i]
            target_vals = ts.random_component_values(copy=False)[...,:self.target_num]
            # 分别生成整体幅度，和最后一天的幅度
            target_vals_begin = target_vals[1:-cut_len]
            target_vals_end = target_vals[cut_len+1:]
            # 计算跨预测区域差值
            combine_value = target_vals_end - target_vals_begin
            combine_value = target_vals
            combine_values.append(combine_value)    
            # range_value = combine_value/target_vals_begin
            # range_values.append(range_value)
            # 累加起始索引，并记录当前索引范围
            end_index = begin_index + combine_value.shape[0]
            index_range.append([begin_index,end_index])
            begin_index = end_index
            
        combine_values = np.concatenate(combine_values,axis=0)
        index_range = np.array(index_range)   
        # 根据单独指标决定如何归一化
        for i in range(combine_values.shape[-1]):
            if scale_mode[i] in [0]:
                # 剔除异常值
                combine_values[:,i] = interquartile_range(combine_values[:,i])
                # 直接归一化
                # combine_values[:,i] = MinMaxScaler(feature_range=(eps, 1)).fit_transform(np.expand_dims(combine_values[:,i],-1)).squeeze(-1)             
            if scale_mode[i] in [1,2,3]:
                # 剔除异常值
                combine_values[:,i] = interquartile_range(combine_values[:,i])
                # 直接归一化
                # combine_values[:,i] = MinMaxScaler(feature_range=(eps, 1)).fit_transform(np.expand_dims(combine_values[:,i],-1)).squeeze(-1) 
            elif scale_mode[i]==5:
                # 只剔除异常值，不归一化
                combine_values[:,i] = interquartile_range(combine_values[:,i])
            # 0值替换为非0
            zero_index = np.where(combine_values[:,i]==0)[0]
            eps_adju = np.random.uniform(low=eps,high=eps*10,size=zero_index.shape[0])
            combine_values[:,i][zero_index] = eps_adju 
            
        # 还原到多个品种维度模式
        for i in range(index_range.shape[0]):
            idx_range = index_range[i]
            data_item = combine_values[idx_range[0]:idx_range[1],:]
            # 填充空白值
            # data_item = np.pad(data_item,((2,cut_len-1),(0,0)),mode='minimum')             
            total_target_vals.append(data_item)
          
        return total_target_vals  
    

    
               
    def __len__(self):
        return self.date_list.shape[0]


    def __getitem__(
        self, idx
    ):
        """以日期为单位,整合行业分类数据"""
        
        past_target_total = np.zeros(self.past_target_shape)
        past_covariate_total = np.zeros(self.covariates_shape)
        target_info_total = [None for _ in range(self.past_target_shape[0])]
        future_target_total = np.zeros(self.future_target_shape)
        static_covariate_total = np.zeros(self.static_covariate_shape)
        covariate_future_total = np.zeros(self.covariates_future_shape)
        future_covariates_total = np.zeros(self.future_covariates_shape)
        historic_future_covariates_total = np.zeros(self.historic_future_covariates_shape)
        target_class_total = np.ones(self.total_instrument_num)*-1
        target_class_total = target_class_total.astype(np.int8)
        round_targets = np.zeros([self.total_instrument_num,self.input_chunk_length + self.output_chunk_length,self.past_target_shape[-1]])
        long_diff_targets = np.zeros([self.total_instrument_num,self.input_chunk_length,self.past_target_shape[-1]])
        price_targets = np.zeros([self.total_instrument_num,self.input_chunk_length + self.output_chunk_length])
        price_targets_ori = np.zeros([self.total_instrument_num,self.input_chunk_length + self.output_chunk_length])
        
        ########### 生成行业数据 #########
        sw_date_mapping = self.date_mappings[idx]
        
        for index,ser_idx in enumerate(sw_date_mapping):
            if ser_idx==-1:
                continue     
            # 取得原序列索引进行series取数,目前一致
            ori_index = index
            if self.total_target_vals[ori_index] is None:
                continue
            code = ori_index + 1
            target_series = self.target_series[ori_index]
            # 如果最后的序列号与当前序列号的差值不足序列输出长度(注意，这里以实际最大序列编号来衡量)，则忽略
            offset_end = target_series.time_index.stop - ser_idx
            if offset_end<self.output_chunk_length:
                continue
            # 如果开始的序列号与当前序列号的差值不足序列输入长度，则忽略
            offset_begin = ser_idx - target_series.time_index[0]
            if offset_begin<self.input_chunk_length:
                continue
            # 使用实际股票对照索引（刨除指数索引）        
            keep_index = self.keep_index[index]  
            # 目标序列
            target_vals = target_series.random_component_values(copy=False)[...,:self.target_num]
            # 对应的起始索引号属于future_datetime,因此减去序列输入长度，即为计算序列起始索引
            past_start = ser_idx - self.input_chunk_length
            past_end = past_start + self.input_chunk_length
            future_start = past_end
            future_end = future_start + self.output_chunk_length
            # 对于series序列，需要从整体偏移量中减去起始偏移量，以得到并使用相对偏移量。（注意对于协变量序列，仍然沿用原始编号）
            past_start_ser = past_start - target_series.time_index[0]
            # 后续索引计算都以past_start为基准
            past_end_ser = past_start_ser + self.input_chunk_length
            future_start_ser = past_end_ser
            future_end_ser = future_start_ser + self.output_chunk_length
            
            # 记录预测未来第一天的关联日期，用于后续数据对齐
            future_start_datetime = self.ass_data[code][3][past_end_ser]
            # 取得整体评估量化数据
            total_target_vals = self.total_target_vals[ori_index][past_start_ser:future_end_ser]
            round_targets[keep_index] = total_target_vals    
            # 使用最后一个目标值分别与所有前值做差分
            long_diff_targets[keep_index] = [abs(total_target_vals[-1] - total_target_vals[i]) for i in range(self.input_chunk_length) ]
            # 取得最后一段涨跌幅度评估
            # 预测目标，包括过去和未来数据
            future_target = target_vals[future_start_ser:future_end_ser]
            past_target = target_vals[past_start_ser:past_end_ser]
            # rank数值就是当前索引加1
            code = ori_index + 1
            instrument = self.ass_data[code][0]
            price_array = self.ass_data[code][2][past_start_ser:future_end_ser]
            diff_range = self.ass_data[code][1][past_start_ser:future_end_ser]
            rsv_diff = self.ass_data[code][4][past_start_ser:future_end_ser]
            scaler = MinMaxScaler(feature_range=(1e-5, 1))
            scaler.fit(np.expand_dims(price_array[:self.input_chunk_length],-1))
            price_targets[keep_index] = scaler.transform(np.expand_dims(price_array,-1)).squeeze()       
            price_targets_ori[keep_index] = price_array    
            datetime_array = self.ass_data[code][3][past_start_ser:future_end_ser]
            # 辅助数据索引数据还需要加上偏移量，以恢复到原索引
            target_info = {"item_rank_code":code,"instrument":instrument,"past_start":past_start,"past_end":past_end,
                               "future_start_datetime":future_start_datetime,"future_start":future_start,"future_end":future_end,
                               "price_array":price_array,"diff_range":diff_range,"datetime_array":datetime_array,"rsv_diff":rsv_diff,
                               "total_start":target_series.time_index.start,"total_end":target_series.time_index.stop}
          
            # 过去协变量序列数据
            covariate_series = self.covariates[ori_index] 
            raise_if_not(
                past_end <= len(covariate_series),
                f"The dataset contains covariates "
                f"that don't extend far enough into the future. ({idx}-th sample)",
            )
    
            covariate_total = covariate_series.random_component_values(copy=False)[
                past_start:future_end
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
            
            past_target_total[keep_index] = past_target
            past_covariate_total[keep_index] = covariate
            target_info_total[keep_index] = target_info
            future_target_total[keep_index] = future_target
            static_covariate_total[keep_index] = static_covariate
            covariate_future_total[keep_index] = covariate_future
            future_covariates_total[keep_index] = future_covariate
            historic_future_covariates_total[keep_index] = historic_future_covariate

            # 缩短未来计算周期，使用倒数第1天计算涨跌幅
            raise_range = (price_array[-1] - price_array[-self.output_chunk_length-1])/price_array[-self.output_chunk_length-1]*100
            p_target_class = get_simple_class(raise_range)
            target_class_total[keep_index] = p_target_class
            
        ######### 分别对目标值和协变量，在个体范围层面进行归一化 #########
        
        real_index = np.where(target_class_total>=0)[0]
        real_past_target = past_target_total[real_index]
        real_future_target = future_target_total[real_index]
        # 目标值逐个进行归一化,过去和未来值需要共用scaler   
        for i in range(real_past_target.shape[0]):
            real_past_target_item = real_past_target[i]
            real_future_reshape_item = real_future_target[i]
            target_scaler = MinMaxScaler(feature_range=(1e-5, 1))
            target_scaler.fit(real_past_target_item)
            scale_data_past = target_scaler.transform(real_past_target_item)
            scale_data_future = target_scaler.transform(real_future_reshape_item)
            past_target_total[real_index[i]] = scale_data_past
            future_target_total[real_index[i]] = scale_data_future
        
        # 过去协变量的归一化处理，逐个进行
        real_covariate_total= past_covariate_total[real_index]
        for i in range(real_covariate_total.shape[0]):
            real_past_conv = real_covariate_total[i]
            past_conv_scale = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(real_past_conv)
            past_covariate_total[real_index[i]] = past_conv_scale
            
        # 未来协变量和静态协变量已经归一化过了，不需要在此进行  
        
        # 整体目标数据拆分为过去值和目标值
        past_round_targets = round_targets[:,:self.input_chunk_length,:]
        future_round_targets = round_targets[:,self.input_chunk_length:,:]
        
        # 行业板块数据归一化
        past_index_round_targets = np.zeros([self.ins_in_indus_index.shape[0],self.input_chunk_length,self.past_target_shape[-1]])
        future_index_round_targets = np.zeros([self.ins_in_indus_index.shape[0],self.output_chunk_length,self.past_target_shape[-1]])
        long_diff_seq_targets = long_diff_targets[:,-self.cut_len:,:]
        
        for i in range(self.ins_in_indus_index.shape[0]):
            # 取得对应的行业序列数据，作为目标数据
            indus_index = self.indus_index[i]
            indus_past_round = past_round_targets[indus_index].copy()
            indus_future_round = future_round_targets[indus_index].copy()
            past_index_round_targets[i] = indus_past_round
            future_index_round_targets[i] = indus_future_round
        
            
        for i in range(self.past_target_shape[-1]):
            if self.scale_mode[i] in [0,1]:
                # 分行业归一化
                scaler = MinMaxScaler(feature_range=(1e-5, 1)).fit(past_index_round_targets[...,i].transpose(1,0)) 
                past_index_round_targets[...,i] = scaler.transform(past_index_round_targets[...,i].transpose(1,0)).transpose(1,0)
                future_index_round_targets[...,i] = scaler.transform(future_index_round_targets[...,i].transpose(1,0)).transpose(1,0)               
                long_diff_seq_targets[:,:,i] = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(long_diff_seq_targets[:,:,i].transpose(1,0)).transpose(1,0)
                
        # 单独归一化未来行业板块整体预测数值
        for i in range(self.past_target_shape[-1]):
            if self.scale_mode[i] in [0]:
                future_index_round_targets[self.indus_rel_index,:,i] = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(future_index_round_targets[self.indus_rel_index,:,i])
        # 合并过去行业整体数值的归一化形态，与未来目标数值的单独形态
        index_round_targets = np.concatenate([past_index_round_targets,future_index_round_targets],axis=1)

        # 整体目标值批次内,分行业归一化
        past_data_scale = np.zeros([self.total_instrument_num,self.input_chunk_length,self.past_target_shape[-1]])
        for idx,ins_indus_index in enumerate(self.ins_in_indus_index):
            # 忽略无效数值
            target_class_ins = target_class_total[ins_indus_index]
            keep_index = np.where(target_class_ins>=0)[0]
            if keep_index.shape[0]==0:
                continue         
            index_real = ins_indus_index[keep_index]
            past_data = past_round_targets[index_real,:,:]
            future_data = future_round_targets[index_real,:,:]        
                   
            for i in range(self.past_target_shape[-1]):
                # 在全品种缩放模式下，对整体指数下的品种进行归一化，忽略行业内部的归一化。在行业品种缩放模式下，针对每个行业内部的品种进行归一化
                if (self.scale_mode[i]==0 and idx!=self.main_index_rel) or (self.scale_mode[i]==2 and idx==self.main_index_rel):
                    # 使用过去数值参考,进行第一次归一化
                    for k in index_real:
                        past_data_item = past_round_targets[k,:,i:i+1]
                        # 过滤全部趋近于0的数据
                        if past_data_item.max()<1e-4:
                            continue
                        scaler = MinMaxScaler(feature_range=(1e-5, 1)).fit(past_data_item)
                        past_data_scale[k,:,i] = scaler.transform(past_data_item).squeeze(-1)
                        scale_data = scaler.transform(future_round_targets[k,:,i:i+1]).squeeze(-1)
                        future_round_targets[k,:,i] = scale_data
                    # 针对目标值，再次归一化以加速收敛                    
                    future_round_targets[index_real,:,i] = MinMaxScaler(feature_range=(1e-5, 1)).fit_transform(future_round_targets[index_real,:,i])
                # 对于数值模式，只进行一次过去未来的归一化，区分全品种和行业内部的归一化操作
                if (self.scale_mode[i]==1 and idx!=self.main_index_rel) or (self.scale_mode[i]==3 and idx==self.main_index_rel):
                    for k in index_real:
                        past_data_item = past_round_targets[k,:,i:i+1]
                        # 过滤全部趋近于0的数据
                        if past_data_item.max()<1e-4:
                            continue
                        scaler = MinMaxScaler(feature_range=(1e-5, 1)).fit(past_data_item)
                        past_data_scale[k,:,i] = scaler.transform(past_data_item).squeeze(-1)
                        scale_data= scaler.transform(future_round_targets[k,:,i:i+1]).squeeze(-1)
                        future_round_targets[k,:,i] = scale_data
        past_future_round_targets = np.concatenate([past_data_scale,future_round_targets],axis=1)

        # if future_start_datetime==20220613:
        #     result_file_path = "custom/data/results/data_compare_val_20220613.pkl"
        #     results = [target_info_total,past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total
        #                ,covariate_future_total,past_future_round_targets,index_round_targets]
        #     with open(result_file_path, "wb") as fout:
        #         pickle.dump(results, fout)    
        #     print("ggg")  
                
        return past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,index_round_targets,long_diff_seq_targets,target_info_total 
                            

