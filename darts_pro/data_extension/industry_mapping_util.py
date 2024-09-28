import pickle
import os
from typing import Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from numba.core.types import none
import torch

from cus_utils.db_accessor import DbAccessor
from darts.logging import raise_if_not
from httpx._status_codes import codes

code_level = "sw_first_code"

class IndustryMappingUtil:
    """用于行业分类的功能类"""
    
    """行业分类和股票映射关系对象数据结构规范
       类型为numpy数组，形状： [行业分类数量，5]
       其中第1列(sw_index)： 行业分类对应的序列（TargetSeries）排序号（排序号非连续），类型为int
          第2列(sw_code)： 行业分类编码，类型为int
          第3列(instrument_rank)： 当前行业分类下的股票对排序号数组（排序号连续），数组元素类型为int
          第4列(instrument_index)： 当前行业分类下的股票对应的序列（TargetSeries）排序号数组（排序号非连续），数组元素类型为int
          第5列(instrument_code)： 当前行业分类下的股票代码数组，数组长度需要与第三列中的每个数组长度一致，数组元素类型为int
          第6列(instrument_ava_flag)： 对应标志当前行业分类下的股票是否可用的mask数组,数组长度需要与第三列中的每个数组长度一致，数组元素类型为int，取值：0 不可用 1 可用
    """
    
    @staticmethod
    def build_accord_instrument_mapping(target_series,sw_indus_df=None,instrument_df=None,dataset=None):
        """筛选出符合条件的股票，以及分类映射关系
            Params:
              target_series： 目标序列，包含股票数据和行业分类数据
              sw_indus_df 行业分类数据集，columns=["code","level"]  
              instrument_df 股票名称编码数据集
            Return:
              sw_ins_mappings: 行业分类和股票映射关系对象，具体规范参考类注释的说明
        """
        
        g_col = dataset.get_group_rank_column()
        sw_indus = sw_indus_df.values
        # 同时找出申万分类数据索引
        keep_index = []
        target_codes = []
        rank_codes = []
        sw_industry_index = []
        sw_industry_codes = []
        for index,ts in enumerate(target_series):
            rank_code = int(ts.static_covariates[g_col].values[0])
            code = dataset.get_group_code_by_rank(rank_code)
            # 匹配后记录索引值
            if np.any(instrument_df["code"]==code):
                keep_index.append([code,index])
                target_codes.append(code)
                rank_codes.append(rank_code)
            # 同时记录类别序号,用于后续target_series的取数对照
            if np.any(sw_indus[:,0]==code):
                sw_industry_index.append([code,index])
                # 保存当前行业分类的时间序列长度，以排查指标日期不全的分类数据
                size = ts.time_index.stop - ts.time_index.start 
                sw_industry_codes.append([code,size])
        keep_index = np.array(keep_index).astype(np.int32)        
        sw_industry_index = np.array(sw_industry_index).astype(np.int32)  
        sw_industry_codes = np.array(sw_industry_codes).astype(np.int32)     
        max_size = sw_industry_codes[:,1].max()
        # 清除日期不全的分类
        sw_industry_codes = sw_industry_codes[sw_industry_codes[:,1]==max_size][:,0]
        # 同时从股票及分类列表中删除不在数据集范围的数据
        instrument_df = instrument_df[instrument_df["code"].isin(target_codes)]
        # 联合筛选行业数据集，取得交集，去掉不在真正序列中的部分
        sec_codes = instrument_df[code_level].unique()
        sec_codes = np.array([code[:-3] for code in sec_codes])
        sw_industry_codes = np.intersect1d(sec_codes,sw_industry_codes)
        instrument_df[code_level] = instrument_df[code_level].str[:-3]
        instrument_df = instrument_df[instrument_df[code_level].isin(sw_industry_codes)]
        # 生成股票数据自身排序号
        instrument_df[g_col] = instrument_df["code"].rank(method='dense',ascending=False).astype("int")  
        # 生成行业数据最终结果      
        sw_indus_df = sw_indus_df[sw_indus_df['code'].isin(sw_industry_codes.astype(str))]
        sw_industry_index = sw_industry_index[np.isin(sw_industry_index[:,0],sw_industry_codes)]
        # 重新调整股票索引映射，保留之前筛选后的部分
        keep_index = keep_index[np.isin(keep_index[:,0],instrument_df["code"].unique().astype(np.int32))]

        # 生成行业分类和股票之间的映射关系，具体数据结构规范参考类注释所述
        sw_ins_mappings = []
        # 给股票进行编号，用于后续对照
        instrument_df[g_col] = instrument_df["code"].rank(method='dense',ascending=False).astype("int") 
        instrument_df[g_col] = instrument_df[g_col] - 1 
        # 排序保证后续对应关系不变if
        instrument_df = instrument_df.sort_values(by=[code_level, 'code'], ascending=[True, True])
        for sw_code,group in instrument_df.groupby(code_level):
            sw_code = int(sw_code)
            # 关联之前的数据，按照规范逐个生成
            industry_index = sw_industry_index[sw_industry_index[:,0]==sw_code][0,1]
            instrument_codes = group["code"].values.astype(np.int32)
            instrument_index = np.array([keep_index[keep_index[:,0]==ins_code][0,1] for ins_code in instrument_codes])
            # 股票编码数组长度和序号数组长度需要相等
            raise_if_not(
                instrument_index.shape[0] == instrument_codes.shape[0],
                f"股票编码数组长度和序号数组长度不一致"
            )              
            ava_flag = np.ones(instrument_codes.shape[0]).astype(np.int16)
            sw_ins_mapping = [industry_index,sw_code,group[g_col].values,instrument_index,instrument_codes,ava_flag]
            sw_ins_mappings.append(sw_ins_mapping)
        sw_ins_mappings = np.array(sw_ins_mappings)
               
        return sw_ins_mappings
    
    @staticmethod
    def valid_mapping_by_associate(sw_ins_mappings,ass_sw_ins_mappings,dataset=None):
        """使用训练集的映射数据，对本数据集的映射数据进行验证"""
 
        # 验证行业分类数据，包括类别和排序
        indus_codes = IndustryMappingUtil.get_sw_industry_codes(sw_ins_mappings)
        ass_indus_codes = IndustryMappingUtil.get_sw_industry_codes(ass_sw_ins_mappings)
        # 和参考数据集取交集，如果有多的就舍弃，如果不足则抛出错误
        remain_index = np.nonzero(np.in1d(indus_codes, ass_indus_codes))
        raise_if_not(
            remain_index[0].shape[0] == ass_indus_codes.shape[0],
            f"映射数据验证：行业分类数据不一致"
        )  
        # 删除多余数据
        sw_ins_mappings = sw_ins_mappings[remain_index]  
        # 遍历行业分类，对分类内的股票进行检验
        for i in range(sw_ins_mappings.shape[0]):
            sw_ins_mapping = sw_ins_mappings[i]
            ass_sw_ins_mapping = ass_sw_ins_mappings[i]
            ins_codes = sw_ins_mapping[4]
            ass_ins_codes = ass_sw_ins_mapping[4]    
            # 和参考数据的行业内股票代码取交集，如果有多的就舍弃，如果不足则抛出错误
            remain_ins_index = np.nonzero(np.in1d(ins_codes, ass_ins_codes))     
            raise_if_not(
                remain_ins_index[0].shape[0] == ass_ins_codes.shape[0],
                "映射数据验证：股票数据不一致,行业编码:{}".format(sw_ins_mapping[1])
            )    
            # 删除多余数据(注意关联数组都需要同步)，并更新序列号(instrument_rank)
            for j in range(2,5):
                sw_ins_mapping[j] = np.array(sw_ins_mapping[j])[remain_ins_index]         
            sw_ins_mapping[2] = ass_sw_ins_mapping[2]                        
                    
        return sw_ins_mappings  
    
    
    @staticmethod
    def get_sw_industry_index(sw_ins_mappings):
        return sw_ins_mappings[:,0].astype(np.int32)

    @staticmethod
    def get_sw_industry_instrument(sw_ins_mapping):
        return sw_ins_mapping[2].astype(np.int32)

    @staticmethod
    def get_instrument_with_industry(sw_ins_mapping,indus_rank,ins_rank):
        return sw_ins_mapping[indus_rank,2][ins_rank].astype(np.int32)

    @staticmethod
    def get_instruments_in_industry(sw_ins_mapping,indus_rank):
        return sw_ins_mapping[indus_rank,2].astype(np.int32)
               
    @staticmethod
    def get_sw_industry_codes(sw_ins_mappings):
        return sw_ins_mappings[:,1].astype(np.int32)

       
    @staticmethod
    def get_keep_index(sw_ins_mappings):
        """取得股票对应的series排序号"""
        
        keep_index_combine = [[],[]]
        for si in sw_ins_mappings[:,[2,3]]:
            for i in range(2):
                keep_index_combine[i].append(si[i])
        for i in range(2):
            keep_index_combine[i] = np.concatenate(keep_index_combine[i])
        keep_index_combine = np.stack(keep_index_combine).transpose(1,0)        
        # 需要按照实际排序号进行重拍
        keep_index = keep_index_combine[np.argsort(keep_index_combine[:,0])][:,1]
        return keep_index

    @staticmethod
    def get_instrument_df(sw_ins_mappings,dataset=None):
        """根据映射关系，生成股票代码数据"""
        
        g_col = dataset.get_group_rank_column()
        codes_combine = [[],[]]
        for si in sw_ins_mappings[:,[2,4]]:
            for i in range(2):
                codes_combine[i].append(si[i])
        for i in range(2):
            codes_combine[i] = np.concatenate(codes_combine[i])
        codes_combine = np.stack(codes_combine).transpose(1,0)
        instrument_df = pd.DataFrame(codes_combine,columns=[g_col,"code"])
        return instrument_df
       
    @staticmethod
    def update_rank_to_data(ori_data,sw_ins_mappings,dataset=None):
        """关联更新序列号到传入的数据"""
        
        indus_data = sw_ins_mappings[:,:2]    
        flag = np.all(ori_data[:,1]==indus_data[:,1])
        if flag:
            ori_data[:,0] = indus_data[:,0]
        else:
            ori_data = None
        return ori_data 
    
    @staticmethod
    def get_industry_info(sw_ins_mappings,dataset=None):
        """关联查询行业分类信息"""
        
        codes = sw_ins_mappings[:,1].astype(str)
        result_df = IndustryMappingUtil.get_industry_info_with_code(codes)
        
        return result_df    

    @staticmethod
    def get_industry_info_with_code(codes,dataset=None):
        """关联查询行业分类信息"""
        
        codes = ",".join(codes)
        dbaccessor = DbAccessor({})
        sql = "select code,name,level,cons_num,yield from sw_industry where left(code, 6) in ({})".format(codes)
        result_rows = dbaccessor.do_query(sql)   
        results = []
        for row in result_rows:
            results.append([row[0],row[1],row[2],row[3],row[4]])
        result_df = pd.DataFrame(results,columns=['code','name','level','cons_num','yield'])
        
        return result_df  
       
    @staticmethod
    def assc_series_and_codes(codes,target_series,dataset=None):
        """关联业务编号和序列排序号
            Params:
               codes： 业务编号数据（1维）
               target_series： 目标序列列表
        """
        
        g_col = dataset.get_group_rank_column()
        combine_codes = np.zeros([codes.shape[0],2])
        combine_codes[:,0] = codes
        for index,ts in enumerate(target_series):
            rank_code = int(ts.static_covariates[g_col].values[0])
            code = int(dataset.get_group_code_by_rank(rank_code))
            match_index = np.where(combine_codes[:,0]==code)[0]
            if match_index.shape[0]>0:
                combine_codes[match_index,1] = index
        
        return combine_codes.astype(np.int64)  
                    