from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch import nn

from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from .layers.Autoformer_EncDec import series_decomp
from .layers.Embed import DataEmbedding_wo_pos

from .fur_ts_inner import FurTimeMixer

class FurIndustryMixer(nn.Module):
    """混合TimeMixer以及STID相关设计思路的序列模型,使用MLP作为底层网络.
       把行业板块信息融合到模型中，不同行业，权重不共享
    """
    
    def __init__(self, seq_len=25, round_skip_len=25,pred_len=5,past_cov_dim=12, dropout=0.3,industry_index=None,hidden_size=16,down_sampling_window=5,
                 combine_nodes_num=None,instrument_index=None,index_num=1,device="cpu"):
        """行业总体网络，分为子网络，以及整合网络2部分"""
        
        super().__init__()
        
        self.combine_nodes_num = combine_nodes_num
        self.combine_instrument_index = instrument_index
        self.industry_index = industry_index
        self.num_nodes = combine_nodes_num.sum()
        self.index_num = index_num
        
        # 循环取得不同行业板块的多个下级模型
        sub_model_list = []
        for num_nodes in combine_nodes_num:
            sub_model = FurTimeMixer(
                num_nodes=num_nodes,
                seq_len=seq_len,
                pred_len=pred_len,
                round_skip_len=round_skip_len,
                past_cov_dim=past_cov_dim,
                dropout=dropout,
                down_sampling_window=down_sampling_window,
                device=device,
                )
            sub_model_list.append(sub_model)
        self.sub_models = nn.ModuleList(sub_model_list)
        # self.sub_models_after = nn.ModuleList(sub_model_after_list)
        # 整合输出网络
        if index_num>1:
            self.combine_layer = nn.Sequential(
                    nn.Linear(self.combine_nodes_num.shape[0], hidden_size),
                    nn.ReLU(), 
                    nn.Linear(hidden_size,index_num),
                    nn.LayerNorm(index_num)
                ).to(device)
        else:
            self.combine_layer = nn.Sequential(
                    nn.Linear(self.combine_nodes_num.shape[0], hidden_size),
                    nn.ReLU(), 
                    nn.Linear(hidden_size,index_num),
                ).to(device)    
        
    def forward(self, x_in): 
        """多个行业板块子模型顺序输出，整合输出形成统一输出"""
        
        cls_out_combine = []
        index_data_combine = []
        # 不同行业分别输出
        for i in range(self.combine_nodes_num.shape[0]):
            m = self.sub_models[i]
            # m_after = self.sub_models_after[i]
            instrument_index = self.combine_instrument_index[i]
            x_enc, historic_future_covariates,future_covariates,past_round_targets,past_index_round_targets = x_in
            x_inner = (x_enc[:,instrument_index,...],historic_future_covariates[:,instrument_index,...],
                        future_covariates[:,instrument_index,...],past_round_targets[:,instrument_index,...],past_index_round_targets[:,i,...])
            _,cls_out,sw_index_data = m(x_inner)
            # 叠加归一化输出
            # cls_out = m_after(cls_out.squeeze(-1)).unsqueeze(-1)
            cls_out_combine.append(cls_out)
            index_data_combine.append(sw_index_data)
        
        index_data_combine = self.combine_layer(torch.cat(index_data_combine,dim=1))     
        return None,cls_out_combine,index_data_combine



class FurIndustryDRollMixer(nn.Module):
    """混合TimeMixer以及STID相关设计思路的序列模型,使用MLP作为底层网络.
       使用二次滚动计算的模式，整合批次和二次序列数组
    """
    
    def __init__(self, seq_len=25,round_skip_len=25, pred_len=5,past_cov_dim=12, dropout=0.3,industry_index=None,hidden_size=16,down_sampling_window=5,
                 main_index=-1,rolling_size=18,num_nodes=6,index_num=1,device="cpu"):
        """行业总体网络，分为子网络，以及整合网络2部分"""
        
        super().__init__()
        
        self.industry_index = industry_index
        self.main_index = main_index
        self.num_nodes = num_nodes
        self.index_num = index_num
        self.rolling_size = rolling_size
        
        # 循环取得不同时间段的多个下级模型
        sub_model_list = []
        for _ in range(rolling_size):
            sub_model = FurTimeMixer(
                num_nodes=num_nodes,
                seq_len=seq_len,
                pred_len=pred_len,
                past_cov_dim=past_cov_dim,
                dropout=dropout,
                round_skip_len=round_skip_len,
                down_sampling_window=down_sampling_window,
                device=device,
                )
            sub_model_list.append(sub_model)
        self.sub_models = nn.ModuleList(sub_model_list)
        # Last Rolling Data
        self.indus_lst_layers = nn.Sequential(
                nn.Linear(rolling_size*num_nodes, hidden_size),
                nn.ReLU(), 
                nn.Linear(hidden_size,num_nodes),
                nn.LayerNorm(num_nodes)
            ).to(device)
                      
        # 整合输出网络
        self.combine_layer = nn.Sequential(
                nn.Linear(rolling_size, hidden_size),
                nn.ReLU(), 
                nn.Linear(hidden_size,index_num),
                nn.LayerNorm(index_num)
            ).to(device)
        indus_combine_layers = []
        for _ in range(num_nodes):
            indus_combine_layer = nn.Sequential(
                    nn.Linear(rolling_size, hidden_size),
                    nn.ReLU(), 
                    nn.Linear(hidden_size,rolling_size),
                    nn.LayerNorm(rolling_size)
                ).to(device)    
            indus_combine_layers.append(indus_combine_layer)      
        self.indus_combine_layers = nn.ModuleList(indus_combine_layers)      
        
    def forward(self, x_in): 
        """多个子模型顺序输出，整合输出形成统一输出"""
        
        cls_out_combine = []
        index_data_combine = []
        rolling_size = self.rolling_size
        x_enc, historic_future_covariates,future_covariates,past_round_targets,past_index_round_targets = x_in
        batch_size = int(x_enc.shape[0]/rolling_size)
        
        x_enc_rs = x_enc.reshape(batch_size,rolling_size,*x_enc.shape[1:])
        future_covariates_rs = future_covariates.reshape(batch_size,rolling_size,*future_covariates.shape[1:])
        historic_future_covariates_rs = historic_future_covariates.reshape(batch_size,rolling_size,*historic_future_covariates.shape[1:])
        past_round_targets_rs = past_round_targets.reshape(batch_size,rolling_size,*past_round_targets.shape[1:])
        past_index_round_targets_rs = past_index_round_targets.reshape(batch_size,rolling_size,*past_index_round_targets.shape[1:])
        # 不同日期滚动范围，使用子模型分别输出
        for i in range(self.rolling_size):
            m = self.sub_models[i]
            x_enc_inner = x_enc_rs[:,i,self.industry_index,...]
            historic_future_covariates_inner = historic_future_covariates_rs[:,i,self.industry_index,...]
            future_covariates_inner = future_covariates_rs[:,i,self.industry_index,...]
            past_round_targets_inner = past_round_targets_rs[:,i,self.industry_index,...]
            past_index_round_targets_inner = past_index_round_targets_rs[:,i,self.main_index,...]
            x_inner = (x_enc_inner,historic_future_covariates_inner,future_covariates_inner,past_round_targets_inner,past_index_round_targets_inner)
            _,cls_out,sw_index_data = m(x_inner)
            # 叠加归一化输出
            # cls_out = m_after(cls_out.squeeze(-1)).unsqueeze(-1)
            cls_out_combine.append(cls_out)
            index_data_combine.append(sw_index_data)
        cls_out_combine = torch.stack(cls_out_combine).permute(1,0,2,3)
        
        indus_out_combine = []
        # 不同行业分别进行多时间段整合连接
        for i in range(self.num_nodes):
            m = self.indus_combine_layers[i]
            indus_out_combine.append(m(cls_out_combine[:,:,i,0]))
        indus_out_combine = torch.stack(indus_out_combine).permute(1,2,0).unsqueeze(-1)
        # 拼接整合整体指数预测数据
        index_data_combine = self.combine_layer(torch.cat(index_data_combine,dim=1))   
        lst_data = self.indus_lst_layers(indus_out_combine.reshape([indus_out_combine.shape[0],-1]))  
        return None,indus_out_combine,lst_data

#####################   新增策略模型  #########################    
      
class FurStrategy(nn.Module):
    """基于前置模型的结果，实现相关策略的模型"""
    
    def __init__(self, target_num=2,hidden_size=8,select_num=10,past_tar_dim=3,trend_threhold=0.55):    
        """ 使用MLP基础网络实现策略选择
            Params:
                features_dim: 特征维度(即所有品种的数量)
                target_num: 输入中的目标值个数
                hidden_dim: 隐藏维度
                select_num: 一次筛选的数量
                past_tar_dim: 过去目标值时间段数量
                trend_threhold: 整体趋势阈值
        """
    
        super().__init__()
        
        self.select_num = select_num
        self.trend_threhold = trend_threhold
        self.past_tar_dim = past_tar_dim
        # 条件选取网络，输出形状为1，结合第一维度，形成一维的特征值，后续进行排序取值.
        # 在此根据趋势分别设置多头和空头2个网络
        self.l_net = nn.Sequential(nn.Linear(past_tar_dim*2+1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.s_net = nn.Sequential(nn.Linear(past_tar_dim*2+1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        
    def forward(self, x1,x2,past_targets,ignore_next=False):
        """策略：1.判断趋势 2.根据排序进行一次筛选 3.使用网络实现二次筛选
            Params:
                x1 1号目标输出值
                x2 2号目标输出值
                past_targets 过去目标值(包含价格目标值，包含最近3个时间段),shape:[batch_size,时间段,2]
        """
        
        # 使用RSV指标作为整体趋势判断指标
        output_mean_value = torch.mean(x1,dim=1)
        # 取得排名靠前的索引
        selected_index_s = x1.argsort(dim=1)[:,:self.select_num]
        selected_index_l = x2.argsort(dim=1,descending=True)[:,:self.select_num]
        # 取得排序后的索引
        trend_value = (output_mean_value>self.trend_threhold)
        combine_index = torch.zeros([x1.shape[0],self.select_num]).long().to(x1.device)
        l_index = torch.where(trend_value)[0].to(x1.device)
        combine_index[l_index] = selected_index_l[l_index]
        s_index = torch.where(~trend_value)[0].to(x1.device)
        combine_index[s_index] = selected_index_s[s_index]   
        if ignore_next:
            return (torch.ones([x1.shape[0],self.select_num]).to(x1.device),trend_value,combine_index) 
        # 取得排名靠前的输出值(RSV取得反向用于空房判断，CCI使用正向用于多方判断  
        x1_input = torch.gather(x1, 1, selected_index_s)       
        x2_input = torch.gather(x2, 1, selected_index_l)      
        # 整合RSV指标和过去参考数值，用于空方网络判断
        past_targets_s = [torch.gather(past_targets[...,i], 1, selected_index_s)  for i in range(past_targets.shape[-1])]
        past_targets_s = torch.stack(past_targets_s,dim=-1)
        x1_input = torch.cat([x1_input.unsqueeze(-1),past_targets_s],dim=-1)
        s_output = self.s_net(x1_input)
        # 整合CCI指标和过去参考数值，用于多方网络判断
        past_targets_l = [torch.gather(past_targets[...,i], 1, selected_index_l)  for i in range(past_targets.shape[-1])]
        past_targets_l = torch.stack(past_targets_l,dim=-1)
        x2_input = torch.cat([x2_input.unsqueeze(-1),past_targets_l],dim=-1)
        l_output = self.l_net(x2_input)
        # 根据总体趋势，合并输出
        output = torch.zeros([x1.shape[0],self.select_num]).to(x1.device)

        if l_index.shape[0]>0:
            output[l_index] = l_output[l_index,:,0]
        if s_index.shape[0]>0:
            output[s_index] = s_output[s_index,:,0]
        
        return (output,trend_value,combine_index)            
