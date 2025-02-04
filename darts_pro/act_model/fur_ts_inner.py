from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from darts_pro.data_extension.industry_mapping_util import FuturesMappingUtil
from .layers.Autoformer_EncDec import series_decomp
from .layers.Embed import DataEmbedding_wo_pos

class FurTimeMixer(nn.Module):
    """混合TimeMixer以及STID相关设计思路的序列模型,使用MLP作为底层网络"""
    
    def __init__(self, c_in=10,c_out=1,seq_len=25, pred_len=5,past_cov_dim=12, dropout=0.3,decomp_method='moving_avg',d_ff=2048,moving_avg=25,
                 e_layers:int=3, d_model=16,down_sampling_method='avg',down_sampling_window=5,down_sampling_layers=1,hidden_size=8,
                 num_nodes=0,node_dim=16,day_of_week_size=5,temp_dim_diw=8,month_of_year_size=12,temp_dim_miy=8,day_of_month_size=31,temp_dim_dim=8,
                 device="cpu"):
        """Params:
                c_in:输入目标维度
                c_out: 输出目标维度，默认1
                seq_len： 输入序列长度
                pred_len: 预测序列长度
                past_cov_dim: 输入协变量维度
                e_layers: 多尺度计算的神经网络层数
                d_model： 嵌入编码维度
                day_of_week_size: 每周工作日数量
                month_of_year_size: 每年月数量
                temp_dim_diw： 周日期参数对应的嵌入维度
                temp_dim_miy: 月份参数对应的嵌入维度
                temp_dim_dim： 月日期参数对应的嵌入维度
                
        """
        
        super().__init__()
        
        
        self.pred_len = pred_len
        self.temp_dim_diw = temp_dim_diw
        self.temp_dim_miy = temp_dim_miy
        self.temp_dim_dim = temp_dim_dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        ti_sp_dim = temp_dim_diw + temp_dim_miy + temp_dim_dim + node_dim
                
        ###### TimeMixer部分 #####
        self.c_out = c_out
        self.layers = e_layers
        self.down_sampling_method = down_sampling_method
        self.down_sampling_window = down_sampling_window  
        self.down_sampling_layers = down_sampling_layers      
        # 为每个尺度进行趋势项和周期项的计算
        configs = (seq_len,pred_len,d_model,dropout,down_sampling_window,down_sampling_layers,decomp_method,d_ff,moving_avg)
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(e_layers)]) 
        # 向量编码，只考虑数值向量，输入维度为过去协变量维度     
        self.enc_embedding = DataEmbedding_wo_pos(past_cov_dim, d_model, dropout=dropout)    
        # 投影层，单通道模式
        self.projection_layer = nn.Linear(d_model, 1, bias=True)   
        # 整体投影层，映射为1段
        self.comp_proj_layer = nn.Linear(pred_len, 1, bias=True)   
        # 添加层归一
        predict_layers = []
        for i in range(down_sampling_layers + 1):
            layer = nn.Sequential(
                torch.nn.Linear(seq_len // (down_sampling_window ** i),pred_len),
                nn.LayerNorm(pred_len)
            ).to(device)
            predict_layers.append(layer)
        self.predict_layers = nn.ModuleList(predict_layers)
        # 使用残差计算量化数值（整体走势）预测
        self.last_tar_skip_layer = nn.Linear(seq_len, 1, bias=True)  
        
        ###### STID部分 #####

        # 全局参数矩阵,每周的工作日部分,注意嵌入维度需要与TimeMixer部分的特征维度（d_model）保持一致
        self.day_in_week_emb = nn.Parameter(
            torch.empty(day_of_week_size, temp_dim_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        # 每年的月份部分
        self.month_in_year_emb = nn.Parameter(
            torch.empty(month_of_year_size, temp_dim_miy))
        nn.init.xavier_uniform_(self.month_in_year_emb)    
        # 每月的日期部分
        self.day_in_month_emb = nn.Parameter(
            torch.empty(day_of_month_size, temp_dim_dim))
        nn.init.xavier_uniform_(self.day_in_month_emb)          
        # 空间参数，对应不同的品种或行业分类
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)     
        # 空间及时间编码层   
        self.ti_sp_enc = nn.Linear(ti_sp_dim,d_model)
        # 未来时空投影层，整合投影到单通道
        self.tisp_projection_layer = nn.Linear(ti_sp_dim*pred_len, 1, bias=True)      
        
        # 品种数据投影到板块指数数据，按照不同分类板块分别投影
        self.index_projection_layer = nn.Linear(num_nodes, 1, bias=True)        
        self.all_to_index_projection_layer = nn.Linear((self.num_nodes-1)*pred_len, pred_len, bias=True)          
        # 整合指数过去数据的残差,注意使用的不是过去数值长度，而是再次拆分的长度,以避免未来数值泄露
        self.index_skip_layer = nn.Linear(seq_len-pred_len, 1, bias=True)   
        self.round_skip_layer = nn.Linear(seq_len-pred_len, 1, bias=True)   
                           
    def forward(self, x_in): 
        
        # 分别对应输入协变量，过去时间变量,未来时间变量。过去整体量化数值,过去指数数值
        # 其中输入协变量形状：[批次数（B），节点数（C或N），序列长度(T),协变量维度(D)]
        x_enc, historic_future_covariates,future_covariates,past_round_targets,past_index_round_targets = x_in
        # 采样得到对应时间变量
        x_time_mark_past = historic_future_covariates[:,0,:,:]
        x_time_mark_future = future_covariates[:,0,:,:]
        # 输入序列中对应的2类过去时间协变量，形状为: [Batch_size,Seq_len]
        xpast_day_of_week,xpast_month_of_year,xpast_day_of_month = x_time_mark_past[...,0],x_time_mark_past[...,1],x_time_mark_past[...,2]
        
        # 引入STID的全局查询逻辑，分别对时间和空间进行全局嵌入映射
        batch_size,node_num,seq_len = x_enc.shape[0],x_enc.shape[1],x_enc.shape[2] 
        # 全局时间变量映射到当前时间协变量,形状为: [B,T,temp_dim_xxx]
        day_in_week_emb = torch.zeros([xpast_day_of_week.shape[0],xpast_day_of_week.shape[1],self.day_in_week_emb.shape[1]]).to(xpast_day_of_week.device)
        for i in range(self.day_in_week_emb.shape[0]):
            index = torch.where(xpast_day_of_week==i)
            if index[0].shape[0]>0:
                day_in_week_emb[index[0],index[1],:] = self.day_in_week_emb[i]
        month_in_year_emb = torch.zeros([xpast_month_of_year.shape[0],xpast_month_of_year.shape[1],self.month_in_year_emb.shape[1]]).to(xpast_day_of_week.device)
        for i in range(self.month_in_year_emb.shape[0]):
            index = torch.where(xpast_month_of_year==i)
            if index[0].shape[0]>0:
                month_in_year_emb[index[0],index[1],:] = self.month_in_year_emb[i]
        day_in_month_emb = torch.zeros([xpast_day_of_month.shape[0],xpast_day_of_month.shape[1],self.day_in_month_emb.shape[1]]).to(xpast_day_of_week.device)
        for i in range(self.day_in_month_emb.shape[0]):
            index = torch.where(xpast_day_of_month==i)
            if index[0].shape[0]>0:
                day_in_month_emb[index[0],index[1],:] = self.day_in_month_emb[i]                
        # 把时间变量数据扩充维度到：[B,N,T,temp_dim_xxx]
        day_in_week_emb = day_in_week_emb.unsqueeze(1).repeat(1,node_num,1,1)       
        month_in_year_emb = month_in_year_emb.unsqueeze(1).repeat(1,node_num,1,1)       
        day_in_month_emb = day_in_month_emb.unsqueeze(1).repeat(1,node_num,1,1)   
        # 全局空间关系变量映射，扩展后的形状: [B,N,T,node_dim]
        node_emb = self.node_emb.unsqueeze(1).unsqueeze(0).repeat(batch_size,1,seq_len,1)  
        # 合并时间和空间协变量，形状：[B,N,T,合并后的特征维度]
        x_mark_enc = torch.cat([day_in_week_emb,month_in_year_emb,day_in_month_emb,node_emb],dim=-1)
        
        # 通过池化下采样，得到不同尺度的序列.使用5->25->125,对应业务周期为：日，月，半年
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N, D = x.size()
                # 合并批次和节点，形成新的形状，Shape为：[B*N，多尺度序列长度,协变量维度]
                x = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
                x_list.append(x)
                # 修改形状为: [B*N,T,时间空间特征维度]
                x_mark = x_mark.reshape(batch_size*node_num,T,-1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N, D = x.size()
                # 合并批次和节点，形成新的形状，Shape为：[B * N,，序列长度，新通道数（1),协变量维度]
                x = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, 1,D)
                x_list.append(x)
                            
        # embedding
        enc_out_list = []
        x_list = (x_list, None)
        if x_mark_enc is not None:
            # 嵌入向量部分，生成新特征向量,形状为[B*N,T,d_model]
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                # x的形状：[B * N, T, past_cov_dim],输出形状: [B * N, T, d_model]
                enc_out = self.enc_embedding(x, None)
                # 对x_mark进行嵌入计算，原形状：[B * N, T, D]，输出形状: [B * N, T, d_model]
                x_mark = self.ti_sp_enc(x_mark)
                # 合并两类数据,输出形状[B * N, T, d_model]
                enc_out = enc_out + x_mark
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0],x_mark_list):
                B, T, N, D = x.size()
                enc_out = self.enc_embedding(x, None)                                  
                enc_out_list.append(enc_out)
                

        # 过去值的序列分解（趋势项和周期项），以及多尺度信息融合
        for i in range(self.layers):
            # 融合后维度保持不变
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        
        # 未来目标预测，所有尺度都计算，然后相加
        dec_out_list,comp_out_list,x_mark_dec = self.future_multi_mixing(enc_out_list, len(x_list[0]),x_time_mark_future,batch_size=batch_size,node_num=node_num)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # 叠加未来时空变量投影
        x_mar_dec_out = self.tisp_projection_layer(x_mark_dec).reshape([batch_size,node_num]).unsqueeze(-1)       
        comp_out = torch.stack(comp_out_list, dim=-1).sum(-1).unsqueeze(-1)
        # 叠加整体数值残差计算
        comp_out = comp_out + x_mar_dec_out + self.round_skip_layer(past_round_targets)
        # 按照不同分类板块分别投影
        industry_decoded_data = self.index_projection_layer(comp_out.squeeze(-1))
        # 使用整体走势过去值,注意这里的数据做了再次拆分，以避免未来数据泄露
        sw_index_data = industry_decoded_data + self.index_skip_layer(past_index_round_targets)
        return dec_out,comp_out,sw_index_data

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """修改原方法，增加一组维度"""
        
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool2d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv2d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1,3))
        x_mark_sampling_list.append(x_mark_enc.permute(0, 2, 1,3))
        
        B,C,T,D = x_enc.size()
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori.permute(0, 1, 3,2).reshape(B,C*D,-1)).reshape(B,C,D,-1)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 3, 1, 2))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, :, ::self.down_sampling_window, :].permute(0, 2, 1,3))
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def future_multi_mixing(self, enc_out_list, x_len,x_time_mark_future,batch_size=0,node_num=0):
        """未来目标融合，合并所有尺度，并整合全局空间和时间变量的未来变量部分
           Params:
               enc_out_list: 编码数据，数组类型，元素的形状为: [B*N,T,d_model]
               x_len： 多尺度的个数
               x_time_mark_future: 输出序列中对应的2类未来时间协变量，形状为: [Batch_size,pred_len]
        """
        
        
        xfuture_day_of_week,xfuture_month_of_year,xfuture_day_of_month = x_time_mark_future[...,0],x_time_mark_future[...,1],x_time_mark_future[...,2]
                        
        # 全局时间变量映射到当前时间协变量,形状为: [B,T,temp_dim_xxx]
        day_in_week_emb = torch.zeros([xfuture_day_of_week.shape[0],xfuture_day_of_week.shape[1],self.day_in_week_emb.shape[1]]).to(xfuture_day_of_week.device)
        for i in range(self.day_in_week_emb.shape[0]):
            index = torch.where(xfuture_day_of_week==i)
            if index[0].shape[0]>0:
                day_in_week_emb[index[0],index[1],:] = self.day_in_week_emb[i]        
        month_in_year_emb = torch.zeros([xfuture_month_of_year.shape[0],xfuture_month_of_year.shape[1],self.month_in_year_emb.shape[1]]).to(xfuture_month_of_year.device)
        for i in range(self.month_in_year_emb.shape[0]):
            index = torch.where(xfuture_month_of_year==i)
            if index[0].shape[0]>0:
                month_in_year_emb[index[0],index[1],:] = self.month_in_year_emb[i]
        day_in_month_emb = torch.zeros([xfuture_day_of_month.shape[0],xfuture_day_of_month.shape[1],self.day_in_month_emb.shape[1]]).to(xfuture_day_of_month.device)
        for i in range(self.day_in_month_emb.shape[0]):
            index = torch.where(xfuture_day_of_month==i)
            if index[0].shape[0]>0:
                day_in_month_emb[index[0],index[1],:] = self.day_in_month_emb[i]                
        # 把时间变量数据扩充维度到：[B,N,T,temp_dim_xxx]
        day_in_week_emb = day_in_week_emb.unsqueeze(1).repeat(1,node_num,1,1)       
        month_in_year_emb = month_in_year_emb.unsqueeze(1).repeat(1,node_num,1,1)      
        day_in_month_emb = day_in_month_emb.unsqueeze(1).repeat(1,node_num,1,1)    
        # 全局空间关系变量映射，扩展后的形状: [B,N,pred_len,node_dim]
        node_emb = self.node_emb.unsqueeze(1).unsqueeze(0).repeat(batch_size,1,self.pred_len,1)  
        # 合并时间和空间协变量，形状：[B,N,pred_len,合并后的特征维度]
        x_mark_dec = torch.cat([day_in_week_emb,month_in_year_emb,day_in_month_emb,node_emb],dim=-1)
        x_mark_dec = x_mark_dec.reshape(batch_size,node_num,-1) # [B*N,pred_len,时空特征维度]
                        
        dec_out_list = []
        comp_out_list = []
        # 解码生成输出值，把不同尺度整合为统一尺度（seq_len）
        for i, enc_out in zip(range(x_len), enc_out_list):
            # 尺度整合部分，enc_out形状: [B * N, T, d_model],dec_out形状: [B * N, pred_len, d_model]
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # 预测未来变量投影
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(batch_size,node_num, self.pred_len)
            dec_out_list.append(dec_out)
            # 整体走势预估对应的投影
            comp_out = self.comp_proj_layer(dec_out)            
            comp_out = comp_out.reshape(batch_size,node_num)
            comp_out_list.append(comp_out)
        return dec_out_list,comp_out_list,x_mark_dec
    
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        
        (seq_len,down_sampling_window,down_sampling_layers) = configs
        
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        
        (seq_len,down_sampling_window,down_sampling_layers) = configs
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

    
class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        (seq_len,pred_len,d_model,dropout,down_sampling_window,down_sampling_layers,decomp_method,d_ff,moving_avg) = configs
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.dropout = nn.Dropout(dropout)
        self.channel_independence = True

        if decomp_method == 'moving_avg':
            self.decompsition = series_decomp(moving_avg)
        else:
            raise ValueError('decompsition is error')

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing((seq_len,down_sampling_window,down_sampling_layers))

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing((seq_len,down_sampling_window,down_sampling_layers))

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list
    
    
#####################   新增策略模型  #########################    
      
class FurStrategy(nn.Module):
    """基于前置模型的结果，实现相关策略的模型"""
    
    def __init__(self, features_dim,target_num=2,hidden_size=8,select_num=10,past_tar_dim=3,trend_threhold=0.55):    
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
        