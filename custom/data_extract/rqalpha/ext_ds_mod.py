import pandas as pd
import numpy as np
import pickle
import os

from rqalpha.interface import AbstractMod
from rqalpha.const import ORDER_STATUS
from rqalpha.const import SIDE

from .tdx_ds import TdxDataSource
from trader.rqalpha.trade_entity import TRADE_COLUMNS
from trader.utils.date_util import tradedays
from trader.data_viewer import DataViewer
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from cus_utils.log_util import AppLogger
logger = AppLogger()

# 交易扩展信息用于统计，字段包括：买入价，差价、盈亏金额, 购买日期，买卖间隔天数
TRADE_EXT_COLUMNS = ["buy_price","differ_range","gain","buy_date","duration"]

class ExtDataMod(AbstractMod):
    """用于加载自定义数据源"""
    
    def __init__(self):
        pass

    def start_up(self, env, mod_config):
        self.report_save_path = mod_config.report_save_path
        if not os.path.exists(self.report_save_path):
            os.makedirs(self.report_save_path)  
        self.env = env
        env.set_data_source(TdxDataSource(env.config.base.data_bundle_path))

    def tear_down(self, code, exception=None):
        """统计分析入口"""
        
        strategy_obj = self.env.user_strategy.user_context.strategy_class
        ml_context = self.env.user_strategy.user_context.ml_context
        # self.export_trade_data(strategy_obj)
        self.analysis_stat(strategy_obj,ml_context=ml_context)
        return
    
    def export_trade_data(self,strategy_obj):
        save_path = self.report_save_path + "/trade.csv"
        strategy_obj.trade_entity.exp_trade_data(save_path)
    
    def analysis_stat(self,strategy_obj,ml_context=None):
        """实现统计分析功能"""
        
        data_viewer = DataViewer(env_name="stat_trades_5m")
        load_trade_df = self.env.config.mod.ext_ds_mod.load_trade_df
        
        stat_df = self.build_stat_df(strategy_obj,load_trade_df=load_trade_df,load_cache=False)
        # 按照盈亏排序
        stat_df = stat_df.sort_values(by=["differ_range"],ascending=False)
        pred_df = ml_context.pred_df
        pred_df["instrument"] = pred_df["instrument"].astype(int)
        pred_recorder = ml_context.pred_recorder
        dataset = ml_context.dataset
        ext_length = 25
        
        total_gain = stat_df["gain"].sum()
        logger.info("total_gain:{}".format(total_gain))
        ml_context.record_results(stat_df)
        
        save_path = self.report_save_path + "/plot"
        # 取得预测数据和回测数据，并进行图形化展示
        # for index,row in stat_df.iterrows():
        #     trade_date = row["trade_date"]
        #     trade_date_str = trade_date.strftime('%Y%m%d')
        #     instrument = int(transfer_instrument(row["order_book_id"]))
        #     pred_df_item = pred_df[(pred_df["pred_date"]==int(trade_date_str))&(pred_df["instrument"]==instrument)]
        #     complex_df = pred_recorder.combine_complex_df_data(trade_date_str,instrument,pred_df=pred_df_item,df_ref=dataset.df_all,ext_length=ext_length)
        #     data_viewer.show_trades_data_visdom(row,complex_df)
        #     buy_trade_date_str = row["buy_date"]
        #     buy_pred_df_item = pred_df[(pred_df["pred_date"]==int(buy_trade_date_str))&(pred_df["instrument"]==instrument)]
        #     buy_complex_df = pred_recorder.combine_complex_df_data(buy_trade_date_str,instrument,pred_df=buy_pred_df_item,df_ref=dataset.df_all,ext_length=ext_length)
        #     data_viewer.show_single_complex_pred_data(buy_complex_df,correct=-1,save_path=save_path)
        #     data_viewer.show_single_complex_pred_data_visdom(buy_complex_df)
            # logger.debug("complex_df data:{}".format(complex_df))
         
    def build_stat_df(self,strategy_obj,load_trade_df=False,load_cache=False):
        """生成统计数据"""
        
        data_file = self.report_save_path + "/stat.pkl"
        if load_cache:
            with open(data_file, "rb") as fin:
                stat_df = pickle.load(fin)  
        else:   
            trade_data_file = self.report_save_path + "/trade.pkl"    
            if load_trade_df:
                with open(trade_data_file, "rb") as fin:
                    trade_data_df = pickle.load(fin)  
            else:     
                trade_data_df = strategy_obj.trade_entity.trade_data_df
                with open(trade_data_file, "wb") as fout:
                    pickle.dump(trade_data_df, fout)                  
            trade_data_df = trade_data_df.sort_values(by=["trade_date","order_book_id"])
            # 只统计已完成订单
            target_df = trade_data_df[trade_data_df["status"]==ORDER_STATUS.FILLED]
            # 以股票为维度聚合，进行分析
            group_df = target_df.groupby("order_book_id")
            new_columns = TRADE_COLUMNS + TRADE_EXT_COLUMNS
            stat_data = []
            for name,instrument_df in group_df:
                instrument_df = instrument_df.sort_values(by=["trade_date"]).reset_index(drop=True)
                for index,row in instrument_df.iterrows():
                    # 买卖分别配对，进行额度计算
                    if index%2==0 and row["side"]!=SIDE.BUY:
                        # 需要符合先买后卖原则
                        logger.warning("buy index not fit:{},{}".format(index,row))
                        break
                    if index%2==1 and row["side"]!=SIDE.SELL:
                        # 需要符合先买后卖原则
                        logger.warning("sell index not fit:{},{}".format(index,row))    
                        break    
                    if row["side"]==SIDE.SELL:
                        # 取得买入时记录，并计算差价
                        prev_buy_row = instrument_df.iloc[index-1]
                        buy_price = prev_buy_row["price"]
                        differ_range = (row["price"] - buy_price)/buy_price
                        gain = (row["price"] - buy_price) * row["quantity"]
                        buy_day = prev_buy_row["trade_date"].strftime('%Y%m%d')
                        sell_day = row["trade_date"].strftime('%Y%m%d')
                        duration = tradedays(buy_day,sell_day)
                        new_row = row.values.tolist() + [buy_price,differ_range,gain,buy_day,duration]
                        stat_data.append(new_row)
            stat_df = pd.DataFrame(np.array(stat_data),columns = new_columns)
            with open(data_file, "wb") as fout:
                pickle.dump(stat_df, fout)          
        return stat_df
                    
def load_mod():
    return ExtDataMod()   
