import numpy as np
import pandas as pd
import datetime
from enum import Enum
import os

from rqalpha.const import POSITION_EFFECT, SIDE
from rqalpha.const import ORDER_STATUS
from .trade_entity import TradeEntity

from cus_utils.log_util import AppLogger
logger = AppLogger()

# 交易信息表字段，分别为交易日期，品种代码，买卖类型，多空类型，成交价格，成交量,总价格，成交状态，订单编号,平仓原因,附加订单编号
TRADE_COLUMNS = ["trade_date","order_book_id","side","long_short","price","quantity","total_price","status","order_id","close_reason","secondary_order_id"]
TRADE_LOG_COLUMNS = TRADE_COLUMNS + ["create_time"]

class FuturesTradeEntity(TradeEntity):
    """期货交易对象处理类"""
    
    def __init__(
        self,
        save_path=None,
        log_save_path=None,
        **kwargs,
    ):
        super().__init__(save_path,log_save_path,**kwargs)
        

    def add_trade(self,trade,multiplier=1,default_status=ORDER_STATUS.FILLED):
        """添加交易信息，需要先具备订单信息"""
        
        order = self.get_order_by_id(trade.order_id)
        if order is None or order.shape[0]==0:
            logger.warning("order id not exists:{}".format(trade.order_id))
            return
        
        trade_date = trade.datetime
        order_book_id = trade.order_book_id
        side = trade.side
        price = trade.last_price
        quantity = trade.last_quantity
        position_effect = trade.position_effect
        # 自己计算总成交额
        total_price = price*quantity*multiplier + trade.tax + trade.transaction_cost
        # 交易状态为已成交
        status = default_status
        order_id = trade.order_id
        # 存储对于实际仿真或实盘系统的交易订单号
        secondary_order_id = order.secondary_order_id
        if secondary_order_id is None:
            secondary_order_id = 0       
        if "close_reason" in order:
            close_reason = order['close_reason'] 
        else:
            close_reason = None
        row_data = [trade_date,order_book_id,side,position_effect,price,quantity,multiplier,total_price,status,order_id,close_reason,secondary_order_id]
        # 使用订单号查询并更新记录
        self.trade_data_df[self.trade_data_df["order_id"]==order_id] = row_data
        # 变更后保存数据
        if self.save_path is not None:
            self.exp_trade_data(self.save_path)   
            # 日志记录
            self.add_log(row_data)
            
    def get_trade_date_by_instrument(self,order_book_id,position_effect,before_date):  
        """查询某个品种的最近已成交交易日期
            Params:
                order_book_id p品种编码
                position_effect 开平类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)
                                  &(trade_data_df["position_effect"]==position_effect)&
                                  (trade_data_df["trade_date"]<=pd.to_datetime(before_date))]
        if target_df.shape[0]==0:
            return None
        # 取得最后一个交易
        return target_df["trade_date"].dt.to_pydatetime().tolist()[-1].strftime('%Y%m%d') 


    def get_open_list(self,trade_date):   
        """取得所有已开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]                  
        return target_df  
    
    def get_open_list_filled(self,trade_date):   
        """取得所有已成交订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.FILLED)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]                  
        return target_df  
    
    def get_open_list_active(self,trade_date):   
        """取得所有未成交开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]      
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]                  
        return target_df  
    
    def get_close_list_active(self,trade_date):   
        """取得所有未成交平仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]      
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]                  
        return target_df  
    
    def get_open_list_reject(self,trade_date):   
        """取得所有被拒绝的开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)]             
        return target_df          
                