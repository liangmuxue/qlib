import numpy as np
import pandas as pd
import datetime
from enum import Enum
import os

from rqalpha.const import SIDE
from rqalpha.const import ORDER_STATUS
from .trade_entity import TradeEntity

from cus_utils.log_util import AppLogger
logger = AppLogger()

# 交易信息表字段，分别为交易日期，品种代码，买卖类型，多空类型，成交价格，成交量,总价格，成交状态，订单编号,卖出原因,附加订单编号
TRADE_COLUMNS = ["trade_date","order_book_id","side","long_short","price","quantity","total_price","status","order_id","sell_reason","secondary_order_id"]
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
        

    def add_trade(self,trade,default_status=ORDER_STATUS.FILLED):
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
        # 自己计算总成交额
        total_price = price*quantity + trade.tax + trade.transaction_cost
        # 交易状态为已成交
        status = default_status
        order_id = trade.order_id
        # 存储对于实际仿真或实盘系统的交易订单号
        secondary_order_id = order.secondary_order_id
        if secondary_order_id is None:
            secondary_order_id = 0        
        row_data = [trade_date,order_book_id,side,price,quantity,total_price,status,order_id,order.sell_reason,secondary_order_id]
        # 使用订单号查询并更新记录
        self.trade_data_df[self.trade_data_df["order_id"]==order_id] = row_data
        # 变更后保存数据
        if self.save_path is not None:
            self.exp_trade_data(self.save_path)   
            # 日志记录
            self.add_log(row_data)