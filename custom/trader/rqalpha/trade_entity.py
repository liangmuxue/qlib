import numpy as np
import pandas as pd
import datetime
from enum import Enum

from rqalpha.const import SIDE
from rqalpha.const import ORDER_STATUS


# 交易信息表字段，分别为交易日期，股票代码，成交价格，成交量,总价格，成交状态，订单编号
TRADE_COLUMNS = ["trade_date","instrument","side","price","quantity","total_price","status","order_id"]

class TradeEntity():
    """交易对象处理类"""
    
    def __init__(
        self,
        **kwargs,
    ):
        self.trade_data = None
        
        
    def add_trade(self,trade,default_status=ORDER_STATUS.FILLED):
        """添加交易信息"""
        
        trade_date = trade.datetime
        instrument = trade.order_book_id
        side = trade.side
        price = trade.last_price
        quantity = trade.last_quantity
        total_price = price*quantity + trade.tax + trade.transaction_cost
        # 交易状态为已成交
        status = default_status
        order_id = trade.order_id
        row_data = [trade_date,instrument,side,price,quantity,total_price,status,order_id]
        row_data = np.expand_dims(np.array(row_data),axis=0)
        # 使用numpy存储，后续动态生成DataFrame
        if self.trade_data is None:
            self.trade_data = row_data
        else:
            self.trade_data = np.concatenate((self.trade_data,row_data),axis=0)
            
    def get_trade_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的已成交交易
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = pd.DataFrame(self.trade_data,columns=TRADE_COLUMNS)
        target_df = trade_data_df[(trade_data_df["instrument"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&(trade_data_df["trade_date"]<=before_date)]
        return target_df
        
    def get_trade_date_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的已成交交易日期
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = pd.DataFrame(self.trade_data,columns=TRADE_COLUMNS)
        target_df = trade_data_df[(trade_data_df["instrument"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&(trade_data_df["trade_date"]<=pd.to_datetime(before_date))]
        if target_df.shape[0]==0:
            return None
        return target_df["trade_date"].dt.to_pydatetime().tolist()[0].strftime('%Y%m%d') 
        
