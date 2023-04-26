import numpy as np
import pandas as pd
import datetime
from enum import Enum

from rqalpha.const import SIDE
from rqalpha.const import ORDER_STATUS

from cus_utils.log_util import AppLogger
logger = AppLogger()

# 交易信息表字段，分别为交易日期，股票代码，成交价格，成交量,总价格，成交状态，订单编号,卖出原因
TRADE_COLUMNS = ["trade_date","order_book_id","side","price","quantity","total_price","status","order_id","sell_reason"]

class TradeEntity():
    """交易对象处理类"""
    
    def __init__(
        self,
        **kwargs,
    ):
        self.trade_data_df = pd.DataFrame(columns=TRADE_COLUMNS)
        self.trade_data_df.set_index("order_id",inplace=True)
        # 映射系统订单
        self.sys_orders = {}
        
    def add_or_update_order(self,order,trade_day):
        """添加(或更新)订单信息"""
        
        logger.debug("add_or_update_order in,order:{}".format(order))
        trade_date = order.datetime
        order_book_id = order.order_book_id
        side = order.side
        # 订单价格为冻结价格
        price = order.frozen_price
        quantity = order.quantity
        # 交易状态为进行中
        status = order.status
        order_id = order.order_id
        
        if hasattr(order, "sell_reason"):
            sell_reason = order.sell_reason
        else:
            sell_reason = 0
        row_data = [trade_date,order_book_id,side,price,quantity,0,status,order_id,sell_reason]
        row_data = np.expand_dims(np.array(row_data),axis=0)
        # 生成DataFrame
        if self.trade_data_df.shape[0]==0:
            logger.debug("add trade,data:{}".format(row_data))
            self.trade_data_df = pd.DataFrame(row_data,columns=TRADE_COLUMNS)
        else:
            item_df = self.trade_data_df.loc[(self.trade_data_df["order_book_id"]==order.order_book_id)&
                                         (self.trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_day)]
            if item_df.shape[0]>0:
                # 有可能是之前撤单后新发起的订单，这类订单需要更新
                self.trade_data_df.loc[(self.trade_data_df["order_book_id"]==order.order_book_id)&
                    (self.trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_day)] = pd.DataFrame(row_data,columns=TRADE_COLUMNS)
                logger.debug("update trade,data:{}".format(row_data))
            else:
                logger.debug("concat trade,data:{}".format(row_data))
                self.trade_data_df = pd.concat([self.trade_data_df,pd.DataFrame(row_data,columns=TRADE_COLUMNS)], axis=0)
        # 映射系统订单
        self.sys_orders[order.order_book_id] = order
    
    def get_sys_order(self,order_book_id):   
        if order_book_id in self.sys_orders:
            return self.sys_orders[order_book_id]
        return None
        
    def update_order_status(self,order,status=ORDER_STATUS.ACTIVE,price=0):
        """更新订单状态"""
        
        self.trade_data_df.loc[self.trade_data_df["order_id"]==order.order_id,"status"] = status
        if price!=0:
            self.trade_data_df.loc[self.trade_data_df["order_id"]==order.order_id,"price"] = price
                               
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
        row_data = [trade_date,order_book_id,side,price,quantity,total_price,status,order_id,order.sell_reason]
        # 使用订单号索引直接更新记录
        self.trade_data_df.loc[self.trade_data_df["order_id"]==order_id] = row_data
    
    def get_order_by_id(self,order_id):
        """根据订单号查询订单"""
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_id"]==order_id)]
        return target_df
         
    def get_trade_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的已成交交易
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&(trade_data_df["trade_date"]<=before_date)]
        return target_df
        
    def get_trade_date_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的已成交交易日期
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&
                                  (trade_data_df["trade_date"]<=pd.to_datetime(before_date))]
        if target_df.shape[0]==0:
            return None
        return target_df["trade_date"].dt.to_pydatetime().tolist()[0].strftime('%Y%m%d') 
    
    def get_order_list(self,trade_date=None):   
        """取得指定日期的订单"""
        
        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        target_df = self.trade_data_df[(self.trade_data_df["trade_date"].dt.strftime('%Y%m%d') ==trade_date)]
        return target_df  
        
    def get_sell_list_active(self,trade_date):   
        """取得所有未成交卖单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.SELL)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]      
        else:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.SELL)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]                  
        return target_df  
        
    def get_buy_list_active(self,trade_date):   
        """取得所有未成交买单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]             
        return target_df  
 
    def get_buy_list_reject(self,trade_date):   
        """取得所有被拒绝买单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)&
                                      (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)]             
        return target_df  
       
    def get_buy_list_eff(self,trade_date):   
        """取得所有指定日期的有效买单(包括已成交和待成交的)"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["side"]==SIDE.BUY)&
                                  (trade_data_df["trade_date"].dt.strftime('%Y%m%d')==trade_date)&
                        ((trade_data_df["status"]==ORDER_STATUS.ACTIVE)|(trade_data_df["status"]==ORDER_STATUS.FILLED))]             
        return target_df    
    
    def exp_trade_data(self,file_path):
        
        self.trade_data_df.to_csv(file_path)
        
        