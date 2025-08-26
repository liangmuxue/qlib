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
TRADE_COLUMNS = ["trade_datetime","update_datetime","order_book_id","side","long_short","price","quantity","total_price","status","order_id","close_reason","secondary_order_id"]
TRADE_LOG_COLUMNS = TRADE_COLUMNS + ["create_time"]
LOCK_COLUMNS = ['date','order_book_id','instrument']

class FuturesTradeEntity(TradeEntity):
    """期货交易对象处理类"""
    
    def __init__(
        self,
        save_path=None,
        log_save_path=None,
        **kwargs,
    ):
        if save_path is not None:
            self.trade_data_df = self.imp_trade_data(save_path)
            if self.trade_data_df is None or self.trade_data_df.shape[0]==0:
                self.trade_data_df = pd.DataFrame(columns=TRADE_COLUMNS)
                self.sys_orders = {} 
        else:
            self.trade_data_df = pd.DataFrame(columns=TRADE_COLUMNS)
            # 映射系统订单
            self.sys_orders = {}            

        self.save_path = save_path
        # 锁定品种保存路径
        self.lock_save_path = os.path.join(os.path.dirname(self.save_path),"lock.csv")
        self.lock_data = pd.DataFrame(columns=LOCK_COLUMNS)
        # 初始化交易日志数据
        self.trade_log_df = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        self.log_save_path = log_save_path
    
    def set_trade_data(self,trade_data):
        self.trade_data_df = trade_data
      
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
        trade_datetime = order['trade_datetime']
        update_datetime = datetime.datetime.now()
        row_data = [trade_datetime,update_datetime,order_book_id,side,position_effect,price,quantity,multiplier,total_price,status,order_id,close_reason,secondary_order_id]
        # 使用订单号查询并更新记录
        self.trade_data_df[self.trade_data_df["order_id"]==order_id] = row_data
        # 变更后保存数据
        if self.save_path is not None:
            self.exp_trade_data(self.save_path)   
            # 日志记录
            self.add_log(row_data)
    
    def update_status(self,order):
        """修改订单状态"""
        
        print("update_order_status in,order:{}".format(order))
        # 通过订单编号定位记录并修改
        df = self.trade_data_df
        df.loc[df['order_id']==order.order_id,'status'] = order.status
        # 同时更新时间
        df.loc[df['order_id']==order.order_id,'update_datetime'] = datetime.datetime.now()
        self.exp_trade_data(self.save_path)  
    
    def get_trade_by_date(self,date):
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[trade_data_df["trade_datetime"]==pd.to_datetime(date)]
        return target_df
               
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
                                  (trade_data_df["trade_datetime"]<=pd.to_datetime(before_date))]
        if target_df.shape[0]==0:
            return None
        # 取得最后一个交易
        return target_df["trade_datetime"].dt.to_pydatetime().tolist()[-1].strftime('%Y%m%d') 

    def get_trade_in_pos(self,order_book_id):   
        """取得指定的已持仓交易订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)]               
        return target_df  
    
    def get_open_list(self,trade_date):   
        """取得所有已开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]                  
        return target_df  
    
    def get_open_list_filled(self,trade_date):   
        """取得所有已成交订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.FILLED)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]                  
        return target_df  
    
    def get_open_list_active(self,trade_date):   
        """取得所有未成交开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]      
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]                  
        return target_df  

    def get_open_order_active(self,trade_date,order_book_id):   
        """取得指定未成交开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["order_book_id"]==order_book_id)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]  
        if target_df.shape[0]==0:
            return None    
        return target_df.iloc[0]
       
    def get_close_list_active(self,trade_date):   
        """取得所有未成交平仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]      
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)]                  
        return target_df  
    
    def get_open_list_reject(self,trade_date):   
        """取得所有被拒绝的开仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.OPEN)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)]             
        return target_df         

    def get_close_list_reject(self,trade_date):   
        """取得所有被拒绝的平仓订单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["position_effect"]==POSITION_EFFECT.CLOSE)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)]             
        return target_df   
        
    def move_order_by_date(self,date): 
        """移除指定日期的数据"""
        
        data = self.get_trade_by_date(date)
        target_df = self.trade_data_df[self.trade_data_df["trade_datetime"]!=pd.to_datetime(date)]
        self.trade_data_df = target_df
        
        return data
        
    def add_lock_candidate(self,lock_list,date):
        """锁定品种保存到本地存储"""
        
        for item in lock_list:
            instrument = item[:-4]
            item_data = pd.DataFrame(np.array([[date,item,instrument]]),columns=LOCK_COLUMNS)
            self.lock_data = pd.concat([self.lock_data,item_data])
        self.lock_data.to_csv(self.lock_save_path,index=False)
    
    def get_lock_item(self,date,instrument):
        
        lock_data = self.lock_data
        return lock_data[(lock_data['date']==date)&(lock_data['instrument']==instrument)]
        
                