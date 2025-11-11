import numpy as np
import pandas as pd
import datetime
from enum import Enum
import os

from rqalpha.const import SIDE
from rqalpha.const import ORDER_STATUS

from cus_utils.log_util import AppLogger
logger = AppLogger()

class TradeEntity():
    """交易对象处理类"""

    # 交易信息表字段，分别为交易日期，品种代码，买卖类型，成交价格，成交量,总价格，成交状态，订单编号,平仓原因,附加订单编号
    TRADE_COLUMNS = ["trade_date","trade_datetime","update_datetime","order_book_id","side","position_effect","price",
                     "quantity","multiplier","total_price","status","order_id","close_reason","secondary_order_id"]
    TRADE_LOG_COLUMNS = TRADE_COLUMNS + ["create_time"]
   
    def __init__(
        self,
        save_path=None,
        log_save_path=None,
        **kwargs,
    ):
        if save_path is not None:
            self.trade_data_df = self.imp_trade_data(save_path)
            if self.trade_data_df is None or self.trade_data_df.shape[0]==0:
                self.trade_data_df = pd.DataFrame(columns=self.TRADE_COLUMNS)
                self.sys_orders = {} 
        else:
            self.trade_data_df = pd.DataFrame(columns=self.TRADE_COLUMNS)
            # 映射系统订单
            self.sys_orders = {}            

        self.save_path = save_path
        # 初始化交易日志数据
        self.trade_log_df = pd.DataFrame(columns=self.TRADE_LOG_COLUMNS)
        self.log_save_path = log_save_path
        
    def clear_his_data(self):
        self.trade_data_df = pd.DataFrame(columns=self.TRADE_COLUMNS)
        self.exp_trade_data(self.save_path)
           
        
        
    def get_sys_order(self,order_book_id):   
        if order_book_id in self.sys_orders:
            return self.sys_orders[order_book_id]
        return None
        
    def update_order_status(self,order,status=ORDER_STATUS.ACTIVE,price=0):
        """更新订单状态"""
        
        print("update_order_status in")
        self.trade_data_df.loc[self.trade_data_df["order_id"]==order.order_id,"status"] = status
        if price!=0:
            self.trade_data_df.loc[self.trade_data_df["order_id"]==order.order_id,"price"] = price
        # 变更后保存数据
        if self.save_path is not None:
            self.exp_trade_data(self.save_path)
                                           
    def get_order_by_id(self,order_id):
        """根据订单号查询订单"""
        
        trade_data_df = self.trade_data_df
        # logger.debug("trade_data_df in get order:{}".format(trade_data_df))
        target_df = trade_data_df[(trade_data_df["order_id"]==order_id)]
        if target_df.shape[0]==0:
            return None
        return target_df.iloc[0]
         
    def get_trade_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的已成交交易
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&(trade_data_df["trade_datetime"]<=before_date)]
        return target_df
        
    def get_trade_date_by_instrument(self,order_book_id,trade_side,before_date):  
        """查询某个股票的最近已成交交易日期
            Params:
                order_book_id 股票编码
                trade_side 买卖类别
                before_date 查询指定日期之前的交易
        """
        
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["order_book_id"]==order_book_id)
                                  &(trade_data_df["side"]==trade_side)&
                                  (trade_data_df["trade_datetime"]<=pd.to_datetime(before_date))]
        if target_df.shape[0]==0:
            return None
        # 取得最后一个交易
        return target_df["trade_datetime"].dt.to_pydatetime().tolist()[-1].strftime('%Y%m%d') 
    
    def get_exits_order_list(self,trade_date=None):   
        """取得指定日期的当前以报订单"""
        
        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        # 查询已报单的活动中状态的记录
        target_df = self.trade_data_df[(self.trade_data_df["trade_datetime"].dt.strftime('%Y%m%d') ==trade_date)&
                                       (self.trade_data_df["status"]==ORDER_STATUS.ACTIVE)]
        return target_df  
        
    def get_sell_list_active(self,trade_date):   
        """取得所有未成交卖单"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        if trade_date is not None:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.SELL)&(trade_data_df["status"]==ORDER_STATUS.ACTIVE)&
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]      
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
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]       
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
                                      (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)]       
        else:
            target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["status"]==ORDER_STATUS.REJECTED)]             
        return target_df  
       
    def get_buy_list_eff(self,trade_date):   
        """取得所有指定日期的有效买单(包括已成交和待成交的)"""

        if self.trade_data_df.shape[0]==0:
            return self.trade_data_df
        trade_data_df = self.trade_data_df
        target_df = trade_data_df[(trade_data_df["side"]==SIDE.BUY)&(trade_data_df["side"]==SIDE.BUY)&
                                  (trade_data_df["trade_datetime"].dt.strftime('%Y%m%d')==trade_date)&
                        ((trade_data_df["status"]==ORDER_STATUS.ACTIVE)|(trade_data_df["status"]==ORDER_STATUS.FILLED))]             
        return target_df    
    
    def exp_trade_data(self,file_path):
        # pd.set_option('display.max_columns', 20) 
        self.trade_data_df.to_csv(file_path,index=False)
        
    def imp_trade_data(self,file_path):
        """从文件加载交易订单信息"""
        
        if not os.path.exists(file_path):
            return None
        trade_data = pd.read_csv(file_path,parse_dates=['trade_datetime'],infer_datetime_format=True)  
        self.sys_orders = {}    
        # 追加到系统订单信息中
        for index,row in trade_data.iterrows():   
            self.sys_orders[row["order_book_id"]] = row
        return trade_data     

    def add_log(self,row_data):
        now_time = datetime.datetime.now()
        # logger.debug("add log,row_data:{},now_time:{}".format(row_data,now_time))
        log_row_data = row_data + [now_time]
        log_row_data = np.expand_dims(np.array(log_row_data),axis=0)
        self.trade_log_df = pd.concat([self.trade_log_df,pd.DataFrame(log_row_data,columns=self.TRADE_LOG_COLUMNS)], axis=0)
        self.trade_log_df.to_csv(self.log_save_path,index=False)
        
