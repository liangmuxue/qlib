from enum import Enum, unique
import time
import threading
import copy
import numpy as np

from rqalpha.apis import Environment
from rqalpha.const import SIDE,ORDER_STATUS as RQ_ORDER_STATUS
from rqalpha.portfolio.position import Position as RQPosition
from rqalpha.const import POSITION_DIRECTION
from rqalpha.apis import get_position,get_positions
from rqalpha.core.events import EVENT, Event
from trader.rqalpha.model.trade import Trade
from trader.emulator.base_trade_proxy import BaseTrade


from cus_utils.log_util import AppLogger
logger = AppLogger()
 
class FuturesTrade(BaseTrade):
    """期货的仿真交易类"""
    
    def __init__(
        self,
        context=None,
        **kwargs,
    ):   
        super().__init__(context,**kwargs)  
        self.account_alias = kwargs["account_alias"]
        self.order_queue = []
        self.start()
        
    def init_env(self):
        # 待处理订单序列
        self.order_queue = []
        
    def start(self):
        pass
        # # 启动撮合线程
        # order_thread = OrderListThread("order_thread",self)
        # order_thread.start()    
    
    ########################内部数据同步#################################
    
    def handler_bar(self,context_now):
        """模拟撮合，通过bar事件触发"""
        
        env = Environment.get_instance()
        q_list = self.get_order_queue()
        # 不在交易时间则不处理
        if not env.data_source.is_trade_opening(context_now):
            return
        for order in q_list:
            # 对待处理订单进行判定
            if order.status==RQ_ORDER_STATUS.ACTIVE:
                # 取得当前价格，如果小于订单报价，则成单处理
                cur_price = env.data_source.get_last_price(order.order_book_id,env.trading_dt)  
                if cur_price is None or np.isnan(cur_price):
                    continue
                # 卖单的挂单价需要小于等于当前价格
                if order.price>cur_price and order.side==SIDE.SELL:
                    continue
                # 买单的挂单价需要大于等于当前价格
                if order.price<cur_price and order.side==SIDE.BUY:
                    continue           
                try:     
                    trade = Trade.__from_create__(
                        order_id=order.order_id,
                        price=cur_price,
                        amount=order.quantity,
                        side=order.side,
                        position_effect=order.position_effect,
                        order_book_id=order.order_book_id,
                        # 冻结价格取当前成交价格
                        frozen_price=cur_price,
                        # 当日可平仓位取0
                        close_today_amount=0
                    )
                except RuntimeError as e:
                    logger.error("trade create err,order.order_book_id:{},price:{},amount:{},error:{}".format(order.order_book_id,cur_price,order.quantity,e))
                    continue
                order.fill(trade)       
                # 手续费
                total_turnover = order.quantity * order.price
                trade._commission = total_turnover * 0.03/100
                # 印花税
                trade._tax = total_turnover * 0.01/100      
                account = env.get_account(order.order_book_id)  
                # 修改撮合系统的订单状态
                self.update_order_status(order.order_book_id,RQ_ORDER_STATUS.FILLED)                       
                # 发送成单事件          
                env.event_bus.publish_event(Event(EVENT.TRADE, account=account, trade=trade, order=order))       
                            
    def get_context_now(self):
        env = Environment.get_instance()
        cnow = env.calendar_dt.time()
        return cnow    
    
    def context_time_match(self,outer_time):
        cnow = self.get_context_now()
        if outer_time.hour==cnow.hour and outer_time.minute==cnow.minute:
            return True
        return False
    
    def get_order_queue(self):
        return self.order_queue
    
    def update_order_status(self,order_book_id,status):
        for order in self.order_queue:
            if order.order_book_id==order_book_id:
                order._status = status
    
    
    ########################交易数据获取#################################  
     
    def get_position(self,order_book_id):  
        positions = get_positions()
        for pos in positions:
            if pos.order_book_id==order_book_id:
                return pos
            
    def get_positions(self):  
        """取得持仓信息""" 
        
        positions = get_positions()
        return positions
        
    def get_portfolio(self):
        """取得当前快照信息"""   
             
        env = Environment.get_instance()
        portfolio = env.portfolio
        return portfolio
    
    ########################事件调用#################################  
    
    def submit_order(self,order):
        """下单"""
        
        env = Environment.get_instance()
        account = env.get_account(order.order_book_id)  
        order.active()  
        self.context._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))   
        # 放入待处理队列
        order_in = copy.copy(order)
        self.order_queue.append(order_in)

    def cancel_order(self,order):
        """撤单"""
        
        env = Environment.get_instance()
        for order_q in self.order_queue:
            if order_q.order_book_id==order.order_book_id:
                # 对于待处理的订单，进行取消处理
                if order_q.status==RQ_ORDER_STATUS.ACTIVE:
                    order_q._status = RQ_ORDER_STATUS.CANCELLED
                    # 发送取消成功的事件
                    account = env.get_account(order.order_book_id) 
                    self.context._env.event_bus.publish_event(Event(EVENT.ORDER_CANCELLATION_PASS, account=account, order=order_q))  
                else:
                    logger.warning("order can not cancel:{}".format(order_q))       
        
        
        