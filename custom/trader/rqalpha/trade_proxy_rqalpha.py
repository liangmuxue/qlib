from enum import Enum, unique
import time
from rqalpha.apis import Environment
from rqalpha.const import SIDE,ORDER_STATUS as RQ_ORDER_STATUS
from rqalpha.portfolio.position import Position as RQPosition
from rqalpha.const import POSITION_DIRECTION
from rqalpha.apis import get_position,get_positions
from rqalpha.core.events import EVENT, Event
from rqalpha.model.trade import Trade
from trader.emulator.base_trade_proxy import BaseTrade


from cus_utils.log_util import AppLogger
logger = AppLogger()

class RqalphaTrade(BaseTrade):
    """掘进的仿真交易类"""
    
    def __init__(
        self,
        context=None,
        **kwargs,
    ):   
        super().__init__(context,**kwargs)  
        self.account_alias = kwargs["account_alias"]

        
    def init_env(self):
        pass
        
    def start(self):
        pass      
    
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

    def submit_order(self,order):
        """下单"""
        
        env = Environment.get_instance()
        account = env.get_account(order.order_book_id)  
        order.active()  
        self.context._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))    
        
        time.sleep(3)
        trade = Trade.__from_create__(
            order_id=order.order_id,
            price=order.price,
            amount=order.quantity,
            side=order.side,
            position_effect=order.position_effect,
            order_book_id=order.order_book_id,
            # 冻结价格取当前成交价格
            frozen_price=order.price,
            # 当日可平仓位取0
            close_today_amount=0
        )
        order.fill(trade)       
        # 手续费
        total_turnover = order.quantity * order.price
        trade._commission = total_turnover * 0.35/100
        # 印花税
        trade._tax = total_turnover * 0.35/100                
        self.context._env.event_bus.publish_event(Event(EVENT.TRADE, account=account, trade=trade, order=order))  
              
    def cancel_order(self,order):
        """撤单"""
        
        # 根据RQ订单号，查找到对应掘金订单号，并执行 
        order = self.api.find_cache_order(order_id=order.order_id,mode=OrderMode.RQALPHA.value)
        if order is None:
            logger.error("not found order:{}".format(order.order_id))
            return   
        juejin_order = order["juejin_order"]
        self.api.order_cancel([juejin_order])        
        
        
        
        