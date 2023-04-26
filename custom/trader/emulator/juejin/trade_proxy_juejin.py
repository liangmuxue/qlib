from enum import Enum, unique

from rqalpha.const import SIDE,ORDER_STATUS as RQ_ORDER_STATUS
from rqalpha.portfolio import Portfolio
from rqalpha.portfolio.position import Position
from rqalpha.const import POSITION_DIRECTION
from rqalpha.core.events import EVENT, Event
from rqalpha.model.trade import Trade
from trader.emulator.base_trade_proxy import BaseTrade
from trader.emulator.juejin.callback import callback_controller
from trader.emulator.juejin.storage import ctx

from .trade import set_token,set_endpoint,account,login,get_positions,get_cash

from gmtrade.api import PositionSide_Long,PositionSide_Short,order_volume,OrderSide_Buy,OrderSide_Sell,OrderType_Limit,PositionEffect_Open
from gmtrade.csdk.c_sdk import c_status_fail,py_gmi_set_data_callback, py_gmi_start, py_gmi_stop
from gmtrade.enum import *

from cus_utils.log_util import AppLogger
logger = AppLogger()

@unique
class OrderMode(Enum):
    """卖出类别，1.根据预测数据 2.止盈 3.止损 4.超期卖出"""
    JUEJIN = 1 
    RQALPHA = 2
    SYMBOL = 3

class TraderSpi(object):
    def __init__(self):
        self.cache_orders = []
    
    def append_order(self,order):
        self.cache_orders.append(order)
        logger.debug("append order,self.cache_orders:{}".format(self.cache_orders))
    
    def find_cache_order(self,juejin_order=None,order_id=None,mode=OrderMode.JUEJIN.value):
        """查找已生成订单
           Params: 
               order_id 订单编号
               mode 订单编号类别，1 掘金订单 2 rq订单
        """
        
        # 新增订单阶段，只能通过股票代码和挂单金额进行查找
        if mode==OrderMode.SYMBOL.value:
            order_book_id = self.transfer_symbol(juejin_order.symbol, mode=1)
            logger.debug("juejin_order.symbol:{},price:{}".format(order_book_id,juejin_order.price))
            for order in ctx.cached_orders:
                logger.debug("order.order_book_id:{},order.price:{}".format(order.order_book_id,order.price))
                # 如果委托价格和股票代码都相等，则匹配该订单
                if order.order_book_id==order_book_id and order.price==juejin_order.price:
                    return order
            return None
        # 后续阶段通过订单自身的编号进行匹配
        for order in ctx.cached_orders:
            if order["juejin_order"].cl_ord_id==order_id:
                return order
    
    def transfer_symbol(self,symbol,mode=1):
        """股票代码规范转换,1--掘金转RQ 2--RQ转掘金"""
        
        if mode==1:
            symbol_number = symbol.split(".")[1]
            end_fix = symbol.split(".")[0]
            if end_fix=="SHSE":
                # 沪市A股 
                rq_end_fix = "XSHG"
            else:
                # 深市A股 
                rq_end_fix = "XSHE"               
            symbol_rtn = "{}.{}".format(symbol_number,rq_end_fix)
        if mode==2:
            symbol_number = symbol.split(".")[0]
            end_fix = symbol.split(".")[1]
            if end_fix=="XSHG":
                # 沪市A股 
                rq_pre_fix = "SHSE"
            else:
                # 深市A股 
                rq_pre_fix = "SZSE"               
            symbol_rtn = "{}.{}".format(rq_pre_fix,symbol_number)    
        return symbol_rtn
    #################################################事件回调###############################################################
    
    # 委托状态变化时触发
    def on_order_status(self,juejin_order):
        logger.info('order_stats_count:{}'.format(juejin_order))
        # 已报单事件回调
        if juejin_order.status==OrderStatus_New:
            order = self.find_cache_order(juejin_order,mode=OrderMode.SYMBOL.value)
            if order is None:
                logger.error("not found order:{}".format(juejin_order))
                return            
            # 放入掘金订单信息
            order["juejin_order"] = juejin_order            
            order.active()
            ctx.context.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))     
        # 成交事件回调
        if juejin_order.status==OrderStatus_Filled:
            order = self.find_cache_order(juejin_order.cl_ord_id,mode=OrderMode.JUEJIN.value)
            if order is None:
                logger.error("not found order:{}".format(juejin_order))
                return               
            order.status = RQ_ORDER_STATUS.FILLED
        # 订单拒绝事件回调
        if juejin_order.status==OrderStatus_Rejected:
            logger.debug("reject process")
            order = self.find_cache_order(juejin_order,mode=OrderMode.SYMBOL.value)
            if order is None:
                logger.error("not found order:{}".format(juejin_order))
                return       
            logger.debug("order get,status is:{}".format(order._status))        
            order._status = RQ_ORDER_STATUS.REJECTED    
            # 发布RQ事件
            ctx.context.event_bus.publish_event(Event(EVENT.ORDER_CREATION_REJECT, account=account, order=order))    
        # 订单已撤单事件回调
        if juejin_order.status==OrderStatus_Canceled:
            order = self.find_cache_order(order_id=juejin_order.cl_ord_id,mode=OrderMode.JUEJIN.value)
            if order is None:
                logger.error("not found order:{}".format(juejin_order))
                return               
            order.status = RQ_ORDER_STATUS.CANCELLED 
            ctx.context.event_bus.publish_event(Event(EVENT.ORDER_CANCELLATION_PASS, account=account, order=order))
                                 
    def on_execution_report(self,execrpt):
        """委托执行回报的事件回调，委托成交时触发"""        

        order = self.find_cache_order(order_id=execrpt.cl_ord_id,mode=OrderMode.JUEJIN.value)
        if order is None:
            logger.error("not found order:{}".format(execrpt.cl_ord_id))
            return     
        trade = Trade.__from_create__(
            order_id=order.order_id,
            price=execrpt.price,
            amount=execrpt.volumn,
            side=order.side,
            position_effect=order.position_effect,
            order_book_id=order.order_book_id,
            # 冻结价格取当前成交价格
            frozen_price=execrpt.price,
            # 当日可平仓位取0
            close_today_amount=0
        )
        order.fill(trade)        
        trade = order["trade"]   
        # 手续费
        trade._commission = execrpt.commission
        # 印花税
        trade._tax = self._env.get_trade_tax(trade)     
        self.context.event_bus.publish_event(Event(EVENT.TRADE, account=account, trade=trade, order=order))   
           
    # 交易服务连接成功后触发
    def on_trade_data_connected(self):
        logger.info('on_trade_data_connected in.................')
        
    # 交易服务断开后触发
    def on_trade_data_disconnected(self):
        logger.debug('已断开交易服务.................')
    
    # 回报到达时触发
    def on_account_status(self,account_status):
        logger.debug(f'on_account_status status={account_status}')
        
    def on_error(self,error):
        logger.debug(f'on_error error={error}')

    def on_shutdown(self):
        logger.debug('on_shutdown in')
                              
class JuejinTrade(BaseTrade):
    """掘进的仿真交易类"""
    
    def __init__(
        self,
        context=None,
        **kwargs,
    ):   
        super().__init__(context,**kwargs)  
        self.token = kwargs["token"]
        self.end_point = kwargs["end_point"]
        self.account_id = kwargs["account_id"]
        self.account_alias = kwargs["account_alias"]
        
        # 初始化环境,植入外部上下文环境
        self.api = TraderSpi()
        # 登录
        self.init_env()
        # 启动仿真环境
        status = self.start()
        
    def init_env(self):
         
        # 设置基础配置
        set_token(self.token)
        set_endpoint(self.end_point )
        # 登录
        a1 = account(account_id=self.account_id, account_alias=self.account_alias)
        login(a1)  # 注意，可以输入账户也可以输入账户组成的list
        
    def start(self):
        
        ctx.on_execution_report_fun = self.api.on_execution_report
        ctx.on_order_status_fun = self.api.on_order_status
        ctx.on_account_status_fun = self.api.on_account_status
    
        ctx.on_trade_data_connected_fun = self.api.on_trade_data_connected
        ctx.on_trade_data_disconnected_fun = self.api.on_trade_data_disconnected
    
        ctx.on_error_fun = self.api.on_error
        ctx.on_shutdown_fun = self.api.on_shutdown
    
        py_gmi_set_data_callback(callback_controller)  # 设置事件处理的回调函数
        
        global running
        status = py_gmi_start()  # type: int
        if c_status_fail(status, 'gmi_start'):
            running = False
            return status
        else:
            # 连接后，植入环境变量
            ctx.context = self.context._env
            ctx.own_caller = self.context
            running = True
        return status        
    
    def get_position(self,order_book_id):  
        positions = self.get_positions()
        for pos in positions:
            if pos.order_book_id==order_book_id:
                return pos
            
    def get_positions(self):  
        """取得持仓信息""" 
        
        positions = get_positions()
        positions_rtn = []
        for pos in positions:
            # 持仓方向
            if pos.side==PositionSide_Long:
                side = POSITION_DIRECTION.LONG
            if pos.side==PositionSide_Short:
                side = POSITION_DIRECTION.SHORT             
            pos_rtn = Position(pos.symbol,side)
            # 持仓量         
            pos_rtn.quantity = pos.volume 
            # 持仓市值 
            pos_rtn.market_value = pos.amount
            positions_rtn.append(pos_rtn)
        return positions_rtn
        
    def get_portfolio(self):
        """取得当前快照信息,借用rqalpha相关对象"""        
        
        portfolio = Portfolio()
        # 掘进的get_cash功能和快照蕾西
        cash = get_cash()    
        # 统一接口规范
        portfolio.cash = cash.available
        portfolio.frozen_cash = cash.order_frozen # 冻结资金
        portfolio.market_value = cash.frozen # 持仓资金

    def submit_order(self,order):
        """下单"""
        
        # 数据规范转换，包括股票代码、成交量、报价、成交方向等
        symbol = self.api.transfer_symbol(order.order_book_id, mode=2)
        volume = order.quantity
        price = order.price
        if order.side==SIDE.BUY:
            target_side = OrderSide_Buy
        if order.side==SIDE.SELL:
            target_side = OrderSide_Sell           
         
        order_volume(symbol=symbol, volume=volume, side=target_side, order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=price)
        # 加入匹配订单队列      
        ctx.cached_orders.append(order)   
        
    def cancel_order(self,order):
        """撤单"""
        
        # 根据RQ订单号，查找到对应掘金订单号，并执行 
        order = self.find_cache_order(order_id=order.order_id,mode=OrderMode.RQALPHA.value)
        if order is None:
            logger.error("not found order:{}".format(order.order_id))
            return   
        juejin_order = order["juejin_order"]
        self.api.order_cancel([juejin_order])        
        
        
        
        