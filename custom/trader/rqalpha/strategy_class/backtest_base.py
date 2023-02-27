from rqalpha.apis import *
import rqalpha

from trader.rqalpha.ml_context import MlIntergrate
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.rqalpha.trade_entity import TradeEntity
from trader.utils.date_util import tradedays

from cus_utils.log_util import AppLogger
logger = AppLogger()


# 交易信息表字段，分别为交易日期，股票代码，成交价格，成交量,总价格，成交状态，订单编号
TRADE_COLUMNS = ["trade_date","instrument","side","price","quantity","total_price","status","order_id"]

class BaseStrategy():
    """交易对象处理BASE类"""
    
    def __init__(self):
        pass
    
    def __build_with_context__(self,context):
        self.context = context
        # 使用rqalpha模式下预设值的参数
        config_path = context.config.extra.context_vars.strategy_class.config_path
        provider_uri = context.config.extra.context_vars.strategy_class.provider_uri
        # 加载qlib上下文  
        context.ml_context = MlIntergrate(ext_length=25,config_path=config_path,provider_uri=provider_uri)
        self.strategy = context.config.extra.context_vars.strategy
        # 创建预测的集成数据
        context.ml_context.create_bt_env()
        # 交易对象上下文
        context.trade_entity = TradeEntity()
        # 注册订单事件
        context.fired = False
        subscribe_event(EVENT.TRADE, self.on_trade_handler)
        subscribe_event(EVENT.ORDER_CREATION_PASS, self.on_order_handler)           
    
    def before_trading(self,context):
        pass

    def handle_bar(self,context, bar_dict):
        pass


    ############################事件注册部分######################################
    def on_trade_handler(self,context, event):
        pass
    
    def on_order_handler(self,context, event):
        pass    