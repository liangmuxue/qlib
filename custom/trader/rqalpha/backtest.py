from rqalpha.apis import *
import rqalpha

from qlib.utils import init_instance_by_config

from trader.rqalpha.ml_context import MlIntergrate
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.rqalpha.trade_entity import TradeEntity
from trader.utils.date_util import tradedays

from cus_utils.log_util import AppLogger
logger = AppLogger()

    
def init(context):
    """回测入口，这里只负责与相关的回测类进行对接"""
    
    logger.info("init in")
    strategy_class_config = context.strategy_class
    # 使用qlib模式，动态类定义，以及传参
    context.strategy_class = init_instance_by_config(strategy_class_config)
    context.strategy_class.__build_with_context__(context)
    
def before_trading(context):
    """交易前准备"""
    
    context.strategy_class.before_trading(context)

def handle_bar(context, bar_dict):
    """主要的算法逻辑入口"""
    context.strategy_class.handle_bar(context,bar_dict)

    
