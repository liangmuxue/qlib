import time

from rqalpha.apis import *
import rqalpha
from rqalpha.const import SIDE,ORDER_STATUS
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays
from trader.emulator.sim_strategy import SimStrategy

from cus_utils.log_util import AppLogger
logger = AppLogger()

class SimBacktestStrategy(SimStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        super().__init__(proxy_name=proxy_name)
        
    
    def init_env(self):

        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
         
    def before_trading(self,context):
        """交易前准备"""
        
        super().before_trading(context)
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        super().open_auction(context, bar_dict)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        super().handle_bar(context,bar_dict)

    