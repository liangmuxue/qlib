import time
import os
import datetime
from rqalpha.apis import *
import rqalpha
from rqalpha.const import SIDE,ORDER_STATUS
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_furtures_order_book_id,transfer_instrument
from trader.rqalpha.futures_trade_entity import FuturesTradeEntity
from trader.emulator.futures_backtest_strategy import FurBacktestStrategy
from trader.utils.date_util import tradedays,get_tradedays_dur

from data_extract.his_data_extractor import PeriodType
from cus_utils.data_filter import get_topN_dict
from cus_utils.log_util import AppLogger
from gunicorn import instrument
logger = AppLogger()

class FurSimulationStrategy(FurBacktestStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        super().__init__(proxy_name=proxy_name)
    
    def init_env(self):
        
        self.data_source = Environment.get_instance().data_source
        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        # 根据标志，决定是否清空目录下的历史交易记录
        if emu_args["clear_data"]:
            self.trade_entity.clear_his_data()
            
        # 订阅合约行情
        sub_contract_names = self.data_source.get_all_contract_names(self.context.now)
        self.context.s_arr = sub_contract_names
        for name in sub_contract_names:
            subscribe(name)
         
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
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        # 临时限制时间
        if context.now.hour>=10 or (context.now.hour==9 and context.now.minute>35):
            return
        
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        self.time_line = 2
        
        # 如果非实时模式，则需要在相应前等待几秒，以保证先处理外部通知事件
        if self.handle_bar_wait:
            time.sleep(3)
        # 已提交订单检查，包括开仓和平仓
        self.verify_order_closing(context)
        self.verify_order_opening(context)
        
        # 卖出逻辑，止跌卖出        
        self.stop_fall_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，止盈卖出        
        self.stop_raise_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，持有股票超期卖出        
        self.expire_day_logic(context,bar_dict=bar_dict)     
        
        # 统一执行买卖挂单处理
        self.order_process(context)
        

    def order_process(self,context):
        """挂单流程，先平仓后开仓"""
        
        self.close_order(context)
        self.open_order(context) 
        
                
    ############################事件注册部分######################################
    
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        account.get_positions()
        self.logger_debug("on_trade_handler in,order:{}".format(order))
        # 保存成单交易对象
        self.trade_entity.add_trade(trade,multiplier=order.kwargs['multiplier'])
        # 修改当日仓位列表中的状态为已成交
        self.update_order_status(order,ORDER_STATUS.FILLED,side=order.side, context=self.context)     
        # 维护仓位数据
        self.apply_trade_pos(trade)
        # 平仓一个，就可以再开仓一个。从候选列表中挑选新的品种，放入待开仓列表中
        if order.position_effect==POSITION_EFFECT.CLOSE:
            self.get_next_candidate()          
    
    def on_order_handler(self,context, event):
        super().on_order_handler(context, event) 
                    