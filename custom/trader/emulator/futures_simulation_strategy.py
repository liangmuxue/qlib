import time
import os
import datetime
import pandas as pd
import numpy as np

from rqalpha.environment import Environment

from trader.rqalpha.ml_wf_context import FurWorkflowIntergrate
from trader.emulator.futures_backtest_strategy import FurBacktestStrategy,POS_COLUMNS
from trader.rqalpha.dict_mapping import transfer_furtures_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays,get_tradedays_dur
from rqalpha.const import ORDER_STATUS

from cus_utils.log_util import AppLogger
logger = AppLogger()

class FurSimulationStrategy(FurBacktestStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        self.time_begin = None
        self.proxy_name = proxy_name
        
        # 设置策略模拟仓位，用于策略逻辑判断
        self.sim_position = pd.DataFrame(columns=POS_COLUMNS)   
        self.contract_today = {}

    def __build_with_context__(self,context,workflow_mode=False):
        self.context = context
        provider_uri = context.config.provider_uri
        # 加载qlib上下文  
        task_config = context.config
        context.ml_context = FurWorkflowIntergrate(task_config=task_config,provider_uri=provider_uri,ext_length=25
                                    ,task_id=context.config.extra.task_id,dump_path=context.config.extra.dump_path)
        self.strategy = context.config.extra.context_vars.strategy
        # 交易对象上下文
        save_path = context.config.extra.report_save_path
        data_save_path = save_path + "/trade_data.csv"
        log_save_path = save_path + "/trade_data_log.csv"
        self.trade_entity = self.create_trade_entity(save_path=data_save_path,log_save_path=log_save_path)
           
    def init_env(self):
        
        env = Environment.get_instance()
        self.data_source = env.data_source
        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        # 根据标志，决定是否清空目录下的历史交易记录
        if emu_args.clear_data:
            self.trade_entity.clear_his_data()
        # 加载当日可以交易的合约品种
        self.data_source.load_all_contract()
        # sub_contract_names = self.data_source.get_all_contract_names(env.trading_dt)
        # for name in sub_contract_names:
        #     self.contract_today.append(name)
                     
    def before_trading(self,context):
        """交易前准备"""
        
        super().before_trading(context)
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        self.order_process(context)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        # 临时限制时间
        if context.now.hour>=10 or (context.now.hour==9 and context.now.minute>35):
            return
        
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        self.time_line = 2
        
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

    def submit_order(self,amount,order_in=None):
        """代理api的订单提交方法"""
        
        order_book_id = order_in.order_book_id
        env = Environment.get_instance()
        order_in._quantity = int(amount)

        if self.can_submit_order(order_book_id):
            # 订单编号转换为字符串
            if not str(order_in._order_id).startswith("rq_"):
                order_in._order_id = "rq_{}".format(order_in._order_id)    
            # 添加到本地订单库
            self.trade_entity.add_or_update_order(order_in,str(self.trade_day))            
            # 调用代理方法        
            self.context.get_trade_proxy().submit_order(order_in)
            return order_in
    
    def cancel_order(self,order):
        """撤单，直接调用broker的方法"""
        
        self.logger_info("cancel_order in ,order:{}".format(order.order_book_id))
        # 这里需要修改状态为待取消
        self.update_order_status(order,ORDER_STATUS.PENDING_CANCEL,side=order.side, context=self.context,price=order.price)     
        self.trade_entity.add_or_update_order(order,str(self.trade_day))
        # 调用代理的撤单方法    
        self.context.get_trade_proxy().cancel_order(order)
                
    ###############################数据逻辑处理部分########################################  

    def get_last_price(self,order_book_id):
        """取得指定标的最近报价信息"""

        return self.data_source.get_last_price(order_book_id)
    
    def get_last_or_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天或上一交易时间段的数据"""

        env = Environment.get_instance()
        # 如果还没有开盘，取上一交易日数据，否则取上一交易时间段数据
        if self.is_trade_opening(env.trading_dt):
            bar = self.data_source.get_last_price(order_book_id)
        else:
            prev_day = get_tradedays_dur(env.trading_dt, -1)
            bar = self.data_source.get_bar(order_book_id,prev_day,"1d")            
        if bar is not None and not np.isnan(bar['open']):
            return bar       
        return None

    def get_portfolio(self):
        """取得投资组合信息"""
        
        env = Environment.get_instance()
        return env.portfolio
    
    def get_positions(self):
        """取得持仓信息"""
    
        env = Environment.get_instance()
        return env.portfolio.get_positions()
    
    def get_ava_positions(self):
        """取得当日可以买卖的持仓信息"""
        
        positions = []
        for pos in self.get_positions():
            symbol = transfer_instrument(pos.order_book_id)
            # 如果当日无数据，则忽略
            if self.has_current_data(symbol):
                positions.append(pos)
        return positions 
           
    def has_current_data(self,date,code,mode="instrument"):
        """当日是否开盘交易,使用懒加载缓存模式"""

        if mode=="instrument":
            # 品种模式查询，需要先根据品种取得合约代码再查询
            symbol = self.data_source.get_main_contract_name(code,date)
        else:
            symbol = code
                                
        if symbol in self.contract_today:
            # 如果已经在缓存里，则直接放回缓存中的结果
            return self.contract_today[symbol]==1
        
        # 取得实时价格，如果有则说明当日有交易
        price = self.data_source.get_last_price(symbol)
        flag = 1
        if price is None:
            flag = 0
        # 写入缓存
        self.contract_today[symbol] = flag
        
        return self.contract_today[symbol]==1
                    
    ############################事件注册部分######################################
    
    def on_trade_handler(self,context, event):
        super().on_trade_handler(context, event)         
    
    def on_order_handler(self,context, event):
        super().on_order_handler(context, event)    
                    