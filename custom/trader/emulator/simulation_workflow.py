from rqalpha.environment import Environment
from rqalpha.core.events import EVENT

import os
import threading
import queue
import time
from datetime import datetime

from rqalpha.data.bar_dict_price_board import BarDictPriceBoard

from data_extract.rqalpha.fur_ds_proxy import FurDataProxy
from cus_utils.data_aug import DictToObject
from qlib.utils import init_instance_by_config
from trader.utils.date_util import tradedays
from trader.emulator.qidian.futures_proxy_ctp import CtpFuturesTrade
from trader.emulator.futures_real_ds import FuturesDataSourceSql

from cus_utils.log_util import AppLogger
logger = AppLogger()

class Executor(threading.Thread):
    """模拟盘执行器"""
    
    def __init__(self, trade_date,env=None,wait_time=3):
        threading.Thread.__init__(self)
        
        self.trade_date = trade_date
        self.env = env
        self.wait_time = wait_time
        # 预定义事件集合
        self.event_Coll = {EVENT.BEFORE_TRADING:0,EVENT.OPEN_AUCTION:0,EVENT.BAR:0,EVENT.AFTER_TRADING:0}
        # 业务响应事件，以应对异步问题
        self.busi_event_queue = queue.Queue(maxsize=60)
        

    def pop_event(self,now_time):
        """交易过程过程中的事件生成"""
        
        # 优先响应业务事件
        if self.busi_event_queue.qsize()>0:
            event = self.busi_event_queue.get(timeout=2)
            return event
        # 交易准备事件
        if self.event_Coll[EVENT.BEFORE_TRADING]==0:
            self.event_Coll[EVENT.BEFORE_TRADING] = 1
            return EVENT.BEFORE_TRADING
        # 8点半进行开盘竞价
        auction_time = datetime(self.trade_date.year,self.trade_date.month,self.trade_date.day,8,30,0) 
        if now_time>auction_time and self.event_Coll[EVENT.OPEN_AUCTION]==0:
            self.event_Coll[EVENT.OPEN_AUCTION] = 1
            return EVENT.OPEN_AUCTION
        # 9点进入正式交易时段，生成BAR事件
        bar_time = datetime(self.trade_date.year,self.trade_date.month,self.trade_date.day,9,0,0) 
        if now_time>bar_time:
            self.event_Coll[EVENT.BAR] += 1
            return EVENT.BAR        
        
        return None
               
    def run(self):
        """模拟盘运行，循环并进行事件推送"""
        
        # 按照间隔时间一直循环，并进行相关事件调用
        while True:
            now_time = datetime.now()
            # 更新运行环境的时间
            self.env.update_time(now_time)            
            event = self.pop_event(now_time)
            if event is not None:
                self.env.publish_event(event)
            # 间隔几秒再重复
            time.sleep(self.wait_time)
            
class AsisExecutor(Executor):
    """辅助执行器"""

    def __init__(self, trade_date,env=None,wait_time=3):
        
        super().__init__(trade_date,env=env,wait_time=wait_time)
        self.event_Coll = {EVENT.BEFORE_TRADING:0,EVENT.POST_BAR:0}  
        self.env.set_asis_execute(False)

    def pop_event(self,now_time):
        """交易过程过程中的事件生成"""
        
        # 优先响应业务事件
        if self.busi_event_queue.qsize()>0:
            event = self.busi_event_queue.get(timeout=2)
            logger.info("busi event pop,type:{}".format(event.event_type))
            return event
        # 交易准备事件
        if self.event_Coll[EVENT.BEFORE_TRADING]==0:
            self.event_Coll[EVENT.BEFORE_TRADING] = 1
            return EVENT.BEFORE_TRADING
        # 执行辅助事件,借用POST_BAR事件
        self.event_Coll[EVENT.POST_BAR] += 1
        return EVENT.POST_BAR        

class SimulationContext():
    """策略上下文"""
    
    def __init__(self,config=None):
        self.config = config       
        self.now = None
        
    def set_nowtime(self,time):
        self.now = time
        
    def get_trade_proxy(self):
        return self.trade_proxy
       
    def set_trade_proxy(self,trade_proxy):
        self.trade_proxy = trade_proxy   
                
           
class SimulationWorkflow():
    
    def __init__(self,**kwargs):
        """仿真入口(工作流模式)，这里只负责与相关的回测类进行对接"""
        
        logger.info("init in")
        self.sim_config = kwargs['simulation']
        # 使用qlib模式，动态类定义，以及传参
        self.strategy_class = init_instance_by_config(self.sim_config['strategy_class'])
        # 生成实际的策略类
        config = self.sim_config['standard']
        config = DictToObject(config) 
        self.context = SimulationContext(config)
        self.strategy_class.__build_with_context__(self,workflow_mode=True)
        # 整体数据以及上下文环境，复用RQALPHA的设计模式
        env = Environment(config)
        self.env = env
        # 设置数据源
        ds = FuturesDataSourceSql(env.config.base.data_bundle_path,stock_data_path=env.config.extra.stock_data_path,
                            sim_path=env.config.extra.stock_data_path)
        env.set_data_source(ds)  
        # 设置中间数据代理
        price_board = BarDictPriceBoard()
        data_proxy = FurDataProxy(ds,price_board)
        env.set_data_proxy(data_proxy)   
        # 设置交易代理
        proxy_config = config.mod.ext_emulation_mod.emu_args
        trade_proxy = CtpFuturesTrade(context=self,account_alias=proxy_config)      
        self.context.set_trade_proxy(trade_proxy)        
        # 策略类初始化
        self.strategy_class.init_env()
        # 执行器
        self.executor = AsisExecutor(datetime.now().date(),env=self)
        self.executor = Executor(datetime.now().date(),env=self)
        # 注册相关回调事件
        env.event_bus.add_listener(EVENT.DO_RESTORE, self.strategy_class.refresh_portfolio)   
        env.event_bus.add_listener(EVENT.ORDER_CREATION_PASS, self.strategy_class.on_order_handler)     
        env.event_bus.add_listener(EVENT.ORDER_CREATION_REJECT, self.strategy_class.on_order_handler)  
        env.event_bus.add_listener(EVENT.TRADE, self.strategy_class.on_trade_handler)     
        # 信号控制
        self.semaphore = threading.Semaphore(0)
              
    def run(self):
        """执行入口"""
        
        # 初始化行情数据环境
        # TODO
        # 开启执行器
        self.run_status = 1 
        self.executor.start()
        # 锁定当前主线程，直到结束事件生成
        self.semaphore.acquire()
    
    def update_time(self,time):
        
        self.context.set_nowtime(time)
        self.env.update_time(time, time)  
    
    def publish_event(self,event):
        """事件统一发布"""
        
        env = Environment.get_instance()
        if event == EVENT.BAR:
            self.handle_bar(bar_dict=None)
        elif event == EVENT.POST_BAR:
            self.asis_func()            
        elif event == EVENT.OPEN_AUCTION:
            self.open_auction(bar_dict=None)
        elif event == EVENT.BEFORE_TRADING:
            self.before_trading()
        else:
            # 其他的属于业务事件，异步响应模式
            env.event_bus.publish_event(event) 
 
    def before_trading(self):
        """交易前准备"""
        
        context= self.context
        if context.config.ignore_mode:
            return     
        self.strategy_class.before_trading(context)
    
    def handle_bar(self,bar_dict=None):
        """主要的算法逻辑入口"""
        
        context= self.context
        if context.config.ignore_mode:
            return      
        self.strategy_class.handle_bar(context,bar_dict=bar_dict)
        
    def open_auction(self, bar_dict=None):
        """集合竞价入口"""
        
        context= self.context
        if context.config.ignore_mode:
            return      
        self.strategy_class.open_auction(context,bar_dict=bar_dict)

    def add_busi_event(self,event):
        self.executor.busi_event_queue.put(event)

    ######################### 辅助功能实现 ####################################
    def set_asis_execute(self,flag):
        self.asis_execute = flag
    
    def asis_func(self):
        """辅助功能调用"""
        
        if not self.asis_execute:
            # 开仓指定品种
            # self.strategy_class.open_trade_order("HC2510")
            # 清空所有持仓
            # self.strategy_class.clear_position()
            # self.strategy_class.clear_order()
            self.strategy_class.query_position()     
            # self.strategy_class.query_trade()   
            # orders = self.strategy_class.query_order_info("")
            # print("orders len:{}".format(len(orders)))
            # self.strategy_class.query_account()   
            self.asis_execute = True
        else:
            return
            # 检查持仓以及订单交易情况
            self.strategy_class.query_position()
            # self.strategy_class.query_trade()
        
        
        
        
    
