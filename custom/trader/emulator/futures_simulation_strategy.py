import time
import os
import datetime
import pandas as pd
import numpy as np

from rqalpha.environment import Environment

from CTPAPI.build.thosttraderapi import THOST_FTDC_OST_AllTraded,THOST_FTDC_OST_Canceled

from trader.emulator.portfolio import Portfolio
from trader.rqalpha.ml_wf_context import FurWorkflowIntergrate
from trader.emulator.futures_backtest_strategy import FurBacktestStrategy,POS_COLUMNS
from trader.rqalpha.dict_mapping import transfer_furtures_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays,get_tradedays_dur
from rqalpha.const import ORDER_STATUS,SIDE,POSITION_EFFECT,POSITION_DIRECTION

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
        self.context.get_trade_proxy().init_env()        
                     
    def before_trading(self,context):
        """交易前准备"""
        
        self.logger_info("before_trading.now:{}".format(context.now))
        
        cur_date = context.now.date()
        self.trade_day = int(cur_date.strftime('%Y%m%d'))
        env = Environment.get_instance()
        
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        # 加载当日可以交易的合约品种
        self.data_source.load_all_contract()
        # 投资组合信息加载，包括账户、持仓、交易等
        persis_path = env.config.extra.persis_path
        
        # 同步数据，从CTP远端系统中同步账户、持仓、交易等信息到本地
        if emu_args.sync_data:
            # 请求远端数据
            por_info = self.query_ctp_por_data()
            # 同步到投资组合对象
            portfolio = self.sync_portfolio(cur_date,por_info)        
            env.set_portfolio(portfolio)
            # 保存投资组合数据到本地存储
            # TODO
            # 同步当日订单
            orders = self.sync_orders(cur_date,por_info)   
            if orders is not None:
                # 同步到到交易存储类
                self.transfer_order(orders,date=cur_date)
        else:
            starting_cash = env.config.base.accounts.future 
            financing_rate = env.config.mod.sys_account.financing_rate
            portfolio = Portfolio(starting_cash,[],financing_rate,trade_date=cur_date,data_proxy=env.get_data_proxy(),persis_path=persis_path)
        env.set_portfolio(portfolio)
                
        pred_date = self.trade_day
        # 设置上一交易日，用于后续挂牌确认
        self.prev_day = self.get_previous_trading_date(self.trade_day)
        # 初始化当日合约对照表
        self.date_trading_mappings = self.data_source.build_trading_contract_mapping(context.now)        
        # 根据当前日期，进行预测计算
        context.ml_context.prepare_data(pred_date)        
        # 根据预测计算，筛选可以买入的品种
        candidate_list = self.get_candidate_list(pred_date,context=context)
        # candidate_list = ["000702"]
        self.lock_list = {}        
        candidate_order_list = {}  
        # 根据开仓数量配置，设置开仓列表
        position_max_number = self.strategy.position_max_number
        self.open_list = {} 
        self.close_list = {}     
        # 撤单列表
        self.cancel_list = []        
        # 综合候选品种以及当前已持仓品种，生成维护开仓和平仓列表
        positions = self.get_positions()
        pos_number = len(positions)
        # 由于当前可能已经进入交易时间了，因此首先查找当日订单，并维护相关数据
        exists_orders = self.trade_entity.get_order_list(context.now.date())
        exists_order_ids = []
        for index,row in exists_orders.iterrows():
            pos_number += 1
            exists_order_ids.append(row['order_book_id'].values[0])
        # 处理候选列表
        for item in candidate_list:
            trend = item[0]
            instrument = item[1]
            # 剔除没有价格数据的品种
            if not self.has_current_data(pred_date,instrument,mode="instrument"):
                logger.warning("no data for buy:{},ignore".format(instrument))
                continue
            # 代码转化为标准格式
            order_book_id = self.data_source.transfer_furtures_order_book_id(instrument,datetime.datetime.strptime(str(pred_date), '%Y%m%d'))
            # 如果已在订单中，则忽略
            if order_book_id in exists_order_ids:
                continue            
            # 以昨日收盘价格作为当前卖盘价格
            h_bar = self.data_source.history_bars(order_book_id,1,"1d",dt=context.now,fields="close")
            if h_bar is None:
                logger.warning("history bar None:{},date:{}".format(order_book_id,context.now))
                continue
            price = h_bar[0]
            
            # 根据多空标志决定买卖方向
            if trend==1:
                side = SIDE.BUY
            else:
                side = SIDE.SELL
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）,仓位类型为开仓
            order = self.create_order(order_book_id, 0, side,price,position_effect=POSITION_EFFECT.OPEN)
            # 对于开仓候选品种，如果已持仓当前品种，则进行锁仓
            if self.get_position(order_book_id) is not None:
                self.lock_list[order_book_id] = order        
                continue               
            # 加入到候选开仓订单
            candidate_order_list[order_book_id] = order
            
        # 开仓候选的订单信息，保存到上下文
        self.candidate_list = candidate_order_list   
                    
        # 循环从候选中取数，直到取完或者超出开仓数量为止
        while pos_number<position_max_number and len(self.candidate_list)>0:
            order_book_id = order.order_book_id
            # 依次从候选中选取对应品种并放入开仓列表
            candidate_order = self.get_next_candidate()
            if candidate_order is not None:
                pos_number += 1
        
        # 生成热加载数据，提升查询性能
        if self.strategy.building_hot_data:
            self.data_source.build_hot_loading_data(pred_date,self.close_list,reset=True) 
            self.data_source.build_hot_loading_data(pred_date,self.open_list)        
            self.data_source.build_hot_loading_data(pred_date,self.get_positions())    
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        self.order_process(context)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        
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
    
    def sync_portfolio(self,date,por_info):
        """ctp数据同步到投资组合"""
        
        (account,positions,orders) = por_info
        env = Environment.get_instance()
        
        persis_path = env.config.extra.persis_path
        financing_rate = env.config.mod.sys_account.financing_rate
        starting_cash = account.Available
            
        portfolio = Portfolio(starting_cash,positions,financing_rate,trade_date=date,data_proxy=env.data_proxy,persis_path=persis_path)
        return portfolio
    
    def sync_orders(self,date,por_info):
        """ctp订单数据同步"""
        
        (_,_,orders) = por_info
        if orders is None:
            return None
        
        persis_orders = []
        for order in orders:
            side = SIDE.BUY if order.Direction==0 else SIDE.SELL
            effect = POSITION_EFFECT.OPEN if order.CombOffsetFlag==0 else POSITION_EFFECT.CLOSE
            order_p = self.create_order(order.InstrumentID, order.VolumeTraded, side, order.LimitPrice,position_effect=effect)
            # 根据远端状态设置本地状态
            if order.OrderStatus==THOST_FTDC_OST_AllTraded:
                order_p.status = ORDER_STATUS.FILLED
            else:
                order_p.status = ORDER_STATUS.ACTIVE
            persis_orders.append(order_p)
        
        return persis_orders
    
    def transfer_order(self,orders,date=None):
        """"把远程订单信息同步到本地交易存储类"""
        
        # 首先移除当日全部订单数据
        moved_data = self.trade_entity.move_order_by_date(date)
        
        for order in orders:
            # 遍历从远程取得的订单信息，逐个进行业务添加
            self.trade_entity.add_or_update_order(order,str(date))          
    
    def query_ctp_por_data(self):
        """请求CTP远端系统数据"""
        
        ctp_trade_proxy = self.context.get_trade_proxy()
        # 请求远端CTP数据
        account = ctp_trade_proxy.query_account_info()
        positions = ctp_trade_proxy.query_position_info("")
        orders = ctp_trade_proxy.query_order_info("")
        
        return (account,positions,orders)
    
    def get_last_price(self,order_book_id):
        """取得指定标的最近报价信息"""

        return self.data_source.get_last_price(order_book_id)
    
    def get_last_or_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天或上一交易时间段的数据"""

        env = Environment.get_instance()
        # 如果还没有开盘，取上一交易日数据，否则取上一交易时间段数据
        if self.is_trade_opening(env.trading_dt):
            bar = self.data_source.get_last_bar(order_book_id)
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
        
        env = Environment.get_instance()
        cur_date = self.trade_day
        positions = []
        for pos in self.get_positions():
            # 如果当日无数据，则忽略
            if self.has_current_data(cur_date,pos.order_book_id,mode="symbol"):
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
                    