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
        
        self.logger_info("before_trading.now:{}".format(context.now))
        
        env = Environment.get_instance()
        # 初始化代理的当日环境
        env.broker.trade_proxy.init_env()
        # 时间窗定义
        self.time_line = 0
        pred_date = int(context.now.strftime('%Y%m%d'))
        self.trade_day = pred_date
        # 设置上一交易日，用于后续挂牌确认
        self.prev_day = get_previous_trading_date(self.trade_day)
        # 初始化当日合约对照表
        self.date_trading_mappings = self.data_source.build_trading_contract_mapping(context.now)        
        # 根据当前日期，进行预测计算
        context.ml_context.prepare_data(pred_date)        
        # 根据预测计算，筛选可以买入的股票
        candidate_list = self.get_candidate_list(pred_date,context=context)
        # candidate_list = [""000702"]
        # candidate_list = []
        buy_list = {}
        sell_list = {}
        # 从文件中加载的未成单的订单记录，维护到上下文
        buy_orders = self.trade_entity.get_buy_list_active(str(self.trade_day))
        for index,row in buy_orders.iterrows():   
            order = self.create_order(row["order_book_id"], row["quantity"], SIDE.BUY,row["price"],position_effect=row["position_effect"])
            buy_list[row["order_book_id"]] = order
        sell_orders = self.trade_entity.get_sell_list_active(str(self.trade_day))
        for index,row in sell_orders.iterrows():  
            order = self.create_order(row["order_book_id"], row["quantity"], SIDE.SELL,row["price"],position_effect=row["position_effect"])
            order.set_frozen_cash(0)
            order._status = row["status"]            
            sell_list[row["order_book_id"]] = order
        
        candidate_order_list = {}                  
        for item in candidate_list:
            trend = item[0]
            instrument = item[1]
            # 剔除没有价格数据的品种
            if not self.has_current_data(pred_date,instrument,mode="instrument"):
                logger.warning("no data for buy:{},ignore".format(instrument))
                continue
            # 代码转化为标准格式
            order_book_id = self.data_source.transfer_furtures_order_book_id(instrument,datetime.datetime.strptime(str(pred_date), '%Y%m%d'))
            # 如果已持仓当前品种，则忽略
            if self.get_postion_by_order_book_id(order_book_id) is not None:
                continue
            # 以昨日收盘价格作为当前卖盘价格,注意使用未复权的数据
            h_bar = history_bars(order_book_id,1,"1d",fields="close",adjust_type="none")
            if h_bar is None:
                logger.warning("history bar None:{},date:{}".format(order_book_id,context.now))
                continue
            price = h_bar[0]
            # # fake
            # if order_book_id.startswith("000410"):
            #     price = 7.17
            
            # 如果已经在待开仓列表中了，则不处理
            if trend==1 and order_book_id in buy_list:
                continue
            if trend==0 and order_book_id in sell_list:
                continue            
            # 根据多空标志决定买卖方向
            if trend==1:
                side = SIDE.BUY
            else:
                side = SIDE.SELL
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）,仓位类型为开仓
            order = self.create_order(order_book_id, 0, side,price,position_effect=POSITION_EFFECT.OPEN)
            if side==SIDE.BUY:
                buy_list[order_book_id] = order
            else:
                sell_list[order_book_id] = order
            # 加入到候选开仓订单
            candidate_order_list[order_book_id] = order
        # 需要开仓的订单信息，保存到上下文
        self.candidate_list = candidate_order_list         
        # 根据买单数量配置，设置买单列表
        position_max_number = self.strategy.position_max_number
        # 从买入候选列表中，根据配置取得待买入列表
        self.buy_list = get_topN_dict(buy_list,position_max_number)
        # 卖单保存到上下文    
        self.sell_list = sell_list
        # 新增开仓列表
        self.new_open_list = {}
        # 撤单列表
        self.cancel_list = []
        self.open_try_cnt = 0
        # 在每日开盘前计算单个品种购买的额度
        self.single_value = self.day_compute_quantity()
        
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
        env.broker.trade_proxy.handler_bar(context.now)
        self.time_line = 2
        
        # 如果之前有新增候选买入，在此添加
        for index,(k,v) in enumerate(self.new_open_list.items()):
            self.buy_list[k] = v            
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
        self.logger_debug("on_trade_handler in,order:{}".format(order))
        # 保存成单交易对象
        self.trade_entity.add_trade(trade,multiplier=order.kwargs['multiplier'])
        # 更新卖单状态
        if order.side == SIDE.SELL:
            sell_order = self.get_sell_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            sell_order._status = ORDER_STATUS.FILLED      
            # 卖出一个，就可以再买一个。从候选列表中挑选新的股票，放入待买入列表中
            self.pick_to_open_list(context)                 
        # 更新买单状态
        if order.side == SIDE.BUY:
            buy_order = self.get_buy_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            buy_order._status = ORDER_STATUS.FILLED          
    
    def on_order_handler(self,context, event):
        order = event.order
        self.logger_info("order handler,order:{}".format(order))
        # 已接单事件
        if order.status==ORDER_STATUS.ACTIVE:
            self.logger_info("order active:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            # 订单已接受，设置第二订单号          
            if order.side==SIDE.BUY:
                self.buy_list[order.order_book_id].set_secondary_order_id(order.secondary_order_id) 
            else:
                self.sell_list[order.order_book_id].set_secondary_order_id(order.secondary_order_id) 
            # 更新跟踪变量状态
            self.update_order_status(order,ORDER_STATUS.ACTIVE,side=order.side, context=self.context)   
            # 更新存储状态               
            self.trade_entity.add_or_update_order(order,str(self.trade_day))  
            return        
        # 如果订单被拒绝，则忽略,仍然保持新单状态，后续会继续下单
        if order.status==ORDER_STATUS.REJECTED:
            self.logger_info("order reject:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            self.trade_entity.add_or_update_order(order,str(self.trade_day))  
            return
        # 已撤单事件
        if order.status==ORDER_STATUS.CANCELLED:
            self.logger_info("order CANCELLED:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            # 这里需要修改状态为已撤单
            self.update_order_status(order,ORDER_STATUS.CANCELLED,side=order.side, context=self.context)      
            self.trade_entity.add_or_update_order(order,str(self.trade_day))     
            if order.side==SIDE.BUY and self.buy_list[order.order_book_id].kwargs["need_resub"]:
                self.logger_info("need resub order:{}".format(order))
                # 如果具备重新报单标志，则以最新价格重新生成订单
                cur_snapshot = self.get_current_snapshot(order.order_book_id)
                price_now = cur_snapshot.last
                # 创建新订单对象并重置原数据
                order_resub = self.create_order(order.order_book_id, order.quantity, SIDE.BUY, price_now)
                self.buy_list[order.order_book_id] = order_resub
                self.logger_debug("resub buylist set end:{}".format(order.order_book_id))
            if order.side==SIDE.SELL and self.sell_list[order.order_book_id].kwargs["need_resub"]:
                self.logger_info("need resub order:{}".format(order))
                # 如果具备重新报单标志，则以最新价格重新生成订单
                price_now = self.get_last_price(order.order_book_id)
                # 创建新订单对象并重置原数据
                order_resub = self.create_order(order.order_book_id, order.quantity, SIDE.SELL, price_now)
                order_resub.kwargs["sell_reason"] = order.kwargs["sell_reason"]
                self.sell_list[order.order_book_id] = order_resub
                self.logger_debug("resub sell_list set end:{}".format(order.order_book_id))     
                    