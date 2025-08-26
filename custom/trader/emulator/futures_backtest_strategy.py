import time
import os
import datetime
from datetime import timedelta
from rqalpha.apis import *
from rqalpha.const import SIDE,ORDER_STATUS
from rqalpha.data.bar_dict_price_board import BarDictPriceBoard
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_futures_order_book_id,transfer_instrument
from trader.rqalpha.futures_trade_entity import FuturesTradeEntity
from trader.emulator.sim_strategy import SimStrategy
from trader.utils.date_util import tradedays,get_tradedays_dur,is_working_day

from data_extract.rqalpha.fur_ds_proxy import FurDataProxy
from data_extract.his_data_extractor import PeriodType
from cus_utils.data_aug import DictToObject
from cus_utils.data_filter import get_topN_dict
from cus_utils.log_util import AppLogger
from gunicorn import instrument
logger = AppLogger()

def get_time(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        if f.__name__=="build_hot_loading_data":
            print('func:{},time:{}'.format(f.__name__,(e_time - s_time)))
        return res
    return inner

POS_COLUMNS = ['order_book_id', 'quantity','side', 'direction', 'avg_price', 'datetime', 'order_id']

class FurBacktestStrategy(SimStrategy):
    """回测交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        super().__init__(proxy_name=proxy_name)
        self.time_begin = None
        
        # 设置策略模拟仓位，用于策略逻辑判断
        self.sim_position = pd.DataFrame(columns=POS_COLUMNS)        
    
    def time_inject(self,code_name=None,begin=False):
        if begin:
            self.time_begin = time.time()
        else:
            if self.time_begin is None:
                self.time_begin = time.time()
            elapsed_time = time.time() - self.time_begin   
            self.time_begin = time.time()
            # print("{} Elapsed time: {} seconds".format(code_name,elapsed_time))    
        
    def init_env(self):
        
        self.data_source = Environment.get_instance().data_source
        price_board = BarDictPriceBoard()
        data_proxy = FurDataProxy(self.data_source,price_board)
        Environment.get_instance().set_data_proxy(data_proxy)
        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        # 根据标志，决定是否清空目录下的历史交易记录
        if emu_args["clear_data"]:
            self.trade_entity.clear_his_data()

        # 加载当日可以交易的合约品种
        # self.data_source.load_all_contract()               
        # 订阅合约行情
        sub_contract_names = self.data_source.get_all_contract_names(self.context.now.strftime("%Y%m%d"))
        self.context.s_arr = sub_contract_names
        for name in sub_contract_names:
            self.subscribe_market_trading(name)
        # 初始化代理的当日环境
        env = Environment.get_instance()
        env.broker.trade_proxy.init_env()
    
    def subscribe_market_trading(self,id_or_symbols):
        """订阅合约行情"""
        
        current_universe = Environment.get_instance().get_universe()
        if isinstance(id_or_symbols, six.string_types):
            instrument_arr = self.data_source.get_instruments([id_or_symbols])
            if instrument_arr is None or len(instrument_arr)==0:
                logger.warning("instruments Empty:{}".format(id_or_symbols))
                return 
            order_book_id = instrument_arr[0].order_book_id
            current_universe.add(order_book_id)
        elif isinstance(id_or_symbols, Instrument):
            current_universe.add(id_or_symbols.order_book_id)
        elif isinstance(id_or_symbols, Iterable):
            for item in id_or_symbols:
                current_universe.add(assure_order_book_id(item))
        else:
            raise RQInvalidArgument(_(u"unsupported order_book_id type"))
        verify_that("id_or_symbols")._are_valid_instruments("subscribe", id_or_symbols)
        Environment.get_instance().update_universe(current_universe)
               
    def before_trading(self,context):
        """交易前准备"""
        
        self.logger_info("before_trading.now:{}".format(context.now))
        
        # 时间窗定义
        self.time_line = 0
        pred_date = int(context.now.strftime('%Y%m%d'))
        self.trade_day = pred_date
        # 设置上一交易日，用于后续挂牌确认
        self.prev_day = self.get_previous_trading_date(self.trade_day)
        if pred_date==20240823:
            print("ggg")
        # 初始化当日合约对照表
        self.date_trading_mappings = self.data_source.build_trading_contract_mapping(context.now)        
        # 根据当前日期，进行预测计算
        context.ml_context.prepare_data(pred_date)        
        # 根据预测计算，筛选可以买入的品种
        candidate_list = self.get_candidate_list(pred_date,context=context)
        # candidate_list = ["000702"]
        
        self.lock_list = {}        
        candidate_order_list = {}  
             
        for item in candidate_list:
            trend = item[0]
            instrument = item[1]
            # 剔除没有价格数据的品种
            if not self.has_current_data(pred_date,instrument,mode="instrument"):
                logger.warning("no data for buy:{},ignore".format(instrument))
                continue
            # 代码转化为标准格式
            order_book_id = self.data_source.transfer_futures_order_book_id(instrument,datetime.datetime.strptime(str(pred_date), '%Y%m%d'))
            # 以昨日收盘价格作为当前卖盘价格,注意使用未复权的数据
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
            if self.match_position(instrument) is not None:
                self.lock_list[order_book_id] = order        
                continue             
            # 加入到候选开仓订单
            candidate_order_list[order_book_id] = order
        # 锁定品种保存到存储
        self.trade_entity.add_lock_candidate(self.lock_list,str(pred_date))
        # 开仓候选的订单信息，保存到上下文
        self.candidate_list = candidate_order_list   

        # 根据开仓数量配置，设置开仓列表
        position_max_number = self.strategy.position_max_number
        self.open_list = {} 
        self.close_list = {}     
        # 撤单列表
        self.cancel_list = []        
        # 综合候选品种以及当前已持仓品种，生成维护开仓和平仓列表
        positions = self.get_positions()
        pos_number = len(positions)
        # 全局维护上一候选品种的买卖方向
        self.prev_side = SIDE.BUY
        
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
        
        super().open_auction(context, bar_dict)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        self.time_line = 2
        
        # 如果非实时模式，则需要在相应前等待几秒，以保证先处理外部通知事件
        if self.handle_bar_wait:
            time.sleep(3)
        # 已提交订单检查，包括开仓和平仓
        self.verify_order_closing(context)
        self.verify_order_opening(context)
        
        # 平仓逻辑，止跌平仓       
        self.stop_fall_logic(context,bar_dict=bar_dict) 
        # 平仓逻辑，止盈平仓        
        self.stop_raise_logic(context,bar_dict=bar_dict) 
        # 平仓逻辑，持有品种超期平仓       
        self.expire_day_logic(context,bar_dict=bar_dict)     
        
        # 统一执行买卖挂单处理
        self.order_process(context)
 
        # 首先进行策略，然后进行撮合
        env = Environment.get_instance()
        env.broker.trade_proxy.handler_bar(context.now)       

    def order_process(self,context):
        """挂单流程，先平仓后开仓"""
        
        self.close_order(context)
        self.open_order(context) 
        
    ########################################逻辑判断部分################################################# 

    @get_time
    def open_order(self,context):
        """开仓挂单"""
        
        # 如果持仓品种超过指定数量，则不操作
        position_number = len(self.get_positions())
        position_max_number = self.strategy.position_max_number
        opened_list = self.trade_entity.get_open_list_filled(self.trade_day)
        if opened_list is None:
            return
        # 进行中的订单加上已持仓订单数量，不能大于规定数量阈值 
        if position_number+len(opened_list)>=position_max_number:
            self.logger_info("full pos")
            return
        
        # 轮询候选列表进行买入操作
        for order in self.get_need_open_list():
            # 只对待开仓状态进行挂单
            logger.info("open order loop,order_book_id:{},status:{}".format(order.order_book_id,order.status))
            if order.status!=ORDER_STATUS.PENDING_NEW:
                continue
            # 以昨日收盘价格挂单买入
            order_book_id = order.order_book_id
            price = order.kwargs["price"]
            # 买入数量需要根据当前额度进行计算,还需要兼顾合约乘数
            multiplier = self.data_source.get_contract_info(order_book_id)["multiplier"].astype(float).values[0]
            # 实时计算单个品种购买的额度
            single_value = self.compute_build_quantity()
            quantity = int(single_value/price/multiplier)
            # if price*quantity>30000:
            #     print("nnn")
            # 资金不足时，不进行处理
            if quantity==0:
                logger.warning("volume exceed,order_book_id:{}".format(order_book_id))
                continue
            order = self.submit_order(quantity,order_in=order)
            if order is None:
                logger.warning("open order submit fail,order_book_id:{}".format(order_book_id))
                continue
            # 手动累加，如果购买不成功，后续还需要有流程进行再次购买
            position_number += 1
            if position_number>=position_max_number:
                self.logger_info("full pos in buy list")
                break        
    
    @get_time       
    def close_order(self,context):
        """平仓挂单"""

        # 检查待卖出列表，匹配卖出
        for order in self.get_need_close_list():
            order_book_id = order.order_book_id
            # 如果是当日开仓的，则不处理
            if self.get_today_opened_order(order_book_id, context) is not None:
                continue
            # # 检查可平仓数量是否大于0
            # if order.closable==0:
            #     self.logger_info("closable 0 with:{}".format(order_book_id))
            #     continue
            pos_info = self.get_position(order_book_id)
            # 从待平仓列表中取得相关订单
            close_order = self.get_today_closed_order(order_book_id)
            # 只对待平仓状态进行挂单
            if close_order is not None and close_order.status==ORDER_STATUS.PENDING_NEW:
                # 全部挂单
                amount = pos_info.quantity
                # 挂单卖出
                order = self.submit_order(amount,order_in=close_order)
                if order is None:
                    logger.warning("close order submit fail,order_book_id:{}".format(order_book_id))
                    continue               
                order.close_reason = close_order.kwargs["close_reason"]
                
    @get_time
    def expire_day_logic(self,context,bar_dict=None):
        """持有品种超期平仓逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
        expire_raise_percent = self.strategy.close_opt.expire_raise_percent
        
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_today_closed_order(order_book_id,context=context) is not None:
                # 如果已经在平仓列表中，则不操作
                continue
            # 如果在锁仓列表中，则不处理
            if order_book_id in self.lock_list:
                continue            
            pos_info = self.get_position(order_book_id)
            amount = pos_info.quantity
            now_date = context.now.strftime('%Y%m%d')
            # 通过之前存储的交易信息，查找到对应交易
            trade_date = self.trade_entity.get_trade_date_by_instrument(order_book_id,POSITION_EFFECT.OPEN,context.now)
            if trade_date is None:
                logger.warning("trade not found:{},{}".format(order_book_id,now_date))
                continue
            # 检查是否超期，以决定是否平仓
            dur_days = tradedays(trade_date,now_date)
            if dur_days>=keep_day_number:
                # 检查当前时间段是否可以挂单平仓
                if not self.is_trade_opening(context.now):
                    continue
                # 早盘不进行超期平仓
                if context.now.hour<=12:
                    continue
                # 根据原持仓品种的多空类别决定平仓相关参数
                if pos_info.direction==POSITION_DIRECTION.LONG:
                    side = SIDE.SELL
                    side_flag = 1
                else:
                    side = SIDE.BUY
                    side_flag = -1
                # 根据当前价格挂单
                last_price = self.get_last_price(order_book_id)
                if last_price is None:
                    continue
                # 如果幅度足够，则进行平仓,如果接近收盘，也进行平仓--Cancel
                if (side_flag*(pos_info.avg_price-last_price)/last_price*100>expire_raise_percent) or True:
                    # 生成订单
                    order = self.create_order(position.order_book_id, amount, side,last_price,
                                    position_effect=POSITION_EFFECT.CLOSE,close_reason=SellReason.EXPIRE_DAY.value)  
                    self.append_to_close_list(order)
    
    @get_time                                  
    def stop_fall_logic(self,context,bar_dict=None):
        """止跌卖出逻辑"""
        
        stop_threhold = self.strategy.close_opt.stop_fall_percent
        for position in self.get_ava_positions():
            self.time_inject(begin=True)
            order_book_id = position.order_book_id
            if self.get_today_closed_order(order_book_id,context=context) is not None:
                # 如果已经在平仓列表中，则不操作
                continue
            self.time_inject(code_name="get_today_closed_order")
            pos_info = self.get_position(order_book_id)
            amount = pos_info.quantity
            self.time_inject(code_name="get_position")
            # 根据原持仓品种的多空类别决定平仓相关参数
            if pos_info.direction==POSITION_DIRECTION.LONG:
                side = SIDE.SELL
                side_flag = 1
            else:
                side = SIDE.BUY  
                side_flag = -1       
            last_price = self.get_last_price(order_book_id)
            # open_price = self.trade_entity.get_trade_in_pos(order_book_id)
            self.time_inject(code_name="get_last_price")
            if last_price is None:
                continue
            # 如果下跌幅度(与开仓价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if side_flag*(last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                order = self.create_order(position.order_book_id, amount, side, last_price,
                                          position_effect=POSITION_EFFECT.CLOSE,close_reason=SellReason.STOP_FALL.value)    
                # 当日开仓的不能平仓
                if self.check_close_order_contraint(order):
                    self.append_to_close_list(order)
                self.time_inject(code_name="create_order")
                                           
    @get_time  
    def stop_raise_logic(self,context,bar_dict=None):
        """止盈卖出逻辑"""
        
        stop_threhold = self.strategy.close_opt.stop_raise_percent
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_close_order(order_book_id,context=context) is not None:
                # 如果已经在平仓列表中，则不操作
                continue
            if self.get_today_opened_order(order_book_id,context=context) is not None:
                # 当日开仓，不操作
                continue            
            # 如果在锁仓列表中，则不处理
            if order_book_id in self.lock_list:
                continue             
            pos_info = self.get_position(order_book_id)
            amount = pos_info.quantity
            # 根据原持仓品种的多空类别决定平仓相关参数
            if pos_info.direction==POSITION_DIRECTION.LONG:
                side = SIDE.SELL
                side_flag = 1
            else:
                side = SIDE.BUY  
                side_flag = -1         
                
            last_price = self.get_last_price(order_book_id)
            if last_price is None:
                continue 
            # 如果上涨幅度(与开仓价格比较)超过阈值(百分点)，则以当前价格卖出
            if side_flag*(last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                order = self.create_order(position.order_book_id, amount, side, last_price,
                                          position_effect=POSITION_EFFECT.CLOSE,close_reason=SellReason.STOP_RAISE.value)              
                # 当日开仓的不能平仓
                if self.check_close_order_contraint(order):
                    self.append_to_close_list(order)
    
    def reset_order_list(self,order_item):
        """重置当天的挂单数据"""
        
        if order_item.order_book_id in self.open_list:
            self.open_list[order_item.order_book_id].kwargs["need_resub"] = True   
        else:
            self.close_list[order_item.order_book_id].kwargs["need_resub"] = True   
    
    @get_time            
    def verify_order_closing(self,context):
        """核查平仓订单"""
        
        close_list_active = self.trade_entity.get_close_list_active(str(self.trade_day))
        if close_list_active is None or close_list_active.shape[0]==0:
            return
        for index,close_item in close_list_active.iterrows():
            close_order = self.get_today_closed_order(close_item.order_book_id,context=context)
            price_now = self.get_last_price(close_item.order_book_id)
            # 止盈平仓未成交，以当前价格重新挂单
            if close_order.kwargs["close_reason"]==SellReason.STOP_RAISE.value:
                # 更新挂单列表，后续统一处理
                close_order.kwargs["price"] = price_now    
                self.reset_order_list(close_item)
                # 撤单
                self.cancel_order(close_order)
            # 止损卖出未成交，以当前价格重新挂单 
            elif close_order.kwargs["close_reason"]==SellReason.STOP_FALL.value:
                # 更新挂单列表，后续统一处理
                close_order.kwargs["price"] = price_now    
                self.reset_order_list(close_item)              
                # 撤单
                self.cancel_order(close_order)
            # 超期未成交，以当前价格重新挂单 
            elif close_order.kwargs["close_reason"]==SellReason.EXPIRE_DAY.value:
                # 更新挂单列表，后续统一处理
                close_order.kwargs["price"] = price_now    
                self.reset_order_list(close_item)              
                # 撤单
                self.cancel_order(close_order)
            # 其他情况，以当前价格重新挂单 
            else:
                close_order.kwargs["price"] = price_now
                self.reset_order_list(close_item)              
                # 撤单
                self.cancel_order(close_order)                
    
    @get_time
    def verify_order_opening(self,context):
        """核查开仓订单"""

        try_cnt_limit = self.strategy.open_opt.try_cnt_limit
        pred_open_exceed_rate = self.strategy.open_opt.pred_open_exceed_rate
        pred_open_ignore_rate = self.strategy.open_opt.pred_open_ignore_rate     
        pred_open_wait_rate = self.strategy.open_opt.pred_open_wait_rate      
        env = Environment.get_instance()
        open_list_active = self.trade_entity.get_open_list_active(str(self.trade_day))
        # 已下单未成交的处理
        for index,open_item in open_list_active.iterrows():
            self.logger_info("check active,open_item.order_book_id:{}".format(open_item.order_book_id))
            open_order = self.get_today_opened_order(open_item.order_book_id,context=context)
            price_now = self.get_last_price(open_item.order_book_id)
            if price_now is None:
                continue
            dt = env.calendar_dt
            bar = self.get_prevday_bar(open_item.order_book_id)
            price_last_day = bar['close']               
            side_flag = 1 if open_order.side==SIDE.BUY else -1
            # 如果超出昨日收盘,价格在可变范围内(配置项)，则按照新价格重新挂单
            if (price_now - price_last_day)*side_flag/price_last_day*100<pred_open_ignore_rate \
                and (price_now - price_last_day)*side_flag/price_last_day*100>pred_open_wait_rate:
                self.logger_info("pred_open_ignore_rate ,now:{},price_last_day:{}".format(price_now,price_last_day))
                # 设置重新挂单标志，并发起撤单
                self.reset_order_list(open_order)
                self.cancel_order(open_order)                  
                continue
            # 如果超出太多，则更换品种
            if (price_now - price_last_day)*side_flag/price_last_day*100>pred_open_exceed_rate:
                self.logger_info("pred_buy_exceed_rate ,now:{},price_last_day:{}".format(price_now,price_last_day))          
                # 发起撤单
                self.cancel_order(open_order)    
            # 如果超过等待时间次数，则更换品种
            if open_order.kwargs['try_cnt'] > try_cnt_limit:
                self.logger_info("try_cnt expire:{}".format(open_order.kwargs['try_cnt']))
                # 设置重新挂单标志，并发起撤单
                self.reset_order_list(open_order)
                self.cancel_order(open_order)                  
                continue      
            # 等待,累加等待时间值
            open_order.kwargs['try_cnt']  += 1               
        
        # 已拒绝订单，重新按照现在价格下单
        open_list_reject = self.trade_entity.get_open_list_reject(str(self.trade_day))
        for index,open_item in open_list_reject.iterrows():
            price_now = self.get_last_price(open_item.order_book_id)
            # 修改状态    
            self.update_order_status(open_item,ORDER_STATUS.PENDING_NEW,side=open_item.side, context=context)      
            # 更新报价     
            self.open_list[open_item.order_book_id].kwargs["price"] = price_now       
            
        # 已拒绝平仓订单，重新按照现在价格下单                 
        close_list_reject = self.trade_entity.get_close_list_reject(str(self.trade_day))
        for index,close_item in close_list_reject.iterrows():
            price_now = self.get_last_price(close_item.order_book_id)
            # 修改状态    
            self.update_order_status(close_item,ORDER_STATUS.PENDING_NEW,side=open_item.side, context=context)      
            # 更新报价     
            self.close_list[close_item.order_book_id].kwargs["price"] = price_now  
                                            
    ###############################数据逻辑处理部分########################################  

    def get_position(self,order_book_id):
        """取得指定合约的持仓信息，使用当前策略维护的仓位数据"""
        
        # 通过代理类，取得仿真环境的数据
        sim_position = self.sim_position
        pos = sim_position[sim_position['order_book_id']==order_book_id]
        if pos.shape[0]!=1:
            return None
        dict_data = pos.iloc[0].to_dict()
        pos_obj = DictToObject(dict_data)    
        return pos_obj  

    def match_position(self,instrument):
        """取得指定品种的持仓信息"""
        
        match_pos = None
        for pos in self.get_positions():
            order_book_id = pos.order_book_id
            instrument_item = self.data_source.get_instrument_code_from_contract_code(order_book_id)
            if instrument==instrument_item:
                return pos    
            
        return match_pos 
            
    def get_last_price(self,order_book_id):
        """取得指定标的最近报价信息"""

        env = Environment.get_instance()
        last_dt = env.trading_dt + timedelta(minutes=-1)
        return self.data_source.get_last_price(order_book_id,last_dt)
    
    def compute_build_quantity(self):
        """计算可以买入的单个品种的金额限制"""
        
        position_max_number = self.strategy.position_max_number  
        bond_rate = self.strategy.bond_rate    
        # 总资产 
        total_cash = self.get_cash()
        # 保证金不能超出限制
        bond_limit = total_cash * bond_rate
        positions = self.get_positions()
        current_bond = np.array([pos.margin for pos in positions]).sum()
        # 需要满足的资金=剩余资金-已冻结保证金-当前开仓需要保证金
        remain_bond = total_cash - bond_limit - current_bond
        if remain_bond<0:
            return 0
        # 计算剩余可建仓个数，以及剩余的保证金额度，计算单独一个品种建仓的限制金额
        position_number = len(positions)
        remain_number = position_max_number - position_number
        if remain_number<=0:
            return 0
        # 根据剩余资金额度，以及可持仓品种数量和单只持仓份额，计算当前额度限额
        single_value = total_cash * self.strategy.single_buy_mount_percent/100
        
        self.logger_info("remain_number:{},single_value:{},remain_bond:{}".format(remain_number,single_value,remain_bond))
        return single_value
        
    def get_next_candidate(self):
        """取得下一个候选"""
        
        keys = list(self.candidate_list.keys())
        if len(keys)==0:
            return None
        
        key_can = None
        for key in list(self.candidate_list.keys()):
            order = self.candidate_list[key]
            # 轮流挑选不同的买卖方向品种
            if order.side!=self.prev_side:
                key_can = key
                break
        if key_can is None:
            key_can = list(self.candidate_list.keys())[0]
           
        self.open_list[key_can] = self.candidate_list.pop(key_can)
                
        return self.open_list[key_can]       
               
    def append_to_close_list(self,order):
        """根据指定的订单品种，放入待平仓列表中"""
        
        self.close_list[order.order_book_id] = order 

    def check_close_order_contraint(self,order):
        """检查平仓单是否符合要求"""
        
        if order.order_book_id in self.open_list:
            # 如果当日开仓，则当日不能进行平仓
            return False
        return True
        
                
    def check_close_order_timing(self,instrument,context=None):
        """检查平仓挂单的时间区间是否符合策略要求"""

        minutes = np.array(self.data_source.get_trading_minutes_list(instrument, context.now.date())) 
        now = context.now
        match_time = datetime.datetime(now.year,now.month,now.day,15,3,0)  
        # 先不考虑夜盘
        minutes_match = minutes[np.where(minutes<match_time)[0]]          
        # 全天收盘最后10分钟，符合平仓挂单需求,暂时不考虑夜盘   
        if context.now > minutes_match[-10]:
            return True
        return False
    
    def update_order_status(self,order,status,side=SIDE.BUY,context=None,price=0):
        """修改订单状态"""
        
        if order.order_book_id in self.open_list:
            self.open_list[order.order_book_id]._status = status
        elif order.order_book_id in self.close_list:
            self.close_list[order.order_book_id]._status = status
        else:
            logger.warning("no order in close or open list:{}".format(order.order_book_id))
                
    def is_trade_opening(self,dt):
        """检查当前是否已开盘"""
        
        # 非工作日
        if not is_working_day(dt.date()):
            return False
        # 出于数据获取考虑，9点01才算开盘
        open_timing = datetime.datetime(dt.year,dt.month,dt.day,9,0,0)
        if dt>open_timing:
            return True
        return False

    def is_approch_closing(self):
        """检查当前时间是否接近收盘"""
        
        # 下午2点半以后接近收盘
        env = Environment.get_instance()
        dt = env.trading_dt
        closing_timing = datetime.datetime(dt.year,dt.month,dt.day,14,30,0)
        if dt>closing_timing:
            return True
        return False
       
    def get_close_order(self,order_book_id,context=None):
        if order_book_id in self.close_list:
            return self.close_list[order_book_id]
        return None
    
    def get_open_order(self,order_book_id,context=None):
        if order_book_id in self.open_list:
            return self.open_list[order_book_id]
        return None    
    
    def get_previous_trading_date(self,trade_date):
        return get_tradedays_dur(str(trade_date),-1)
    
    def get_last_or_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天或上一交易时间段的数据"""

        env = Environment.get_instance()
        self.time_inject(code_name="data_proxy.instrument")
        # 如果还没有开盘，取上一交易日数据，否则取上一交易时间段数据
        if self.is_trade_opening(env.trading_dt):
            last_dt = env.trading_dt + timedelta(minutes=-1)
            bar = self.data_source.get_bar(order_book_id,last_dt,"1m")
        else:
            prev_day = get_tradedays_dur(self.context.now, -1)
            bar = self.data_source.get_bar(order_book_id,prev_day,"1d")            
        self.time_inject(code_name="get_bar")
        if bar is not None and not np.isnan(bar['open']):
            return bar       
        return None

    def get_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天的数据"""

        env = Environment.get_instance()
        prev_date = get_tradedays_dur(env.trading_dt,-1)
        bar = self.data_source.get_bar(order_book_id,prev_date,"1d")
        if bar is not None and not np.isnan(bar['open']):
            return bar       
        return None      
    
    def can_submit_order(self,order_book_id):
        """重载父类方法，检查是否可以提交订单"""
        
        bar = self.get_last_or_prevday_bar(order_book_id)
        if bar is not None and not np.isnan(bar['open']):
            return True
        return False
        
    def create_order(self,id_or_ins, amount, side,price, position_effect=None,close_reason=None):
        """代理api的订单创建方法"""
        
        order_book_id = id_or_ins
        multiplier = self.data_source.get_contract_info(order_book_id)["multiplier"].astype(float).values[0]
        # 添加交易所编码
        exchange_code = self.data_source.get_exchange_from_instrument(order_book_id[:-4])
        
        style = cal_style(price, None)
        order = Order.__from_create__(
            order_book_id=order_book_id,
            quantity=amount,
            side=side,
            style=style,
            position_effect=position_effect,
            # 自定义属性
            price=price,
            multiplier=multiplier,
            try_cnt=0, 
            close_reason=close_reason,  
            need_resub=False, 
            exchange_id=exchange_code      
        )   
        order.set_frozen_cash(0)    
              
        return order
    
    def get_today_opened_order(self,order_book_id,context=None):
        """取得当日开仓列表"""
        
        order_rtn = None
        if order_book_id in self.open_list:
            order = self.open_list[order_book_id]
            if order.position_effect==POSITION_EFFECT.OPEN:
                order_rtn = order
        # 如果不在当日开仓列表中，再从存储中取，并懒加载
        if order_rtn is None:
            order = self.trade_entity.get_open_order_active(context.now.date().strftime("%Y%m%d"),order_book_id)
            if order is not None:
                order_rtn = self.create_order(order_book_id, order['quantity'], order['side'], order['price'], order['position_effect'])
                self.open_list[order_book_id] = order_rtn
        return order_rtn   

    def get_today_closed_order(self,order_book_id,context=None):
        """取得当日平仓列表"""
        
        if order_book_id in self.close_list:
            order = self.close_list[order_book_id]
            if order.position_effect==POSITION_EFFECT.CLOSE:
                return order
        return None 
       
    def check_instrument(self,instrument,date):
        """检查当前交易日指定品种是否可以交易"""
        
        return self.data_source.check_instrument(self,instrument,date)
    
    def create_trade_entity(self,save_path=None,log_save_path=None):
        return FuturesTradeEntity(save_path=save_path,log_save_path=log_save_path)
        
    def get_context_dataset(self):
        return self.context.ml_context.dataset
    
    def get_candidate_list(self,pred_date,context=None):
        """取得候选列表，重载父类方法"""
        
        candidate_list = context.ml_context.filter_futures_buy_candidate(pred_date)
        # candidate_list = ["000702"]
        
        # 检查是否缺失历史数据,如果缺失则剔除
        filter_list = []
        for item in candidate_list:
            symbol = item[1]
            # 只处理当日有开盘数据的
            if self.has_current_data(pred_date,symbol,mode="instrument"):        
                filter_list.append(item)
        return filter_list

    def get_need_open_list(self,context=None):
        """取得待开仓候选列表"""
        
        result = []
        def check_list(target_list):
            for key in target_list:
                order = target_list[key]
                if order.position_effect==POSITION_EFFECT.OPEN:
                    result.append(order) 
                    
        check_list(self.open_list)
        return result            

    def get_need_close_list(self,context=None):
        """取得待平仓候选列表"""
        
        result = []
        def check_list(target_list):
            for key in target_list:
                order = target_list[key]
                if order.position_effect==POSITION_EFFECT.CLOSE:
                    result.append(order) 
                    
        check_list(self.close_list)
        return result      
    
    
    def get_ava_positions(self):
        """取得当日可以买卖的持仓信息"""
        
        positions = []
        # 通过代理类，取得仿真环境的数据
        trade_proxy = Environment.get_instance().broker.trade_proxy
        pred_date = int(self.context.now.strftime('%Y%m%d'))
        for pos in trade_proxy.get_positions():
            symbol = transfer_instrument(pos.order_book_id)
            # 如果当日无数据，则忽略
            if self.has_current_data(pred_date,symbol):
                positions.append(pos)
        return positions 
        
    def has_current_data(self,day,code,mode="contract"):
        """当日是否开盘交易"""
        
        date_trading_mappings = self.date_trading_mappings
        day_date = datetime.datetime.strptime(str(day), '%Y%m%d').date()
        if mode=="instrument":
            # 品种模式查询，需要先根据品种取得合约代码再查询
            symbol = self.data_source.get_main_contract_name(code,str(day))
            data = date_trading_mappings[(date_trading_mappings['date']==day_date)&(date_trading_mappings['contract']==symbol)]
        else:
            # 合约模式查询，直接查询
            data = date_trading_mappings[(date_trading_mappings['date']==day_date)&(date_trading_mappings['contract']==code)]
            
        if data.shape[0]>0:
            # 有些数据交易量很小，则不计入
            if float(data['volume'].values[0])<=0:
                return False
            return True
        
        return False        
    
    ############################事件注册部分######################################
    
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        account.get_positions()
        self.logger_debug("on_trade_handler in,order:{},trade:{}".format(order,trade))
        # 保存成单交易对象
        self.trade_entity.add_trade(trade,multiplier=order.kwargs['multiplier'])
        # 修改当日仓位列表中的状态为已成交
        self.update_order_status(order,ORDER_STATUS.FILLED,side=order.side, context=self.context)     
        # 维护仓位数据
        self.apply_trade_pos(trade)
        # 平仓一个，就可以再开仓一个。从候选列表中挑选新的品种，放入待开仓列表中
        if order.position_effect==POSITION_EFFECT.CLOSE:
            self.get_next_candidate()              
    
    def apply_trade_pos(self,trade):
        """维护仓位数据"""
        
        sim_position = self.sim_position
        if trade.position_effect==POSITION_EFFECT.OPEN:
            item = pd.DataFrame(np.array([[trade.order_book_id,trade.last_quantity,trade.side,
                            trade.position_direction,trade.last_price,trade.datetime,trade.order_id]]),columns=POS_COLUMNS)
            # 开仓时添加到模拟仓位
            remain_pos = sim_position[sim_position['order_book_id']==trade.order_book_id]
            # 如果原仓位有数据，则累加合并
            if remain_pos.shape[0]==0:
                sim_position = pd.concat([sim_position,item])
            else:
                remain_pos['quantity'] = remain_pos['quantity'].values[0] + trade.last_quantity
                # 计算综合成本
                remain_pos['avg_price'] = (remain_pos['avg_price'].values[0]*remain_pos['quantity'].values[0] + 
                        trade.last_quantity*trade.last_price)/(remain_pos['quantity'].values[0]+trade.last_quantity)
                sim_position[sim_position['order_book_id']==trade.order_book_id] = remain_pos
        else:
            # 平仓时消除原仓位数据
            remain_pos = sim_position[sim_position['order_book_id']==trade.order_book_id]
            remain_quantity = remain_pos['quantity'].values[0]
            if remain_quantity==trade.last_quantity:
                sim_position = sim_position[sim_position['order_book_id']!=trade.order_book_id]
            else:
                remain_pos['quantity'] = remain_pos['quantity'].values[0] - trade.last_quantity
                sim_position[sim_position['order_book_id']==trade.order_book_id] = remain_pos
            
        self.sim_position = sim_position
               
    def on_order_handler(self,context, event):
        order = event.order
        self.logger_info("order handler,event_type::{},order:{}".format(event.event_type,event.order))
        # 未接单事件
        if order.status==ORDER_STATUS.PENDING_NEW:
            self.logger_info("order PENDING_NEW:{},trade_date:{}".format(order.order_book_id,self.trade_day))            
            # 更新跟踪变量状态
            self.update_order_status(order,ORDER_STATUS.PENDING_NEW,side=order.side, context=self.context)   
            # 更新存储状态               
            self.trade_entity.update_status(order)
            return            
        # 已接单事件
        if order.status==ORDER_STATUS.ACTIVE:
            self.logger_info("order active:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            # 订单已接受，设置第二订单号          
            if order.order_book_id in self.open_list:
                self.open_list[order.order_book_id].set_secondary_order_id(order.secondary_order_id) 
            elif order.order_book_id in self.close_list:
                self.close_list[order.order_book_id].set_secondary_order_id(order.secondary_order_id)                 
            # 更新跟踪变量状态
            self.update_order_status(order,ORDER_STATUS.ACTIVE,side=order.side, context=self.context)   
            # 更新存储状态               
            self.trade_entity.add_or_update_order(order,str(self.trade_day))  
            return        
        # 如果订单被拒绝，则忽略,仍然保持新单状态，后续会继续下单
        if order.status==ORDER_STATUS.REJECTED:
            self.logger_info("order reject:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            # 直接修改状态
            self.trade_entity.update_status(order)
            # 更新跟踪变量状态
            self.update_order_status(order,ORDER_STATUS.REJECTED,side=order.side, context=self.context)               
            return
        # 已撤单事件
        if order.status==ORDER_STATUS.CANCELLED:
            self.logger_info("order CANCELLED:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            # 这里需要修改状态为已撤单
            self.update_order_status(order,ORDER_STATUS.CANCELLED,side=order.side, context=self.context)      
            self.trade_entity.add_or_update_order(order,str(self.trade_day))    
            
            if order.order_book_id in self.open_list:
                # 开仓撤单，检查相关标志，决定重新挂单还是换品种
                if self.open_list[order.order_book_id].kwargs["need_resub"]:
                    self.logger_info("need resub order:{}".format(order))
                    # 如果具备重新报单标志，则以最新价格重新生成订单
                    price_now = self.get_last_price(order.order_book_id)
                    # 创建新订单对象并重置原数据
                    order_resub = self.create_order(order.order_book_id, order.quantity, order.side, price_now,position_effect=POSITION_EFFECT.OPEN)
                    self.open_list[order.order_book_id] = order_resub
                    self.logger_debug("resub open list set end:{}".format(order.order_book_id))
                else:
                    # 如果不需要重新报单，则换其他品种
                    self.get_next_candidate()  
            else:
                # 平仓撤单，按照当前时间段的价格重新挂单        
                self.logger_info("need resub close order:{}".format(order))
                # 如果具备重新报单标志，则以最新价格重新生成订单
                price_now = self.get_last_price(order.order_book_id)
                # 创建新订单对象并重置原数据
                order_resub = self.create_order(order.order_book_id, order.quantity, order.side, price_now,close_reason=order.kwargs['close_reason'],position_effect=POSITION_EFFECT.CLOSE)
                self.close_list[order.order_book_id] = order_resub
                self.logger_debug("resub closelist set end:{}".format(order.order_book_id))            
                    