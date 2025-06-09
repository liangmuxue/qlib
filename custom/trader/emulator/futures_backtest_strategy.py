import time
import os
import datetime
from rqalpha.apis import *
import rqalpha
from rqalpha.const import SIDE,ORDER_STATUS
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_furtures_order_book_id,transfer_instrument
from trader.rqalpha.futures_trade_entity import FuturesTradeEntity
from trader.emulator.sim_strategy import SimStrategy
from trader.utils.date_util import tradedays,get_tradedays_dur

from data_extract.his_data_extractor import PeriodType
from cus_utils.data_filter import get_topN_dict
from cus_utils.log_util import AppLogger
from gunicorn import instrument
logger = AppLogger()

class FurBacktestStrategy(SimStrategy):
    """回测交易策略，分钟级别，继承回测基类"""
    
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
            order_book_id = self.data_source.transfer_furtures_order_book_id(instrument,datetime.datetime.strptime(str(pred_date), '%Y%m%d'))
            # 以昨日收盘价格作为当前卖盘价格,注意使用未复权的数据
            h_bar = history_bars(order_book_id,1,"1d",fields="close",adjust_type="none")
            if h_bar is None:
                logger.warning("history bar None:{},date:{}".format(order_book_id,context.now))
                continue
            price = h_bar[0]
            # # fake
            # if order_book_id.startswith("000410"):
            #     price = 7.17
            
            # 根据多空标志决定买卖方向
            if trend==1:
                side = SIDE.BUY
            else:
                side = SIDE.SELL
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）,仓位类型为开仓
            order = self.create_order(order_book_id, 0, side,price,position_effect=POSITION_EFFECT.OPEN)
            # 对于开仓候选品种，如果已持仓当前品种，则进行锁仓
            if self.get_postion_by_order_book_id(order_book_id) is not None:
                self.lock_list[order_book_id] = order        
                continue               
            # 加入到候选开仓订单
            candidate_order_list[order_book_id] = order
        # 开仓候选的订单信息，保存到上下文
        self.candidate_list = candidate_order_list   

        # 根据开仓数量配置，设置开仓列表
        position_max_number = self.strategy.position_max_number
        self.open_list = {} 
        self.close_list = {}     
        # 综合候选品种以及当前已持仓品种，生成维护开仓和平仓列表
        positions = self.get_positions()
        pos_number = len(positions)
        # 全局维护上一候选品种的买卖方向
        self.prev_side = SIDE.BUY
        
        # 循环从候选中取数，直到取完或者超出开仓数量为止
        while pos_number<position_max_number or len(self.candidate_list)==0:
            order_book_id = order.order_book_id
            # 依次从候选中选取对应品种并放入开仓列表
            candidate_order = self.get_next_candidate()
            if candidate_order is not None:
                pos_number += 1
                
        # # 生成平仓列表
        # for pos in positions:
        #     order_book_id = pos.order_book_id
        #     if order_book_id in self.lock_list:
        #         continue
        #     price = self.get_last_or_prevday_bar(order_book_id)['close']
        #     if pos.direction==POSITION_DIRECTION.LONG:
        #         side = SIDE.SELL
        #     else:
        #         side = SIDE.BUY            
        #     order = self.create_order(order_book_id, 0, side,price,position_effect=POSITION_EFFECT.CLOSE)
        #     self.close_list[order_book_id] = order
            
        # 撤单列表
        self.cancel_list = []
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        super().open_auction(context, bar_dict)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        env.broker.trade_proxy.handler_bar(context.now)
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
        

    def order_process(self,context):
        """挂单流程，先平仓后开仓"""
        
        self.close_order(context)
        self.open_order(context) 
        
    ########################################逻辑判断部分################################################# 

    def compute_build_quantity(self):
        """计算可以买入的单个品种的金额限制"""
        
        position_max_number = self.strategy.position_max_number  
        bond_rate = self.strategy.bond_rate    
        # 总资产 
        total_value = self.get_total_value()
        # 保证金不能超出限制
        bond_limit = total_value * bond_rate
        current_bond = np.array([pos.margin for pos in self.get_positions()]).sum()
        remain_bond = bond_limit - current_bond
        if remain_bond<0:
            return 0
        # 计算剩余可建仓个数，以及剩余的保证金额度，计算单独一个品种建仓的限制金额
        position_number = len(self.get_positions())
        remain_number = position_max_number - position_number
        if remain_number<=0:
            return 0
        single_value = remain_bond/bond_rate/remain_number
        
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
            self.logger_debug("open order loop,order_book_id:{},status:{}".format(order.order_book_id,order.status))
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
            
    def close_order(self,context):
        """平仓挂单"""

        # 检查待卖出列表，匹配卖出
        for order in self.get_need_close_list():
            order_book_id = order.order_book_id
            if order_book_id=='RR2206':
                print("ggg")
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

    def expire_day_logic(self,context,bar_dict=None):
        """持有品种超期平仓逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
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
                # 根据开盘价格挂单
                price = self.get_last_price(order_book_id)
                # 根据原持仓品种的多空类别决定平仓相关参数
                if pos_info.direction==POSITION_DIRECTION.LONG:
                    side = SIDE.SELL
                else:
                    side = SIDE.BUY
                # 生成订单
                order = self.create_order(position.order_book_id, amount, side,price,
                                position_effect=POSITION_EFFECT.CLOSE,close_reason=SellReason.EXPIRE_DAY.value)  
                self.append_to_close_list(order)
                                         
    def stop_fall_logic(self,context,bar_dict=None):
        """止跌卖出逻辑"""
        
        stop_threhold = self.strategy.close_opt.stop_fall_percent
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_today_closed_order(order_book_id,context=context) is not None:
                # 如果已经在平仓列表中，则不操作
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
            # 如果下跌幅度(与开仓价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if side_flag*(pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                price = self.get_last_price(order_book_id)
                order = self.create_order(position.order_book_id, amount, side, price,
                                          position_effect=POSITION_EFFECT.CLOSE,close_reason=SellReason.STOP_FALL.value)    
                # 当日开仓的不能平仓
                if self.check_close_order_contraint(order):
                    self.append_to_close_list(order)
                                           
        
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
            pos_info = self.get_position(order_book_id)
            amount = pos_info.quantity
            # 根据原持仓品种的多空类别决定平仓相关参数
            if pos_info.direction==POSITION_DIRECTION.LONG:
                side = SIDE.SELL
                side_flag = 1
            else:
                side = SIDE.BUY  
                side_flag = -1               
            # 如果上跌幅度(与开仓价格比较)超过阈值(百分点)，则以当前价格卖出
            if side_flag*(pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                price = self.get_last_price(order_book_id)
                order = self.create_order(position.order_book_id, amount, side, price,
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
            if close_order.kwargs["close_reason"]==SellReason.STOP_FALL.value:
                # 更新挂单列表，后续统一处理
                close_order.kwargs["price"] = price_now    
                self.reset_order_list(close_item)              
                # 撤单
                self.cancel_order(close_order)
            # 超期未成交，以当前价格重新挂单 
            if close_order.kwargs["close_reason"]==SellReason.EXPIRE_DAY.value:
                # 更新挂单列表，后续统一处理
                close_order.kwargs["price"] = price_now    
                self.reset_order_list(close_item)              
                # 撤单
                self.cancel_order(close_order)

    def verify_order_opening(self,context):
        """核查开仓订单"""

        env = Environment.get_instance()
        open_list_active = self.trade_entity.get_open_list_active(str(self.trade_day))
        # 已下单未成交的处理
        for index,open_item in open_list_active.iterrows():
            self.logger_info("check active,open_item.order_book_id:{}".format(open_item.order_book_id))
            sys_order = self.trade_entity.get_sys_order(open_item.order_book_id)
            open_order = self.get_today_opened_order(open_item.order_book_id,context=context)
            cur_snapshot = self.get_current_snapshot(open_item.order_book_id)
            if cur_snapshot is None:
                logger.warning("cur_snapshot None in verify_order_buying:{}".format(open_item))
                continue            
            price_now = cur_snapshot.last
            dt = env.calendar_dt
            prev_day = get_tradedays_dur(dt,-1)
            # 使用结算价作为上一日的价格
            h_bar = env.data_proxy.history_bars(open_item.order_book_id,1,"1d","settle",prev_day)
            price_last_day = h_bar[0]               
            pred_open_exceed_rate = self.strategy.open_opt.pred_open_exceed_rate
            pred_open_ignore_rate = self.strategy.open_opt.pred_open_ignore_rate
            side_flag = 1 if open_order.side==SIDE.BUY else -1
            # 如果超出昨日收盘,单在可变范围内(配置项)，则按照新价格重新挂单
            if (price_now - price_last_day)*side_flag/price_last_day*100<pred_open_ignore_rate:
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
            # 等待            
        
        # 已拒绝订单，重新按照现在价格下单
        open_list_reject = self.trade_entity.get_open_list_reject(str(self.trade_day))
        for index,open_item in open_list_reject.iterrows():
            try:
                cur_snapshot = self.get_current_snapshot(open_item.order_book_id)
                price_now = cur_snapshot.last
            except Exception as e:
                logger.error("cur_snapshot err:{}".format(e))
                continue
            # 修改状态    
            self.update_order_status(open_item,ORDER_STATUS.PENDING_NEW,side=open_item.side, context=context)      
            # 更新报价     
            self.open_list[open_item.order_book_id].kwargs["price"] = price_now                         
                                
    ###############################数据逻辑处理部分########################################  

    def update_order_status(self,order,status,side=SIDE.BUY,context=None,price=0):
        """修改订单状态"""
        
        if order.order_book_id in self.open_list:
            self.open_list[order.order_book_id]._status = status
        else:
            self.close_list[order.order_book_id]._status = status
                
    def is_trade_opening(self,dt):
        """检查当前是否已开盘"""
        
        # 出于数据获取考虑，9点01才算开盘
        open_timing = datetime.datetime(dt.year,dt.month,dt.day,9,0,0)
        if dt>open_timing:
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
    
    def get_last_or_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天或上一交易时间段的数据"""

        env = Environment.get_instance()
        instrument = env.data_proxy.instrument(order_book_id)
        # 如果还没有开盘，取上一交易日数据，否则取上一交易时间段数据
        if self.is_trade_opening(env.trading_dt):
            bar = self.data_source.get_bar(instrument,env.trading_dt,"1m")
        else:
            bar = self.data_source.get_bar(instrument,env.trading_dt,"1d")
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
        
        order_book_id = assure_order_book_id(id_or_ins)
        multiplier = self.data_source.get_contract_info(order_book_id)["multiplier"].astype(float).values[0]
        # 添加交易所编码
        exchange_code = self.data_source.get_exchange_from_instrument(order_book_id)
        
        style = cal_style(price, None)
        if side==SIDE.BUY:
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
                close_reason=None,  
                need_resub=False, 
                exchange_id=exchange_code      
            )   
        else:
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
        
        if order_book_id in self.open_list:
            order = self.open_list[order_book_id]
            if order.position_effect==POSITION_EFFECT.OPEN:
                return order
        return None   

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
            symbol = self.data_source.get_main_contract_name(code,day_date)
            data = date_trading_mappings[(date_trading_mappings['date']==day_date)&(date_trading_mappings['contract']==symbol)]
            if data.shape[0]>0:
                return True
            return False
        else:
            # 合约模式查询，直接查询
            data = date_trading_mappings[(date_trading_mappings['date']==day_date)&(date_trading_mappings['contract']==code)]
            if data.shape[0]>0:
                return True
            return False           
        return False
    
    def get_last_price(self,order_book_id):
        """重载，这里为取得最近分钟行情"""
        
        env = Environment.get_instance()
        # market_price = env.get_last_price(order_book_id)
        # instrument = env.data_proxy.instruments(order_book_id)
        # market_price = self.get_current_snapshot(instrument)
        current_snapshot = self.get_current_snapshot(order_book_id)
        if current_snapshot is None:
            return None
        
        return current_snapshot.last 
    
    ############################事件注册部分######################################
    
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        self.logger_debug("on_trade_handler in,order:{}".format(order))
        # 保存成单交易对象
        self.trade_entity.add_trade(trade,multiplier=order.kwargs['multiplier'])
        # 修改当日仓位列表中的状态为已成交
        self.update_order_status(order,ORDER_STATUS.FILLED,side=order.side, context=self.context)     
        # 平仓一个，就可以再开仓一个。从候选列表中挑选新的品种，放入待开仓列表中
        if order.position_effect==POSITION_EFFECT.CLOSE:
            self.get_next_candidate()              
    
    def on_order_handler(self,context, event):
        order = event.order
        self.logger_info("order handler,order:{}".format(order))
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
            self.trade_entity.add_or_update_order(order,str(self.trade_day))  
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
                    cur_snapshot = self.get_current_snapshot(order.order_book_id)
                    price_now = cur_snapshot.last
                    # 创建新订单对象并重置原数据
                    order_resub = self.create_order(order.order_book_id, order.quantity, order.side, price_now)
                    self.open_list[order.order_book_id] = order_resub
                    self.logger_debug("resub open list set end:{}".format(order.order_book_id))
                else:
                    # 如果不需要重新报单，则换其他品种
                    self.get_next_candidate()  
            else:
                # 平仓撤单，按照当前时间段的价格重新挂单        
                self.logger_info("need resub close order:{}".format(order))
                # 如果具备重新报单标志，则以最新价格重新生成订单
                cur_snapshot = self.get_current_snapshot(order.order_book_id)
                price_now = cur_snapshot.last
                # 创建新订单对象并重置原数据
                order_resub = self.create_order(order.order_book_id, order.quantity, order.side, price_now,close_reason=order.kwargs['close_reason'])
                self.close_list[order.order_book_id] = order_resub
                self.logger_debug("resub closelist set end:{}".format(order.order_book_id))            
                    