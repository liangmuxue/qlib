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
        # 根据预测计算，筛选可以买入的股票
        candidate_list = self.get_candidate_list(pred_date,context=context)
        # candidate_list = [""000702"]
        # candidate_list = []
        candidate_list_buy = {}
        sell_list = {}
        # 从文件中加载的未成单的订单记录，维护到上下文
        buy_orders = self.trade_entity.get_buy_list_active(str(self.trade_day))
        for index,row in buy_orders.iterrows():   
            # 根据多空标志，决定是下单类型是买还是卖 
            if row["long_short"]==1:
                side = SIDE.BUY
            else:
                side = SIDE.SELL
            order = self.create_order(row["order_book_id"], row["quantity"], side, row["long_short"],row["price"])
            candidate_list_buy[row["order_book_id"]] = order
        sell_orders = self.trade_entity.get_sell_list_active(str(self.trade_day))
        for index,row in sell_orders.iterrows():  
            if row["long_short"]==1:
                side = SIDE.SELL
            else:
                side = SIDE.BUY                        
            order = self.create_order(row["order_book_id"], row["quantity"], side, None,row["price"])
            order.set_frozen_cash(0)
            order._status = row["status"]            
            sell_list[row["order_book_id"]] = order
                                
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
            buy_price = h_bar[0]
            # # fake
            # if order_book_id.startswith("000410"):
            #     buy_price = 7.17
            if order_book_id in candidate_list_buy:
                continue
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
            order = self.create_order(order_book_id, 0, SIDE.BUY, trend,buy_price)
            candidate_list_buy[order_book_id] = order
        # 保存到上下文
        self.candidate_list_buy = candidate_list_buy         
        # 根据买单数量配置，设置买单列表
        position_max_number = self.strategy.position_max_number
        # 从买入候选列表中，根据配置取得待买入列表
        self.buy_list = get_topN_dict(candidate_list_buy,position_max_number)
        self.new_buy_list = {}
        
        # 查看持仓，根据预测模型计算,逐一核对是否需要卖出
        for position in self.get_positions():
            if position.order_book_id in sell_list:
                continue            
            symbol = transfer_instrument(position.order_book_id)
            # 如果当日没有数据（未开盘），则忽略
            if not self.has_current_data(pred_date,symbol):
                logger.warning("no data for sell:{},ignore".format(symbol))
                continue
            # Chang by lmx,暂时不检查负向指标
            # flag = context.ml_context.measure_pos(pred_date,int(instrument))
            # if flag:
            #     pos_info = self.get_position(position.order_book_id)
            #     amount = pos_info.quantity
            #     # 以昨日收盘价格卖出
            #     h_bar = history_bars(position.order_book_id,1,"1d",fields="close",adjust_type="none")
            #     if h_bar is None:
            #         continue
            #     sell_price = h_bar[0,0]                
            #     # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
            #     order = self.create_order(position.order_book_id, amount, SIDE.SELL, sell_price,sell_reason=SellReason.PRED.value)             
            #     sell_list[position.order_book_id] = order
        # 卖单保存到上下文    
        self.sell_list = sell_list
        # 撤单列表
        self.cancel_list = []
        self.buy_try_cnt = 0
        # 在每日开盘前计算单只股票购买的额度
        self.single_value = self.day_compute_quantity()
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        super().open_auction(context, bar_dict)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        env.broker.trade_proxy.handler_bar(context.now)
        super().handle_bar(context,bar_dict)

    def create_order(self,id_or_ins, amount, side, price, position_effect=None,sell_reason=None):
        """代理api的订单创建方法"""
        
        order_book_id = assure_order_book_id(id_or_ins)
        style = cal_style(price, None)
        if side==SIDE.BUY:
            order = Order.__from_create__(
                order_book_id=order_book_id,
                quantity=amount,
                side=side,
                style=style,
                position_effect=position_effect,
                # 自定义属性
                buy_price=price,
                try_cnt=0, 
                sell_reason=None,  
                need_resub=False        
            )   
        else:
            order = Order.__from_create__(
                order_book_id=order_book_id,
                quantity=amount,
                side=side,
                style=style,
                position_effect=position_effect,
                # 自定义属性
                sell_price=price,
                try_cnt=0, 
                sell_reason=sell_reason,  
                need_resub=False         
            )                    
        return order

    def buy_order(self,context):
        """买盘挂单"""
        
        # 如果持仓股票超过指定数量，则不操作
        position_number = len(self.get_positions())
        position_max_number = self.strategy.position_max_number
        active_buy_list = self.trade_entity.get_buy_list_active(self.trade_day)
        # 进行中的订单加上已持仓订单数量，不能大于规定数量阈值 
        if position_number+len(active_buy_list)>=position_max_number:
            self.logger_info("full pos")
            return
        
        # 轮询候选列表进行买入操作
        for buy_order_id in self.buy_list:
            buy_order = self.buy_list[buy_order_id]
            # 只对待买入状态进行挂单
            self.logger_debug("buy order loop,order_book_id:{},status:{}".format(buy_order.order_book_id,buy_order.status))
            if buy_order.status!=ORDER_STATUS.PENDING_NEW:
                continue
            # 以昨日收盘价格挂单买入
            order_book_id = buy_order.order_book_id
            price = buy_order.kwargs["buy_price"]
            # 买入数量需要根据当前额度进行计算,还需要兼顾合约乘数
            multiplier = self.data_source.get_contract_info(order_book_id)["multiplier"].astype(float).values[0]
            try:
                quantity = int(self.single_value/price/multiplier)
            except Exception as e:
                print("eee")
            # if price*quantity>30000:
            #     print("nnn")
            # 资金不足时，不进行处理
            if quantity==0:
                logger.warning("volume exceed,order_book_id:{}".format(order_book_id))
                continue
            order = self.submit_order(quantity,order_in=buy_order)
            if order is None:
                logger.warning("order submit fail,order_book_id:{}".format(order_book_id))
                continue
            # 手动累加，如果购买不成功，后续还需要有流程进行再次购买
            position_number += 1
            if position_number>=position_max_number:
                self.logger_info("full pos in buy list")
                break        
            
    def sell_order(self,context):
        """卖盘挂单"""

        # 检查待卖出列表，匹配卖出
        for position in self.get_positions():
            order_book_id = position.order_book_id
            # 如果是当日买入的，则不处理
            if self.get_buy_order(order_book_id, context) is not None:
                continue
            # 检查可平仓数量是否大于0
            if position.closable==0:
                self.logger_info("closable 0 with:{}".format(order_book_id))
                continue
            pos_info = self.get_position(order_book_id)
            if order_book_id in self.sell_list:
                # 取得预卖出订单
                sell_order = self.sell_list[order_book_id]
                # 只对待卖出状态进行挂单
                if sell_order.status!=ORDER_STATUS.PENDING_NEW:
                    continue                
                # 全部卖出
                sell_amount = pos_info.quantity
                # 挂单卖出
                order = self.submit_order(sell_amount,order_in=sell_order)
                if order is None:
                    logger.warning("order submit fail,order_book_id:{}".format(order_book_id))
                    continue               
                order.sell_reason = sell_order.kwargs["sell_reason"]


    #####################################逻辑判断部分#################################################                                 
    def expire_day_logic(self,context,bar_dict=None):
        """持有股票超期卖出逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            now_date = context.now.strftime('%Y%m%d')
            # 通过之前存储的交易信息，查找到对应交易
            trade_date = self.trade_entity.get_trade_date_by_instrument(order_book_id,SIDE.BUY,context.now)
            if trade_date is None:
                logger.warning("trade not found:{},{}".format(order_book_id,now_date))
                continue
            # 检查是否超期，以决定是否卖出
            dur_days = tradedays(trade_date,now_date)
            if dur_days>keep_day_number:
                if dur_days>4:
                    logger.warning("dur_days exceed,trade_date:{},dur_day:{},order_book_id:{}".format(trade_date,dur_days,order_book_id))
                self.logger_info("expire,trade_date:{},and dur_day:{},order_book_id:{}".format(trade_date,dur_days,order_book_id))
                sell_price = self.get_last_price(order_book_id)
                # 多空单参数
                order = self.create_order(position.order_book_id, sell_amount, SIDE.SELL,sell_price,sell_reason=SellReason.EXPIRE_DAY.value)                
                self.sell_list[position.order_book_id] = order    
                        
    def stop_fall_logic(self,context,bar_dict=None):
        """止跌卖出逻辑"""
        
        stop_threhold = self.strategy.sell_opt.stop_fall_percent
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                sell_price = self.get_last_price(order_book_id)
                order = self.create_order(position.order_book_id, sell_amount, SIDE.SELL, sell_price,sell_reason=SellReason.STOP_FALL.value)                
                self.sell_list[position.order_book_id] = order                
        
    def stop_raise_logic(self,context,bar_dict=None):
        """止盈卖出逻辑"""
        
        stop_threhold = self.strategy.sell_opt.stop_raise_percent
        
        for position in self.get_ava_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            if self.get_buy_order(order_book_id,context=context) is not None:
                # 当日买入，不操作
                continue            
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                sell_price = self.get_last_price(order_book_id)
                order = self.create_order(position.order_book_id, sell_amount, SIDE.SELL, sell_price,sell_reason=SellReason.STOP_RAISE.value)              
                self.sell_list[position.order_book_id] = order  
                
    def verify_order_selling(self,context):
        """核查卖出订单"""
        
        sell_list_active = self.trade_entity.get_sell_list_active(str(self.trade_day))
        if sell_list_active.shape[0]==0:
            return
        for index,sell_item in sell_list_active.iterrows():
            sell_order = self.get_sell_order(sell_item.order_book_id,context=context)
            price_now = self.get_last_price(sell_item.order_book_id)
            # 止盈卖出未成交，以当前价格重新挂单
            if sell_order.kwargs["sell_reason"]==SellReason.STOP_RAISE.value:
                # 更新挂单列表，后续统一处理
                sell_order.kwargs["sell_price"] = price_now    
                self.sell_list[sell_item.order_book_id].kwargs["need_resub"] = True                 
                # 撤单
                self.cancel_order(sell_order)
            # 止损卖出未成交，以当前价格重新挂单 
            if sell_order.kwargs["sell_reason"]==SellReason.STOP_FALL.value:
                # 更新挂单列表，后续统一处理
                sell_order.kwargs["sell_price"] = price_now    
                self.sell_list[sell_item.order_book_id].kwargs["need_resub"] = True                 
                # 撤单
                self.cancel_order(sell_order)
            # 超期未成交，以当前价格重新挂单 
            if sell_order.kwargs["sell_reason"]==SellReason.EXPIRE_DAY.value:
                # 更新挂单列表，后续统一处理
                sell_order.kwargs["sell_price"] = price_now    
                self.sell_list[sell_item.order_book_id].kwargs["need_resub"] = True                 
                # 撤单
                self.cancel_order(sell_order)
            # 预测卖单未成交，如果当前价格与挂单价差在0.5个百分点(配置项)以内，以当前价格重新挂单  
            if sell_order.kwargs["sell_reason"]==SellReason.PRED.value:
                pred_sell_continue_rate = self.strategy.sell_opt.pred_sell_continue_rate
                limit_price = sell_order.kwargs["sell_price"] * (1-pred_sell_continue_rate/100)
                if price_now < limit_price:
                    # 超出价差则忽略
                    self.logger_info("pred_sell pending,ignore,price_now:{},limit_price:{}".format(price_now,limit_price))
                    continue                
                # 先撤单
                self.cancel_order(sell_order)
                # 更新挂单列表，后续统一处理
                sell_order = self.get_sell_order(sell_item.order_book_id, context=context)
                sell_order.kwargs["sell_price"] = limit_price
                self.sell_list[sell_item.order_book_id].kwargs["need_resub"] = True

    def verify_order_buying(self,context):
        """核查买入订单"""

        env = Environment.get_instance()
        buy_list_active = self.trade_entity.get_buy_list_active(str(self.trade_day))
        # 已下单未成交的处理
        for index,buy_item in buy_list_active.iterrows():
            self.logger_info("check active,buy_item.order_book_id:{}".format(buy_item.order_book_id))
            sys_order = self.trade_entity.get_sys_order(buy_item.order_book_id)
            buy_order = self.get_buy_order(buy_item.order_book_id,context=context)
            cur_snapshot = self.get_current_snapshot(buy_item.order_book_id)
            if cur_snapshot is None:
                logger.warning("cur_snapshot None in verify_order_buying:{}".format(buy_item))
                continue            
            price_now = cur_snapshot.last
            dt = env.calendar_dt
            prev_day = get_tradedays_dur(dt,-1)
            h_bar = env.data_proxy.history_bars(buy_item.order_book_id,1,"1d","close",prev_day)
            price_last_day = h_bar[0]               
            pred_buy_exceed_rate = self.strategy.buy_opt.pred_buy_exceed_rate
            try_cnt_limit = self.strategy.buy_opt.try_cnt_limit
            # 如果超出昨日收盘5个百分点(配置项)，则换股票
            if (price_now - price_last_day)/price_last_day*100>pred_buy_exceed_rate:
                self.logger_info("pred_buy_exceed ,now:{},price_last_day:{}".format(price_now,price_last_day))
                # 先撤单
                self.cancel_order(buy_order)
                # 从待买入列表中去除,通过设置状态实现               
                self.update_order_status(buy_item,ORDER_STATUS.CANCELLED,side=SIDE.BUY, context=context)
                # 从候选列表中挑选新的股票，放入待买入列表中
                self.pick_to_buy_list(context)
            # 如果超出重复挂单限制，则换股票
            elif self.buy_try_cnt>try_cnt_limit:
                self.logger_info("try_cnt_limit exceed,{}".format(self.buy_try_cnt))
                # 先撤单
                self.cancel_order(buy_order)
                # 从待买入列表中去除,通过设置状态为已取消               
                self.update_order_status(buy_item,ORDER_STATUS.CANCELLED,side=SIDE.BUY, context=context)
                # 从候选列表中挑选新的股票，放入待买入列表中
                self.pick_to_buy_list(context)                
            # 如果未超出，则按照当前价格重新挂单
            else:
                # 如果当前价格和挂盘价格相差不大，则重报
                self.logger_info("need cancel,price_now:{},sys_order.price:{}".format(price_now,sys_order.price))
                # 如果报价高于当前价格，则不处理
                if price_now<sys_order.price:
                    continue
                # 设置重新报单标志
                self.buy_list[buy_item.order_book_id].kwargs["need_resub"] = True                
                # 发起撤单
                self.cancel_order(buy_order)

        
        # 已拒绝订单，重新按照现在价格下单
        buy_list_reject = self.trade_entity.get_buy_list_reject(str(self.trade_day))
        for index,buy_item in buy_list_reject.iterrows():
            try:
                cur_snapshot = self.get_current_snapshot(buy_item.order_book_id)
                price_now = cur_snapshot.last
            except Exception as e:
                logger.error("cur_snapshot err:{}".format(e))
                continue
            # 修改状态    
            self.update_order_status(buy_item,ORDER_STATUS.PENDING_NEW,side=SIDE.BUY, context=context)      
            # 更新报价     
            self.buy_list[buy_item.order_book_id].kwargs["buy_price"] = price_now                    
                                
    ###############################数据逻辑处理部分########################################  

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
        market_price = env.get_last_price(order_book_id)
        return market_price 
    
    