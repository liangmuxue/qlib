from rqalpha.apis import *
import rqalpha
from rqalpha.model import Order
from rqalpha.const import SIDE

from .backtest_base import BaseStrategy,SellReason

from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays

from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from cus_utils.data_filter import get_topN_dict
from cus_utils.log_util import AppLogger
logger = AppLogger()


class MinuteStrategy(BaseStrategy):
    """交易对象策略类,分钟级别"""
    
    def __init__(self):
        super().__init__()
        env = Environment.get_instance()
        # 穿透取得自定义datasource，后续可以直接使用
        self.ds = env.data_proxy._data_source
        # 记录交易日期
        self.trade_day = None
        self.prev_day = None
        
    def before_trading(self,context):
        """交易前准备"""
        
        logger.info("before_trading.now:{}".format(context.now))
        pred_date = int(context.now.strftime('%Y%m%d'))
        # 设置上一交易日，用于后续挂牌确认
        if self.trade_day is None:
            self.prev_day = None
        else:
            self.prev_day = get_previous_trading_date(self.trade_day)
        self.trade_day = pred_date
        
        # 根据当前日期，进行预测计算
        # TODO
        
        # 根据预测计算，筛选可以买入的股票
        candidate_list = context.ml_context.filter_buy_candidate(pred_date)
        candidate_list_buy = {}
        for instrument in candidate_list:
            # 代码转化为rqalpha格式
            order_book_id = transfer_order_book_id(instrument)
            # 以昨日收盘价格作为当前卖盘价格,注意使用未复权的数据，以保持和通达信导入的数据一致
            buy_price = history_bars(order_book_id,1,"1d",adjust_type="none")
            # 设定单次购买金额限制,以总资产为基准
            amount = (context.portfolio.market_value+context.portfolio.cash) * self.strategy.single_buy_mount_percent / 100            
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
            order = Order.__from_create__(
                order_book_id=order_book_id,
                quantity=amount,
                side=SIDE.BUY,
                # 自定义属性
                buy_price=buy_price,
            )
            candidate_list_buy[order_book_id] = order
            
        # 查看持仓，根据预测模型计算,逐一核对是否需要卖出
        sell_list = {}
        for position in get_positions():
            instrument = int(transfer_instrument(position.order_book_id))
            flag = context.ml_context.measure_pos(pred_date,instrument)
            if flag:
                pos_info = get_position(position.order_book_id)
                amount = pos_info.quantity
                # 以昨日收盘价格卖出
                sell_price = history_bars(order_book_id,1,"1d",adjust_type="none")
                # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
                order = Order.__from_create__(
                    order_book_id=position.order_book_id,
                    quantity=amount,
                    side=SIDE.SELL,
                    # 自定义属性
                    sell_price=sell_price,
                    sell_reason=SellReason.PRED.value
                )                
                sell_list[position.order_book_id] = order
                
        # 保存到上下文
        self.candidate_list_buy = candidate_list_buy
        # 根据买单数量配置，设置买单列表
        position_max_number = self.strategy.position_max_number
        # 从买入股票候选列表中，根据配置取得待买入列表
        context.buy_list = get_topN_dict(candidate_list_buy,position_max_number)
        context.sell_list = sell_list
        # 撤单列表
        context.cancel_list = []
    
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        # 根据盘前的分析计算，在集合竞价阶段直接挂单
        self.order_process(context)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        if not self.ds.valid_bar_date(context.now):
            return
        
        logger.info("handle_bar.now:{}".format(context.now))
        
        # 已提交卖单及买单检查
        self.verify_order_selling(context)
        self.verify_order_buying(context)
        
        # 卖出逻辑，止跌卖出        
        self.stop_fall_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，止盈卖出        
        self.stop_raise_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，持有股票超期卖出        
        self.expire_day_logic(context,bar_dict=bar_dict)     
        
        # 统一执行买卖挂单处理
        self.order_process(context)
      
    def order_process(self,context):
        """挂单流程，先卖后买"""
        
        self.sell_order(context)
        self.buy_order(context) 
        
    def buy_order(self,context):
        """买盘挂单"""
        
        # 如果持仓股票超过指定数量，则不操作
        position_number = len(get_positions())
        position_max_number = self.strategy.position_max_number
        if position_number>=position_max_number:
            logger.info("full pos")
            return
        
        # 轮询候选列表进行买入操作
        for buy_order in context.buy_list:
            # 只对待买入状态进行挂单
            if buy_order.status!=ORDER_STATUS.PENDING_NEW:
                continue
            # 以当日收盘价格挂单买入,买入量参考单次限定购买金额
            order_book_id = buy_order.order_book_id
            price = order.kwargs["buy_price"]
            quantity = order.quantity
            order = submit_order(order_book_id,quantity,SIDE.BUY,price=price)
            # 添加到跟踪变量
            self.trade_entity.add_order(order)
            # 手动累加，如果购买不成功，后续还需要有流程进行再次购买
            position_number += 1
            if position_number>=position_max_number:
                logger.info("full pos in buy list")
                break        
            
    def sell_order(self,context):
        """卖盘挂单"""

        # 检查待卖出列表，匹配卖出
        for position in get_positions():
            order_book_id = position.order_book_id
            pos_info = get_position(order_book_id)
            if order_book_id in context.sell_list:
                # 取得预卖出订单
                sell_order = context.sell_list[order_book_id]
                # 只对待卖出状态进行挂单
                if sell_order.status!=ORDER_STATUS.PENDING_NEW:
                    continue                
                # 全部卖出
                sell_amount = pos_info.quantity
                # 挂单卖出
                order = submit_order(order_book_id,sell_amount,SIDE.SELL,price=sell_order.kwargs["sell_price"])
                order.sell_reason = sell_order.kwargs["sell_reason"]
                # 添加到跟踪变量
                self.trade_entity.add_order(order)

    def verify_order_selling(self,context):
        """核查卖出订单"""
        
        sell_list_active = self.trade_entity.get_sell_list_active(self.trade_day)
        for sell_item in sell_list_active:
            sys_order = self.trade_entity.get_sys_order(sell_item.order_book_id)
            if sell_item.sell_reason==SellReason.STOP_RAISE.value:
                # 止盈卖出未成交，不进行操作
                logger.info("stop raise sell pending,ignore")
            if sell_item.sell_reason==SellReason.STOP_FALL.value:
                # 止损卖出未成交，以低于0.5个百分点(配置项)价格重新挂单
                stop_fall_sell_continue_rate = self.strategy.sell_opt。stop_fall_sell_continue_rate
                new_price = sell_item.sell_price * (1-stop_fall_sell_continue_rate/100)
                # 先撤单
                cancel_order(sys_order)
                # 更新挂单列表，后续统一处理
                sell_order = self.get_sell_order(sell_item.order_book_id, context=context)
                sell_order.kwargs["sell_price"] = new_price
                # 这里需要修改状态为待挂单
                sell_order._status = ORDER_STATUS.PENDING_NEW
            if sell_item.sell_reason==SellReason.PRED.value:
                # 预测卖单未成交，以低于0.5个百分点(配置项)价格重新挂单
                pred_sell_continue_rate = self.strategy.sell_opt。pred_sell_continue_rate
                new_price = sell_item.sell_price * (1-pred_sell_continue_rate/100)
                # 先撤单
                cancel_order(sys_order)
                # 更新挂单列表，后续统一处理
                sell_order = self.get_sell_order(sell_item.order_book_id, context=context)
                # 这里需要修改状态为待挂单
                sell_order._status = ORDER_STATUS.PENDING_NEW
                sell_order.kwargs["sell_price"] = new_price

    def verify_order_buying(self,context):
        """核查买入订单"""
        
        buy_list_active = self.trade_entity.get_buy_list_active(self.trade_day)
        for buy_item in buy_list_active:
            sys_order = self.trade_entity.get_sys_order(buy_item.order_book_id)
            cur_snapshot = current_snapshot(buy_item.order_book_id)
            price_now = cur_snapshot.last
            price_last_day = history_bars(buy_item.order_book_id,1,"1d",adjust_type="none")
            pred_buy_exceed_rate = self.strategy.buy_opt。pred_buy_exceed_rate
            
            # 如果超出昨日收盘1个百分点，则换股票
            if (price_now - price_last_day)/price_last_day*100>pred_buy_exceed_rate:
                # 先撤单
                cancel_order(sys_order)
                # 从待买入列表中去除,通过设置状态实现               
                context.buy_list[buy_item.order_book_id]._status = ORDER_STATUS.CANCELLED
                # 从候选列表中挑选新的股票，放入待买入列表中
                self.pick_to_buy_list(context)
            # 如果未超出，则按照当前价格重新挂单
            else:
                # 先撤单
                cancel_order(sys_order)
                # 修改状态          
                context.buy_list[buy_item.order_book_id]._status = ORDER_STATUS.PENDING_NEW    
                # 更新报价     
                context.buy_list[buy_item.order_book_id].kwargs["buy_price"] = price_now
                                    
    def get_sell_order(self,order_book_id,context=None):
        if order_book_id in context.sell_list:
            return context.sell_list[order_book_id]
        return None
    
    def get_buy_order(self,order_book_id,context=None):
        if order_book_id in context.buy_list:
            return context.buy_list[order_book_id]
        return None    
    
    def pick_to_buy_list(self,context):
        """从买入候选列表中挑选新的股票，放入待买入列表中"""
        
        for index,(k,v) in enumerate(context.candidate_list_buy.items()):
            if k not in context.buy_list:
                context.buy_list[k] = v
                return v
    
    #####################################逻辑判断部分#################################################                                 
    def expire_day_logic(self,context,bar_dict=None):
        """持有股票超期卖出逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
        for position in get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = get_position(order_book_id)
            sell_amount = pos_info.quantity
            before_date = context.now.strftime('%Y%m%d')
            # 通过之前存储的交易信息，查找到对应交易
            trade_date = context.trade_entity.get_trade_date_by_instrument(order_book_id,SIDE.BUY,context.now)
            if trade_date is None:
                logger.warning("trade not found:{},{}".format(order_book_id,before_date))
                continue
            # 检查是否超期，以决定是否卖出
            if tradedays(trade_date,before_date)>keep_day_number:
                order = Order.__from_create__(
                    order_book_id=position.order_book_id,
                    quantity=sell_amount,
                    side=SIDE.SELL,
                    # 自定义属性
                    sell_price=pos_info.last_price,
                    sell_reason=SellReason.EXPIRE_DAY.value
                )                
                context.sell_list[position.order_book_id] = order    
                        
    def stop_fall_logic(self,context,bar_dict=None,stop_threhold=-5):
        """止跌卖出逻辑"""
        
        for position in get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                order = Order.__from_create__(
                    order_book_id=position.order_book_id,
                    quantity=sell_amount,
                    side=SIDE.SELL,
                    # 自定义属性
                    sell_price=pos_info.last_price,
                    sell_reason=SellReason.STOP_FALL.value
                )                
                context.sell_list[position.order_book_id] = order                
        
    def stop_raise_logic(self,context,bar_dict=None,stop_threhold=5):
        """止盈卖出逻辑"""
  
        for position in get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                order = Order.__from_create__(
                    order_book_id=position.order_book_id,
                    quantity=sell_amount,
                    side=SIDE.SELL,
                    # 自定义属性
                    sell_price=pos_info.last_price,
                    sell_reason=SellReason.STOP_RAISE.value
                )                
                context.sell_list[position.order_book_id] = order  
            
    ############################事件注册部分######################################
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        logger.info("*" * 10 + "Trade Handler" + "*" * 10)
        logger.info(trade)
        # 保存成单交易对象
        context.trade_entity.add_trade(trade)
        # 更新卖单状态
        if order.side() == SIDE.SELL:
            sell_order = self.get_sell_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            sell_order._status = ORDER_STATUS.FILLED           
        # 更新买单状态
        if order.side() == SIDE.BUY:
            buy_order = self.get_buy_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            buy_order._status = ORDER_STATUS.FILLED          
    
    def on_order_handler(self,context, event):
        order = event.order
        logger.info("*" * 10 + "Order Handler" + "*" * 10)
        logger.info(order)
        # 更新订单交易对象
        context.trade_entity.update_order_status(order,ORDER_STATUS.ACTIVE)     
        # 更新卖单状态
        if order.side() == SIDE.SELL:
            sell_order = self.get_sell_order(order.order_book_id, context=context)
            # 这里需要修改状态为已挂单
            sell_order._status = ORDER_STATUS.ACTIVE           
        # 更新买单状态
        if order.side() == SIDE.BUY:
            buy_order = self.get_buy_order(order.order_book_id, context=context)
            # 这里需要修改状态为已挂单
            buy_order._status = ORDER_STATUS.ACTIVE                  
            
            
            