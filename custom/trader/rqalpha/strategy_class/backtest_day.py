from rqalpha.apis import *
import rqalpha

from .backtest_base import BaseStrategy

from trader.rqalpha.ml_context import MlIntergrate
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.rqalpha.trade_entity import TradeEntity
from trader.utils.date_util import tradedays

from cus_utils.log_util import AppLogger
logger = AppLogger()


# 交易信息表字段，分别为交易日期，股票代码，成交价格，成交量,总价格，成交状态，订单编号
TRADE_COLUMNS = ["trade_date","instrument","side","price","quantity","total_price","status","order_id"]

class DayStrategy(BaseStrategy):
    """交易策略类，分钟模式"""
    
    def __init__(self):
        super().__init__()
        
    def before_trading(self,context):
        """交易前准备"""
        
        logger.info("before_trading.now:{}".format(context.now))
        pred_date = int(context.now.strftime('%Y%m%d'))
        # 根据当前日期，进行预测计算
        # TODO
        
        # 根据预测计算，筛选可以买入的股票
        candidate_list = context.ml_context.filter_buy_candidate(pred_date)
        # 代码转化为rqalpha格式
        candidate_list = [transfer_order_book_id(instrument) for instrument in candidate_list]
        # 查看持仓，根据预测模型计算,逐一核对是否需要卖出
        sell_list = []
        for position in get_positions():
            pos_info = get_position(position.order_book_id)
            instrument = int(transfer_instrument(position.order_book_id))
            flag = context.ml_context.measure_pos(pred_date,instrument)
            if flag:
                sell_list.append(pos_info)
                
        # 保存到上下文
        context.candidate_list = candidate_list
        context.sell_list = sell_list
        
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        logger.info("handle_bar.now:{}".format(context.now))
        # 计算模型的预测数据
        pred_df = context.ml_context.pred_df
        pred_date = context.now.strftime('%Y%m%d')
    
        # 卖出逻辑1，检查持仓列表，匹配卖出
        for position in get_positions():
            order_book_id = position.order_book_id
            pos_info = get_position(order_book_id)
            if order_book_id in context.sell_list:
                # 卖出思路：以昨日收盘价格卖出
                cur_snapshot = current_snapshot(order_book_id)
                sell_amount = pos_info.quantity
                # 以当日收盘挂单卖出
                submit_order(order_book_id,sell_amount,SIDE.SELL,price=cur_snapshot.prev_close)
        
        # 卖出逻辑，止跌卖出        
        self.stop_fall_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，止盈卖出        
        self.stop_raise_logic(context,bar_dict=bar_dict) 
        # 卖出逻辑，持有股票超期卖出        
        self.expire_day_logic(context,bar_dict=bar_dict)     
        # 买入逻辑，检查持仓列表，匹配卖出
        self.buy_logic(context,bar_dict=bar_dict)        


    def expire_day_logic(self,context,bar_dict=None):
        """持有股票超期卖出逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
        for position in get_positions():
            order_book_id = position.order_book_id
            pos_info = get_position(order_book_id)
            cur_snapshot = current_snapshot(order_book_id)
            sell_amount = pos_info.quantity
            before_date = context.now.strftime('%Y%m%d')
            # 通过之前存储的交易信息，查找到对应交易
            trade_date = context.trade_entity.get_trade_date_by_instrument(order_book_id,SIDE.BUY,before_date)
            if trade_date is None:
                logger.warning("trade not found:{},{}".format(order_book_id,before_date))
                continue
            # 检查是否超期，以决定是否卖出
            if tradedays(trade_date,before_date)>keep_day_number:
                submit_order(order_book_id,sell_amount,SIDE.SELL,price=pos_info.last_price)  
        
    def stop_fall_logic(self,context,bar_dict=None,stop_threhold=-5):
        """止跌卖出逻辑"""
        
        for position in get_positions():
            order_book_id = position.order_book_id
            pos_info = get_position(order_book_id)
            cur_snapshot = current_snapshot(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当日收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                submit_order(order_book_id,sell_amount,SIDE.SELL,price=pos_info.last_price)   
        
    def stop_raise_logic(self,context,bar_dict=None,stop_threhold=5):
        """止盈卖出逻辑"""
        
        for position in get_positions():
            order_book_id = position.order_book_id
            pos_info = get_position(order_book_id)
            cur_snapshot = current_snapshot(order_book_id)
            sell_amount = pos_info.quantity
            # 如果上涨幅度(与买入价格比较)超过阈值(百分点)，则以当日收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                submit_order(order_book_id,sell_amount,SIDE.SELL,price=pos_info.last_price)    
                   
    def buy_logic(self,context,bar_dict=None):
        """买入逻辑"""
        
        # 设定单次购买金额限制,以总资产为基准
        buy_limit = (context.portfolio.market_value+context.portfolio.cash) * self.strategy.single_buy_mount_percent / 100
        # 如果持仓股票超过指定数量，则不操作
        position_number = len(get_positions())
        position_max_number = self.strategy.position_max_number
        if position_number>=position_max_number:
            logger.info("full pos")
            return
        
        # 轮询候选列表进行买入操作
        for instrument in context.candidate_list:
            # 以当日收盘价格挂单买入,买入量参考单次限定购买金额
            cur_snapshot = current_snapshot(instrument)
            order_value(instrument,buy_limit,price=cur_snapshot.last)
            # 手动累加，如果购买不成功，后续还需要有流程进行再次购买
            position_number += 1
            if position_number>=position_max_number:
                logger.info("full pos in candidate list")
                break        
            
            
            
    ############################事件注册部分######################################
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        logger.info("*" * 10 + "Trade Handler" + "*" * 10)
        logger.info(trade)
        logger.info(order)
        logger.info(account)
        # 保存交易对象
        context.trade_entity.add_trade(trade)
    
    def on_order_handler(self,context, event):
        order = event.order
        # logger.info("*" * 10 + "Order Handler" + "*" * 10)
        # logger.info(order)
            
            
            
            