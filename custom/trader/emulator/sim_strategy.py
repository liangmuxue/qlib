import time

from rqalpha.apis import *
import rqalpha
from rqalpha.const import SIDE,ORDER_STATUS
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays
from trader.emulator.juejin.trade_proxy_juejin import JuejinTrade

from cus_utils.data_filter import get_topN_dict
from cus_utils.log_util import AppLogger
logger = AppLogger()

class SimStrategy(BaseStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        super().__init__()
        
        # 设置交易日期为当天
        self.trade_day = None
        self.prev_day = None
        # 穿透取得自定义datasource，后续可以直接使用
        env = Environment.get_instance()
        self.ds = env.data_proxy._data_source
        self.frequency_sim = env.config.base.frequency_sim
        self.handle_bar_wait = env.config.base.handle_bar_wait
        
    def init_env(self):
        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
         
    def before_trading(self,context):
        """交易前准备"""
        
        logger.info("before_trading.now:{}".format(context.now))
            
        pred_date = int(context.now.strftime('%Y%m%d'))
        self.trade_day = pred_date
        # 设置上一交易日，用于后续挂牌确认
        self.prev_day = get_previous_trading_date(self.trade_day)
        # 根据当前日期，进行预测计算
        context.ml_context.prepare_data(pred_date)
        # 根据预测计算，筛选可以买入的股票
        # candidate_list = self.get_candidate_list(pred_date,context=context)
        candidate_list = [600520]
        # candidate_list = []
        candidate_list_buy = {}
        sell_list = {}
        
        # 从文件中加载的未成单的订单记录，维护到上下文
        buy_orders = self.trade_entity.get_buy_list_active(str(self.trade_day))
        for index,row in buy_orders.iterrows():   
            order = self.create_order(row["order_book_id"], row["quantity"], SIDE.BUY, row["price"])
            candidate_list_buy[row["order_book_id"]] = order
        sell_orders = self.trade_entity.get_sell_list_active(str(self.trade_day))
        for index,row in sell_orders.iterrows():              
            order = order = self.create_order(row["order_book_id"], row["quantity"], SIDE.SELL, row["price"])
            order.set_frozen_cash(0)
            order._status = row["status"]            
            sell_list[row["order_book_id"]] = order
                                
        for instrument in candidate_list:
            # 剔除没有价格数据的股票
            if instrument not in self.instruments_dict:
                continue
            market = self.instruments_dict[instrument]["market"]
            # 代码转化为标准格式
            order_book_id = transfer_order_book_id(instrument,type=market)
            # 如果已持仓当前股票，则忽略
            if self.get_postion_by_order_book_id(order_book_id) is not None:
                continue
            # 以昨日收盘价格作为当前卖盘价格,注意使用未复权的数据，以保持和通达信导入的数据一致
            h_bar = history_bars(order_book_id,1,"1d",fields="close",adjust_type="none")
            if h_bar is None:
                logger.warning("history bar None:{},date:{}".format(order_book_id,context.now))
                continue
            buy_price = h_bar[0,0]
            if order_book_id in candidate_list_buy:
                continue
            # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
            order = self.create_order(order_book_id, 0, SIDE.BUY, buy_price)
            candidate_list_buy[order_book_id] = order
        # 保存到上下文
        self.candidate_list_buy = candidate_list_buy         
        # 根据买单数量配置，设置买单列表
        position_max_number = self.strategy.position_max_number
        # 从买入股票候选列表中，根据配置取得待买入列表
        self.buy_list = get_topN_dict(candidate_list_buy,position_max_number)
        self.new_buy_list = {}
                   
        # 查看持仓，根据预测模型计算,逐一核对是否需要卖出
        for position in self.get_positions():
            if position.order_book_id in sell_list:
                continue            
            instrument = int(transfer_instrument(position.order_book_id))
            flag = context.ml_context.measure_pos(pred_date,instrument)
            if flag:
                pos_info = self.get_position(position.order_book_id)
                amount = pos_info.quantity
                # 以昨日收盘价格卖出
                h_bar = history_bars(position.order_book_id,1,"1d",fields="close",adjust_type="none")
                if h_bar is None:
                    continue
                sell_price = h_bar[0,0]                
                # 复用rqalpha的Order类,注意默认状态为新报单（ORDER_STATUS.PENDING_NEW）
                order = self.create_order(position.order_book_id, amount, SIDE.SELL, sell_price,sell_reason=SellReason.PRED.value)             
                sell_list[position.order_book_id] = order
        # 卖单保存到上下文    
        self.sell_list = sell_list
        # 撤单列表
        self.cancel_list = []
        self.buy_try_cnt = 0
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        # 根据盘前的分析计算，在集合竞价阶段直接挂单
        if self.frequency_sim:
            # 如果当前时间已经超过集合竞价时间了，则跳过
            now_time = datetime.datetime.now()
            if now_time.hour>9 or (now_time==9 and now_time.minute>=28):
                return
        self.order_process(context)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        
        logger.info("handle_bar.now:{}".format(context.now))
        
        # 如果之前有新增候选买入，在此添加
        for index,(k,v) in enumerate(self.new_buy_list.items()):
            self.buy_list[k] = v        
        # 如果非实时模式，则需要在相应前等待几秒，以保证先处理外部通知事件
        if self.handle_bar_wait:
            time.sleep(3)
        logger.debug("handle_bar process:{}".format(context.now))
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
        position_number = len(self.get_positions())
        position_max_number = self.strategy.position_max_number
        active_buy_list = self.trade_entity.get_buy_list_active(self.trade_day)
        # 进行中的订单加上已持仓订单数量，不能大于规定数量阈值 
        if position_number+len(active_buy_list)>=position_max_number:
            logger.info("full pos")
            return
        
        # 轮询候选列表进行买入操作
        for buy_order_id in self.buy_list:
            buy_order = self.buy_list[buy_order_id]
            # 只对待买入状态进行挂单
            logger.debug("buy order loop,order_book_id:{},status:{}".format(buy_order.status,buy_order.order_book_id))
            if buy_order.status!=ORDER_STATUS.PENDING_NEW:
                continue
            # 以当日收盘价格挂单买入
            order_book_id = buy_order.order_book_id
            price = buy_order.kwargs["buy_price"]
            # 买入数量需要根据当前现金动态计算
            quantity = self.dynamic_compute_quantity(order_book_id,price)
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
                logger.info("full pos in buy list")
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
                logger.info("closable 0 with:{}".format(order_book_id))
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
                # 添加到跟踪变量
                self.trade_entity.add_or_update_order(order,str(self.trade_day))
    
    def create_order(self,id_or_ins, amount, side, price, position_effect=None,sell_reason=None):
        """代理api的订单创建方法"""
        
        order_book_id = assure_order_book_id(id_or_ins)
        style = cal_style(price, None)
        order = Order.__from_create__(
            order_book_id=order_book_id,
            quantity=amount,
            side=side,
            style=style,
            position_effect=position_effect,
            # 自定义属性
            buy_price=price,
            try_cnt=0, 
            sell_reason=sell_reason           
        )        
        return order
    
    def submit_order(self,amount,order_in=None):
        """代理api的订单提交方法"""
        
        order_book_id = order_in.order_book_id
        env = Environment.get_instance()
        
        if (
                env.config.base.run_type != RUN_TYPE.BACKTEST
                and env.get_instrument(order_book_id).type == "Future"
        ):
            if "88" in order_book_id:
                raise RQInvalidArgument(
                    _(u"Main Future contracts[88] are not supported in paper trading.")
                )
            if "99" in order_book_id:
                raise RQInvalidArgument(
                    _(u"Index Future contracts[99] are not supported in paper trading.")
                )

        order_in._quantity = int(amount)

        if env.can_submit_order(order_in) or True:
            # order._price = 
            env.broker.submit_order(order_in)
            # 订单编号转换为字符串
            order_in._order_id = "rq_{}".format(order_in._order_id)
            return order_in
    
    def cancel_order(self,order):
        logger.debug("cancel_order in ,order:{}".format(order))
        cancel_order(order)
        
    def verify_order_selling(self,context):
        """核查卖出订单"""
        
        sell_list_active = self.trade_entity.get_sell_list_active(str(self.trade_day))
        if sell_list_active.shape[0]==0:
            return
        for index,sell_item in sell_list_active.iterrows():
            sys_order = self.trade_entity.get_sys_order(sell_item.order_book_id)
            sell_order = self.get_sell_order(sell_item.order_book_id,context=context)
            cur_snapshot = self.get_current_snapshot(sell_item.order_book_id)
            price_now = cur_snapshot.last
            if sell_order.kwargs["sell_reason"]==SellReason.STOP_RAISE.value:
                # 止盈卖出未成交，不进行操作
                logger.info("stop raise sell pending,ignore")
            # 止损卖出未成交，如果当前价格与挂单价差在0.5个百分点(配置项)以内，以当前价格重新挂单   
            if sell_order.kwargs["sell_reason"]==SellReason.STOP_FALL.value:
                stop_fall_sell_continue_rate = self.strategy.sell_opt.stop_fall_sell_continue_rate
                limit_price = sell_order.sell_price * (1-stop_fall_sell_continue_rate/100)
                if price_now < limit_price:
                    # 超出价差则忽略
                    logger.info("stop_fall_sell pending,ignore,price_now:{},limit_price:{}".format(price_now,limit_price))
                    continue
                # 先撤单
                self.cancel_order(sell_order)
                # 更新挂单列表，后续统一处理
                sell_order.kwargs["sell_price"] = price_now
                # 这里需要修改状态为待挂单
                self.update_order_status(sell_item,ORDER_STATUS.PENDING_NEW,side=SIDE.SELL, context=context,price=price_now) 
            # 预测卖单未成交，如果当前价格与挂单价差在0.5个百分点(配置项)以内，以当前价格重新挂单  
            if sell_order.kwargs["sell_reason"]==SellReason.PRED.value:
                pred_sell_continue_rate = self.strategy.sell_opt.pred_sell_continue_rate
                limit_price = sell_order.sell_price * (1-pred_sell_continue_rate/100)
                if price_now < limit_price:
                    # 超出价差则忽略
                    logger.info("pred_sell pending,ignore,price_now:{},limit_price:{}".format(price_now,limit_price))
                    continue                
                # 先撤单
                cancel_order(sys_order)
                # 更新挂单列表，后续统一处理
                sell_order = self.get_sell_order(sell_item.order_book_id, context=context)
                # 这里需要修改状态为待挂单
                self.update_order_status(sell_item,ORDER_STATUS.PENDING_NEW,side=SIDE.SELL, context=context,price=limit_price) 
                sell_order.kwargs["sell_price"] = limit_price

    def verify_order_buying(self,context):
        """核查买入订单"""
        
        buy_list_active = self.trade_entity.get_buy_list_active(str(self.trade_day))
        # 已下单未成交的处理
        for index,buy_item in buy_list_active.iterrows():
            logger.info("check active,buy_item.order_book_id:{}".format(buy_item.order_book_id))
            sys_order = self.trade_entity.get_sys_order(buy_item.order_book_id)
            buy_order = self.get_buy_order(buy_item.order_book_id,context=context)
            cur_snapshot = self.get_current_snapshot(buy_item.order_book_id)
            if cur_snapshot is None:
                logger.warning("cur_snapshot None in verify_order_buying:{}".format(buy_item))
                continue            
            price_now = cur_snapshot.last
            h_bar = history_bars(buy_item.order_book_id,1,"1d",fields="close",adjust_type="none")
            price_last_day = h_bar[0,0]               
            pred_buy_exceed_rate = self.strategy.buy_opt.pred_buy_exceed_rate
            pred_buy_ignore_rate = self.strategy.buy_opt.pred_buy_ignore_rate
            try_cnt_limit = self.strategy.buy_opt.try_cnt_limit
            # 如果超出昨日收盘1个百分点，则换股票
            if (price_now - price_last_day)/price_last_day*100>pred_buy_exceed_rate:
                logger.info("pred_buy_exceed ,now:{},price_last_day:{}".format(price_now,price_last_day))
                # 先撤单
                self.cancel_order(buy_order)
                # 从待买入列表中去除,通过设置状态实现               
                self.update_order_status(buy_item,ORDER_STATUS.CANCELLED,side=SIDE.BUY, context=context)
                # 从候选列表中挑选新的股票，放入待买入列表中
                self.pick_to_buy_list(context)
            # 如果超出重复挂单限制，则换股票
            elif self.buy_try_cnt>try_cnt_limit:
                logger.info("try_cnt_limit exceed,{}".format(self.buy_try_cnt))
                # 先撤单
                self.cancel_order(buy_order)
                # 从待买入列表中去除,通过设置状态为已取消               
                self.update_order_status(buy_item,ORDER_STATUS.CANCELLED,side=SIDE.BUY, context=context)
                # 从候选列表中挑选新的股票，放入待买入列表中
                self.pick_to_buy_list(context)                
            # 如果未超出，则按照当前价格重新挂单
            else:
                # 如果当前价格和挂盘价格相差不大，则忽略
                if abs(price_now - sys_order.price)/sys_order.price*100<pred_buy_ignore_rate:
                    logger.info("ignore range,price_now:{},sys_order.price:{}".format(price_now,sys_order.price))
                    # 累加尝试次数
                    buy_order.kwargs["try_cnt"]+=1
                    continue
                logger.info("need cancel,price_now:{},sys_order.price:{}".format(price_now,sys_order.price))
                # 先撤单
                self.cancel_order(buy_order)
                # 修改状态    
                self.update_order_status(buy_item,ORDER_STATUS.PENDING_NEW,side=SIDE.BUY, context=context)      
                # 更新报价     
                self.buy_list[buy_item.order_book_id].kwargs["buy_price"] = price_now
                buy_order.kwargs["try_cnt"]+=1
        
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
                        
    ###############################自有数据逻辑########################################  
    
    def dynamic_compute_quantity(self,order_book_id,price):
        """动态计算需要买入的数量"""
        
        # 使用额度占比
        total_pos_value_rate = self.strategy.single_buy_mount_percent/100*self.strategy.position_max_number
        # 持仓股票个数
        position_number = len(self.get_positions())
        # 通过计算使用额度占比，以及当前持仓数量和单只股票购买额度，动态计算购买数量
        remain_number = self.strategy.position_max_number - position_number       
        # 保持非0
        if remain_number<=0:
            remain_number = 1         
        # 剩余额度为当前现金除以可以买入的股票个数额度，并乘以使用额度占比
        portfolio = self.get_portfolio()
        remain_quota = portfolio.cash/remain_number * total_pos_value_rate      
        amount = remain_quota / price   
        logger.info("total_pos_value_rate:{},position_number:{},remain_number:{},cash:{},remain_quota:{},price:{},amount:{}" \
                    .format(total_pos_value_rate,position_number,remain_number,portfolio.cash,remain_quota,price,amount))
        # 如果数量凑不够100股，则取消
        if amount<100:
            logger.warning("volume exceed:{}".format(order_book_id))
            return 0    
        # 取100的整数倍 
        amount = int(amount/100) * 100
        return amount
    
    def get_candidate_list(self,pred_date,context=None):
        candidate_list = context.ml_context.filter_buy_candidate(pred_date)
        return candidate_list

    def get_last_price(self,order_book_id):
        snapshot = self.get_current_snapshot(order_book_id)
        return snapshot.last
                                     
    def get_sell_order(self,order_book_id,context=None):
        if order_book_id in self.sell_list:
            return self.sell_list[order_book_id]
        return None
    
    def get_buy_order(self,order_book_id,context=None):
        if order_book_id in self.buy_list:
            return self.buy_list[order_book_id]
        return None    
    
    def pick_to_buy_list(self,context):
        """从买入候选列表中挑选新的股票，放入待买入列表中"""
        
        for index,(k,v) in enumerate(self.candidate_list_buy.items()):
            if k not in self.buy_list:
                # 先放入待新增列表中，下一个时间窗进行新增
                self.new_buy_list[k] = v
                return v
    
    def get_postion_by_order_book_id(self,order_book_id,context=None):
        """查看某个股票的持仓"""
        
        for position in self.get_positions():
            if order_book_id==position.order_book_id:
                return position
        return None
    
    def update_order_status(self,order,status,side=SIDE.BUY,context=None,price=0):
        """修改订单状态"""
        
        if side==SIDE.BUY:
            self.buy_list[order.order_book_id]._status = status
        else:
            self.sell_list[order.order_book_id]._status = status
        # 同时修改交易订单状态
        self.trade_entity.update_order_status(order,status,price=price)

    #####################################代理实现相关################################################# 
    def get_portfolio(self):
        """取得投资组合信息"""
        
        # 通过代理类，取得仿真环境的数据
        trade_proxy = Environment.get_instance().broker.trade_proxy
        return trade_proxy.get_portfolio()
    
    def get_position(self,order_book_id):
        """取得指定股票的持仓信息"""
        
        # 通过代理类，取得仿真环境的数据
        trade_proxy = Environment.get_instance().broker.trade_proxy
        return trade_proxy.get_position(order_book_id)

    def get_positions(self):
        """取得持仓信息"""
        
        # 通过代理类，取得仿真环境的数据
        trade_proxy = Environment.get_instance().broker.trade_proxy
        return trade_proxy.get_positions()
     
    def get_current_snapshot(self,order_book_id):  
        """取得指定股票当前快照"""
        
        env = Environment.get_instance()
        snapshot = env.data_proxy.current_snapshot(order_book_id, env.config.base.frequency,env.calendar_dt)
        return snapshot
        
    #####################################逻辑判断部分#################################################                                 
    def expire_day_logic(self,context,bar_dict=None):
        """持有股票超期卖出逻辑"""
        
        keep_day_number = self.strategy.keep_day_number
        for position in self.get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            before_date = context.now.strftime('%Y%m%d')
            # 通过之前存储的交易信息，查找到对应交易
            trade_date = self.trade_entity.get_trade_date_by_instrument(order_book_id,SIDE.BUY,context.now)
            if trade_date is None:
                logger.warning("trade not found:{},{}".format(order_book_id,before_date))
                continue
            # 检查是否超期，以决定是否卖出
            dur_days = tradedays(trade_date,before_date)
            if dur_days>keep_day_number:
                logger.info("expire,trade_date:{},and dur_day:{}".format(trade_date,dur_days))
                order = self.create_order(position.order_book_id, sell_amount, SIDE.SELL, pos_info.last_price,sell_reason=SellReason.EXPIRE_DAY.value)                
                self.sell_list[position.order_book_id] = order    
                        
    def stop_fall_logic(self,context,bar_dict=None):
        """止跌卖出逻辑"""
        
        stop_threhold = self.strategy.sell_opt.stop_fall_percent
        for position in self.get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100<stop_threhold:
                order = self.create_order(position.order_book_id, sell_amount, SIDE.SELL, pos_info.last_price,sell_reason=SellReason.STOP_FALL.value)                
                self.sell_list[position.order_book_id] = order                
        
    def stop_raise_logic(self,context,bar_dict=None):
        """止盈卖出逻辑"""
        
        stop_threhold = self.strategy.sell_opt.stop_raise_percent
        
        for position in self.get_positions():
            order_book_id = position.order_book_id
            if self.get_sell_order(order_book_id,context=context) is not None:
                # 如果已经在卖出列表中，则不操作
                continue
            pos_info = self.get_position(order_book_id)
            sell_amount = pos_info.quantity
            # 如果下跌幅度(与买入价格比较)超过阈值(百分点)，则以当前收盘价格卖出
            if (pos_info.last_price-pos_info.avg_price)/pos_info.avg_price*100>stop_threhold:
                order = Order.__from_create__(
                    order_book_id=position.order_book_id,
                    quantity=sell_amount,
                    side=SIDE.SELL,
                    style=None,
                    position_effect=None,                       
                    # 自定义属性
                    sell_price=pos_info.last_price,
                    sell_reason=SellReason.STOP_RAISE.value
                )                
                self.sell_list[position.order_book_id] = order  

            
    ############################事件注册部分######################################
    def on_trade_handler(self,context, event):
        trade = event.trade
        order = event.order
        account = event.account
        logger.debug("on_trade_handler in,order:{}".format(order))
        # logger.info("*" * 10 + "Trade Handler" + "*" * 10)
        # logger.info(trade)
        # 保存成单交易对象
        self.trade_entity.add_trade(trade)
        # 更新卖单状态
        if order.side == SIDE.SELL:
            sell_order = self.get_sell_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            sell_order._status = ORDER_STATUS.FILLED      
            # 卖出一个，就可以再买一个。从候选列表中挑选新的股票，放入待买入列表中
            self.pick_to_buy_list(context)                 
        # 更新买单状态
        if order.side == SIDE.BUY:
            buy_order = self.get_buy_order(order.order_book_id, context=context)
            # 这里需要修改状态为已成交
            buy_order._status = ORDER_STATUS.FILLED          
    
    def on_order_handler(self,context, event):
        order = event.order
        logger.info("order handler,order:{}".format(order))
        # 如果订单被拒绝，则忽略,仍然保持新单状态，后续会继续下单
        if order.status==ORDER_STATUS.REJECTED:
            logger.warning("order reject:{},trade_date:{}".format(order.order_book_id,self.trade_day))
            self.trade_entity.add_or_update_order(order,str(self.trade_day))  
            return
        # 添加到跟踪变量
        self.trade_entity.add_or_update_order(order,str(self.trade_day))           
        # 更新卖单状态
        if order.side == SIDE.SELL:
            sell_order = self.get_sell_order(order.order_book_id, context=context)
            # 这里需要修改状态为已挂单
            sell_order._status = order.status           
        # 更新买单状态
        if order.side == SIDE.BUY:
            buy_order = self.get_buy_order(order.order_book_id, context=context)
            # 这里需要修改状态为已挂单
            buy_order._status = order.status            
            
    
    