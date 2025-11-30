import time
import os
import datetime
import pandas as pd
import numpy as np

from rqalpha.environment import Environment
from rqalpha.apis import cal_style

from cus_utils.data_aug import DictToObject
from trader.rqalpha.strategy_class.backtest_base import SellReason
from trader.utils.constance import OrderStatusType
from trader.emulator.portfolio import Portfolio,SimOrder
from trader.rqalpha.ml_wf_context import FurWorkflowIntergrate
from rqalpha.core.events import Event
from trader.rqalpha.core.event import EVENT
from trader.emulator.futures_backtest_strategy import FurBacktestStrategy,POS_COLUMNS
from trader.utils.date_util import tradedays,get_tradedays_dur,get_prev_working_day,get_next_working_day
from rqalpha.const import ORDER_STATUS,SIDE,POSITION_EFFECT,POSITION_DIRECTION,DEFAULT_ACCOUNT_TYPE

from cus_utils.log_util import AppLogger
logger = AppLogger()

SIM_POS_COLUMNS = POS_COLUMNS + ['margin']

class FurSimulationStrategy(FurBacktestStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        self.time_begin = None
        self.proxy_name = proxy_name
        
        # 设置策略模拟仓位，用于策略逻辑判断
        self.sim_position = pd.DataFrame(columns=SIM_POS_COLUMNS)  
        # 匹配远程的本地存储的仓位记录，用于对照
        self.real_position = pd.DataFrame(columns=SIM_POS_COLUMNS)   
        self.contract_today = {}
        self.prev_side = SIDE.BUY

    def __build_with_context__(self,work_context,workflow_mode=False):
        
        self.workflow = work_context
        context = work_context.context
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
        # 读取本地存储的仓位数据
        save_path = self.get_pos_storage_filepath()    
        if os.path.exists(save_path):
            self.sim_position = pd.read_csv(save_path,parse_dates=['trade_date','datetime'],infer_datetime_format=True)
                     
    def before_trading(self,context):
        """交易前准备"""
        
        self.logger_info("before_trading.now:{}".format(context.now))
        
        keep_day_number = self.strategy.keep_day_number
        cur_date = context.trade_date.strftime("%Y%m%d")
        self.trade_day = int(cur_date)
        # 检查是否特殊交易时间
        self.spec_trading_time = self.data_source.is_spec_trading_time(context.trade_date)        
        env = Environment.get_instance()
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        
        # 加载当日可以交易的合约品种
        self.data_source.load_all_contract(trade_date_str=cur_date)
        # 设置开仓列表
        self.open_list = {} 
        self.close_list = {}  
                
        # 同步数据，从CTP远端系统中同步账户、持仓、交易等信息到本地
        por_info = self.query_ctp_por_data()
        # 同步到投资组合对象
        portfolio = self.sync_portfolio(cur_date,por_info)    
        logger.info("account info:{}".format(portfolio.get_account()))
        env.set_portfolio(portfolio)   
        # 同步当日订单
        if emu_args.sync_order_data:
            orders = self.sync_orders(cur_date,por_info)   
            if orders is not None:
                # 同步到到交易存储类
                self.transfer_order(orders,date=cur_date)
                
        pred_date = self.trade_day
        # 设置上一交易日，用于后续挂牌确认
        self.prev_day = self.get_previous_trading_date(self.trade_day)
        # 初始化当日合约对照表
        self.date_trading_mappings = self.data_source.build_trading_contract_mapping(context.now)        
        # 根据当前日期，进行预测计算
        context.ml_context.prepare_data(pred_date)   
            
        # 根据预测计算，筛选可以买入的品种
        candidate_list = self.get_candidate_list(pred_date,context=context)
        # candidate_list = [(0,"B"),(0,"M"),(1,"OI"),(1,"MA"),(0,"P"),(0,"FG"),(1,"FB"),(1,"A")]
        # candidate_list = [(0,"SN"),(0,"SS"),(1,"FB"),(1,"RU")]
        # candidate_list = []
        
        self.lock_list = {}        
        candidate_order_list = {}  
        # 撤单列表
        self.cancel_list = []        
        # 综合候选品种以及当前已持仓品种，生成维护开仓和平仓列表
        positions = self.get_positions()
        pos_number = len(positions)
        # 由于当前可能已经进入交易时间了，因此首先查找当日订单，并维护相关数据
        exists_orders = self.trade_entity.get_exits_order_list(context.trade_date.strftime("%Y%m%d"))
        exists_order_ids = {}
        for index,row in exists_orders.iterrows():
            pos = self.get_position(row['order_book_id'],trade_date=context.trade_date)
            # 除了品种代码一致外，方向也需要一致
            if pos is None:
                # 如果不在持仓中，才累加pos总数
                pos_number += 1
            else:
                trend_flag = 1 if pos.direction==POSITION_DIRECTION.LONG else 0
                trade_data_trend = 1 if row['side']==SIDE.BUY else 0
                if trade_data_trend!=trend_flag:
                    pos_number += 1
                else:
                    exists_order_ids[row['order_book_id']] = trend_flag
        # 处理候选列表
        for item in candidate_list:
            trend = item[0]
            instrument = item[1]
            # 剔除没有价格数据的品种
            if not self.has_current_data(pred_date,instrument,mode="instrument"):
                logger.warning("no data for buy:{},ignore".format(instrument))
                continue
            # 代码转化为标准格式
            order_book_id = self.data_source.transfer_futures_order_book_id(instrument,datetime.datetime.strptime(str(pred_date), '%Y%m%d'))
            # 如果已在订单中，则忽略
            if order_book_id in exists_order_ids and trend==exists_order_ids[order_book_id]:
                continue         
            # 如果已在当日持仓中，则忽略  
            if self.is_today_position(order_book_id):
                continue
            # 以昨日收盘价格作为当前卖盘价格
            h_bar = self.data_source.history_bars(order_book_id,1,"1d",dt=context.trade_date,fields="close")
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
            # 如果正好是已有仓位的反向指标，则直接挂平仓单
            inverse_direction = POSITION_DIRECTION.LONG if order.position_direction==POSITION_DIRECTION.SHORT else POSITION_DIRECTION.SHORT
            match_pos = self.match_positions(instrument,inverse_direction)
            inverse_match = False
            for index,row in match_pos.iterrows():
                amount = row['quantity']
                # 开仓编号对照
                open_order_id = row['order_id']
                order = self.create_order(order_book_id, amount, side,price,position_effect=POSITION_EFFECT.CLOSE,open_order_id=open_order_id,
                                    close_reason=SellReason.PRED.value,context=context)
                self.append_to_close_list(order)    
                inverse_match = True
            # 如果已经进入平仓列表，则不开仓了
            if inverse_match:
                continue              
            # 对于开仓候选品种，如果已持仓当前品种，则进行锁仓
            match_pos = self.match_positions(instrument,order.position_direction)
            lock_flag = False
            for index,row in match_pos.iterrows():
                order_book_id = row['order_book_id']
                trade_date = row['trade_date'].strftime("%Y%m%d")
                now_date = context.trade_date.strftime("%Y%m%d")
                dur_days = tradedays(trade_date,now_date)
                # 在模拟中，锁仓锁的是合约号，不是开仓单号
                if dur_days>=keep_day_number:
                    self.lock_list[row['order_book_id']] = trade_date   
                    # 更新本地仓位中的交易日期为当前日期，用于后续超期判断
                    self.update_sim_positions_date(order_book_id, trade_date, context.trade_date)
                    lock_flag = True 
            # 加入到候选开仓订单,注意已锁仓的就不开仓了
            if not lock_flag:
                candidate_order_list[order.order_book_id] = order
            
        # 开仓候选的订单信息，保存到上下文
        self.candidate_list = candidate_order_list   
        
        # 生成热加载数据，提升查询性能
        if self.strategy.building_hot_data:
            self.data_source.build_hot_loading_data(pred_date,self.close_list,reset=True) 
            self.data_source.build_hot_loading_data(pred_date,self.open_list)        
            self.data_source.build_hot_loading_data(pred_date,self.get_positions())    
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        return
        self.order_process(context)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口,首先进行撮合，然后进行策略"""
        
        self.logger_info("handle_bar.now:{}".format(context.now))
        
        self.query_position()
        
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
        self.pick_to_open_list()
        self.open_order(context) 

    def submit_order(self,amount,order_in=None,context=None):
        """代理api的订单提交方法"""
        
        order_book_id = order_in.order_book_id
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
    
    def cancel_order(self,order,need_ref=True):
        """撤单"""
        
        self.logger_info("cancel_order in ,order:{}".format(order.order_book_id))
        # 修改状态为待取消
        self.update_order_status(order,ORDER_STATUS.PENDING_CANCEL,side=order.side, context=self.context,price=order.price)     
        if "OrderSysID" not in order.kwargs:
            # 取得ctp原订单信息,需要匹配引用订单号
            if need_ref:
                orders = self.query_order_info(order.order_book_id,ref_order_id=order.secondary_order_id)
            else:
                orders = self.query_order_info(order.order_book_id)
            if len(orders)==0:
                logger.warning("no order in cancel:{}".format(order.order_book_id))
                return
            order = orders[0]
            if order.status!=ORDER_STATUS.ACTIVE:
                logger.warning("no active order in cancel:{}".format(order.order_book_id))
                return 
        self.context.get_trade_proxy().cancel_order(order)
                
    ###############################数据逻辑处理部分########################################  

    def pick_to_open_list(self):
        """从候选列表中挑选到开仓列表"""
                        
        position_max_number = self.strategy.position_max_number
        position_number = len(self.get_sim_positions())
        while True:
            # 检查是否超出数量限制时，需要包含已提交订单数量
            active_open_list_size = len(self.open_list)
            # 同时计算提交的平仓数量，即允许暂时超出数量限制，先开仓后平仓，当日达到数量平衡即可
            active_close_list_size = len(self.close_list)
            # 已持仓订单数量，不能大于规定数量阈值 
            if position_number+active_open_list_size-active_close_list_size>=position_max_number:
                self.logger_info("full pos")
                break
            # 依次从候选中选取对应品种并放入开仓列表
            candidate_order = self.pop_next_candidate()
            if candidate_order is None:
                self.logger_info("no candidate")
                break   
                
    def sync_portfolio(self,date,por_info):
        """ctp数据同步到投资组合"""
        
        (account,positions,orders) = por_info
        if positions is None:
            positions = []
        env = Environment.get_instance()
        
        persis_path = env.config.extra.persis_path
        financing_rate = env.config.mod.sys_account.financing_rate
        frozen = account['frozen']
        margin = account['margin']
        balance = account['balance']
            
        portfolio = Portfolio(balance,frozen,margin,positions,financing_rate,trade_date=date,data_proxy=env.data_proxy,persis_path=persis_path)
        # 同步到本地持仓存储
        self.sync_to_real_position(positions)
        # Mock Position Data
        if self.context.config.mod.ext_emulation_mod.mock_simulation:
            self.mock_sim_position()            
        return portfolio
 
    def sync_to_real_position(self,positions):
        
        sim_position = []
        for pos in positions:
            side = SIDE.BUY if pos.direction==POSITION_DIRECTION.LONG else SIDE.SELL
            # 根据持仓数据，判断持仓日期,分为：当日，昨日，超出昨日就前推2天
            if pos.TodayPosition==pos.Position:
                trade_date = self.context.trade_date
            elif pos.YdPosition==pos.Position:
                # 注意由于YdPosition代表实际时间下的昨日，因此这里使用当前时间作为参照
                trade_date = get_prev_working_day(self.context.now)
            else:
                trade_date = get_prev_working_day(get_prev_working_day(self.context.now))
            item = pd.DataFrame(np.array([[pos.order_book_id,pos.quantity,side,
                            pos.direction,pos.last_price,trade_date,datetime.datetime.now(),None,pos.margin]]),columns=SIM_POS_COLUMNS)
            sim_position.append(item)
        if len(sim_position)==0:
            return 
        # 直接时添加到仓位记录，不合并之前已有的
        sim_position = pd.concat(sim_position)    
        sim_position['trade_date'] = sim_position['trade_date'].astype('datetime64[ns]')
        self.real_position = sim_position
        
    def mock_sim_position(self):
        self.sim_position = self.sim_position.sort_values(by='order_book_id')
        # mid_book_id = self.sim_position.iloc[3]['order_book_id']
        # self.sim_position.loc[self.sim_position['order_book_id']<=mid_book_id,'trade_date'] = datetime.datetime.strptime("20251121", "%Y%m%d")
        # self.sim_position.loc[self.sim_position['order_book_id']>mid_book_id,'trade_date']  = datetime.datetime.strptime("20251124", "%Y%m%d")
        self.sim_position.loc[self.sim_position['order_book_id'].isin(['A2601','B2601','M2601']),'trade_date'] = datetime.datetime.strptime("20251121", "%Y%m%d")
        self.sim_position['trade_date'] = self.sim_position['trade_date'].astype('datetime64[ns]')
    
    def create_order(self,id_or_ins, amount, side,price, position_effect=None,close_reason=None,try_cnt=0,open_order_id=None,context=None):
        """代理api的订单创建方法"""
        
        order_book_id = id_or_ins
        multiplier = self.data_source.get_contract_info(order_book_id)["multiplier"].astype(float).values[0]
        # 添加交易所编码
        instrument = self.data_source.get_instrument_code_from_contract_code(order_book_id)
        exchange_code = self.data_source.get_exchange_from_instrument(instrument)
        style = cal_style(price, None)
        
        order = SimOrder.__from_create__(
            order_book_id=order_book_id,
            quantity=amount,
            side=side,
            style=style,
            position_effect=position_effect,
            # 自定义属性
            price=price,
            trade_date=self.context.trade_date,
            multiplier=multiplier,
            try_cnt=try_cnt, 
            close_reason=close_reason,  
            open_order_id=open_order_id,
            need_resub=False, 
            exchange_id=exchange_code      
        )   
        order.set_frozen_cash(0)    
        order.set_frozen_price(price)
              
        return order
       
    def sync_orders(self,date,por_info):
        """ctp订单数据同步"""
        
        (_,_,orders) = por_info
        if orders is None:
            return None
        
        persis_orders = []
        for order in orders:
            if order.quantity==0 or order.status==ORDER_STATUS.CANCELLED:
                continue
            persis_orders.append(order)
        
        return persis_orders
    
    def transfer_order(self,orders,date=None,ignore_before=True):
        """"把远程订单信息同步到本地交易存储类"""
        
        # 首先移除当日全部订单数据
        moved_data = self.trade_entity.move_order_by_date(date)
        
        for order in orders:
            # 忽略之前的订单数据，只关注当天的
            if ignore_before and order.trading_datetime.strftime("%Y%m%d")==date.strftime("%Y%m%d"):
                continue
            # 遍历从远程取得的订单信息，逐个进行业务添加
            self.trade_entity.add_or_update_order(order,date.strftime("%Y%m%d"))      
            # 还需要添加到平仓列表
            if order.position_effect==POSITION_EFFECT.CLOSE and order.status!=ORDER_STATUS.CANCELLED:
                trade_day = order.trading_datetime.strftime("%Y%m%d")
                # 确认是否有持仓
                if self.get_position(order.order_book_id,trade_date=date) is None:
                    continue
                # 忽略当天的平仓
                if trade_day==date.strftime("%Y%m%d"):
                    continue
                self.append_to_close_list(order)
    
    def query_ctp_por_data(self,has_order=True):
        """请求CTP远端系统数据"""
        
        ctp_trade_proxy = self.context.get_trade_proxy()
        # 请求远端CTP数据
        account = ctp_trade_proxy.query_account_info()
        if account is None:
            logger.warning("account None,again")
            account = ctp_trade_proxy.query_account_info()
        positions = ctp_trade_proxy.query_position_info("")
        if has_order:
            orders = ctp_trade_proxy.query_order_info("")
        else:
            orders = None
        return (account,positions,orders)
    
    def get_last_price(self,order_book_id):
        """取得指定标的最近报价信息"""

        env = Environment.get_instance()
        return self.data_source.get_last_price(order_book_id,env.trading_dt)
    
    def get_last_or_prevday_bar(self,order_book_id):
        """根据当前时间点，取得昨天或上一交易时间段的数据"""

        env = Environment.get_instance()
        # 如果还没有开盘，取上一交易日数据，否则取上一交易时间段数据
        if self.is_trade_opening(env.trading_dt,order_book_id):
            bar = self.data_source.get_last_bar(order_book_id,env.trading_dt)
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

    def match_position_order(self,pos,exists_order):
        """持仓和订单进行比较，查看是否属于一个品种"""
        
        # 除了品种代码一致外，方向也需要一致
        if pos.order_book_id==exists_order.order_book_id and pos.position_effect==exists_order.position_effect:
            return True
        return False

    def is_in_lock(self,pos):
        """判断是否锁仓，使用合约号加日期进行判断"""
        
        if not pos.order_book_id in self.lock_list:
            return False
        return True
          
    def is_today_position(self,order_book_id):  
        
        trade_date = self.context.trade_date
        sim_position = self.sim_position
        if sim_position.shape[0]==0:
            return False
        try:
            pos = sim_position[(sim_position['order_book_id']==order_book_id)&(sim_position['trade_date'].dt.strftime("%Y%m%d")==trade_date.strftime("%Y%m%d"))]
        except Exception:
            print("eee")
        if pos.shape[0]>0:
            return True
        return False

    def get_position_by_order(self,order):
        """取得指定合约的持仓信息"""
        
        keep_day_number = self.strategy.keep_day_number
        # 使用订单对应的合约编号查询
        order_book_id = order.order_book_id
        sim_position = self.sim_position
        # 根据当前日期前推到之前的日期，并查询
        pos_date = self.context.trade_date
        for _ in range(keep_day_number):
            pos_date = get_prev_working_day(pos_date)
        pos = sim_position[(sim_position['order_book_id']==order_book_id)&(sim_position['trade_date'].dt.strftime("%Y%m%d")==pos_date.strftime("%Y%m%d"))]
        if pos.shape[0]==0:
            return None
        dict_data = pos.iloc[0].to_dict()
        pos_obj = DictToObject(dict_data)          
        return pos_obj
      
    def get_positions(self):
        """取得持仓信息"""

        # 使用本地记录仓位信息
        sim_position = self.sim_position
        positions = []
        for index,pos in sim_position.iterrows():
            dict_data = pos.to_dict()
            pos_obj = DictToObject(dict_data)  
            positions.append(pos_obj)
       
        return positions

    def get_positions_real(self):
        """取得远程实际持仓信息"""
    
        env = Environment.get_instance()
        return env.portfolio.get_positions()
        
    def is_trade_opening(self,dt,order_book_id):
        """检查当前是否已开盘"""
        
        # 强制模拟交易模式
        if self.context.config.mod.ext_emulation_mod.force_trade_mode:
            return True
        
        return self.data_source.is_trade_opening_for_contract(order_book_id,dt)
           
    def has_current_data(self,date,code,mode="contract"):
        """当日是否开盘交易,使用懒加载缓存模式"""

        if mode=="instrument":
            # 品种模式查询，需要先根据品种取得合约代码再查询
            symbol = self.data_source.get_main_contract_name(code,str(date))
        else:
            symbol = code
        if symbol is None:
            return False
                            
        if symbol in self.contract_today:
            # 如果已经在缓存里，则直接放回缓存中的结果
            return self.contract_today[symbol]==1
        
        # 取得实时价格，如果有则说明当日有交易
        dt = self.context.now
        price = self.data_source.get_last_price(symbol,dt)
        flag = 1
        if price is None:
            flag = 0
        # 写入缓存
        self.contract_today[symbol] = flag
        
        return self.contract_today[symbol]==1

    def get_availabel(self):
        """获取可用资金"""
        
        portfolio = self.get_portfolio()  
        return portfolio.cash

    def get_pos_storage_filepath(self):
        """取得持仓存储文件路径"""
    
        env = Environment.get_instance()
        file_path = os.path.join(env.config.extra.report_save_path,"sim_position.csv")   
        return file_path
                            
    ############################事件注册部分######################################
    
    def on_trade_handler(self,event):
        trade = event.trade
        order = event.order
        account = event.account
        account.get_positions()
        self.logger_debug("on_trade_handler in,order:{},trade:{}".format(order,trade))
        # 保存成单交易对象
        self.trade_entity.add_trade(trade,multiplier=order.kwargs['multiplier'],order=order,context=self.context)
        # 修改当日仓位列表中的状态为已成交
        self.update_order_status(order,ORDER_STATUS.FILLED,side=order.side, context=self.context)     
        # 从开仓或平仓列表中删除
        if order.position_effect==POSITION_EFFECT.OPEN:
            self.remove_from_open_list(order.order_book_id)
        else:
            self.remove_from_close_list(order.order_book_id)
        # 维护仓位数据
        self.apply_trade_pos(trade,order)     
    
    def on_order_handler(self,event):
        
        context = self.context
        super().on_order_handler(context, event)    
                    
    def apply_trade_pos(self,trade,order):
        """维护仓位数据，使用事件通知模式，以避免查询异步失败问题"""
        
        logger.info("apply_trade_pos in")
        # 维护本地仓位跟踪记录
        sim_position = self.sim_position
        if trade.position_effect==POSITION_EFFECT.OPEN:
            item = pd.DataFrame(np.array([[trade.order_book_id,trade.last_quantity,trade.side,
                            trade.position_direction,trade.last_price,self.context.trade_date,trade.datetime,trade.order_id]]),columns=POS_COLUMNS)
            # 直接时添加到仓位记录，不合并之前已有的
            sim_position = pd.concat([sim_position,item])
        else:
            # 平仓时消除原仓位数据
            sim_position = sim_position[sim_position['order_id']!=order.kwargs['open_order_id']]
        self.sim_position = sim_position      
        # 实时保存到本地
        save_path = self.get_pos_storage_filepath()    
        self.sim_position.to_csv(save_path,index=False)
         
        # 通过RESOTRE事件，同步远程仓位数据 
        self.workflow.add_busi_event(Event(EVENT.DO_RESTORE))    
       
    def refresh_portfolio(self,event):
        """刷新仓位数据，使用事件通知模式，以避免查询异步失败问题"""
        
        env = Environment.get_instance()
        context = self.context
        cur_date = context.now.date()
        por_info = self.query_ctp_por_data(has_order=False)
        portfolio = self.sync_portfolio(cur_date,por_info)   
        logger.info("refresh_portfolio end,account:{}".format(portfolio.get_account()))
        env.set_portfolio(portfolio)
                             
    ######################### 辅助功能实现 ####################################
        
    def clear_position(self):
        """清空所有持仓"""
        
        # 遍历仓位，并执行平仓单
        for pos in self.get_positions_real():
            order_book_id = pos.order_book_id
            side = SIDE.BUY if pos.direction==POSITION_DIRECTION.SHORT else SIDE.SELL
            # 取值需要低于当前行情，以保证成交
            price = self.get_last_price(order_book_id)
            if side==SIDE.SELL:
                close_price = int(price - price * 0.01)
            else:
                close_price = int(price + price * 0.01)
            quantity = pos.quantity
            if quantity==0:
                continue
            # 根据持仓标志，决定发送平仓还是平今指令
            if pos.today_pos:
                position_effect = POSITION_EFFECT.CLOSE_TODAY
            else:
                position_effect = POSITION_EFFECT.CLOSE
            
            # 单笔最大数量限制
            singel_quantity_limit = 300
            order_time = quantity//singel_quantity_limit + 1
            for _ in range(order_time):
                if quantity>singel_quantity_limit:
                    single_quantity = singel_quantity_limit   
                else:
                    single_quantity =  quantity          
                close_reason = SellReason.FORCE_CLOSE.value
                order = self.create_order(order_book_id, single_quantity, side,close_price,position_effect=position_effect,close_reason=close_reason)
                self.context.get_trade_proxy().submit_order(order)
                quantity = quantity - single_quantity
            
    def query_position(self):
        
        positions = self.get_positions()
        logger.info("positions number:{}".format(len(positions)))
        for pos in positions:
            logger.info("pos:{}".format(pos))
  
    def query_account(self):
        
        ctp_trade_proxy = self.context.get_trade_proxy()
        account = ctp_trade_proxy.query_account_info()
        print("account:{}".format(account))
          
    def query_trade(self,order_code=""):
        
        ctp_trade_proxy = self.context.get_trade_proxy()
        orders = ctp_trade_proxy.query_order_info(order_code)
        logger.info("orders number:{}".format(len(orders)))
        for order in orders:
            logger.info("order:{}".format(order))      
            
    def query_order_info(self,order_code="",ref_order_id=None):
        """取得指定日期的订单"""
        
        ctp_trade_proxy = self.context.get_trade_proxy()
        cur_date = self.context.now.date()
        # orders = ctp_trade_proxy.get_day_orders(cur_date,order_code=order_code)
        orders = ctp_trade_proxy.query_order_info(order_code)
        order_rtn = []
        for order in orders:
            # 如果指定引用订单号，则需要匹配
            if ref_order_id is not None and order.secondary_order_id!=ref_order_id:
                continue
            order_rtn.append(order)
            logger.info("order in query:{}".format(order))      
        return order_rtn       
                               
    def open_trade_order(self,order_book_id,side=SIDE.BUY,quantity=10):
        """开仓指定的品种"""

        price = self.get_last_price(order_book_id)
        if side==SIDE.SELL:
            open_price = int(price - price * 0.03)
        else:
            open_price = int(price + price * 0.03)
        quantity = quantity        
        order = self.create_order(order_book_id, quantity, side,open_price,position_effect=POSITION_EFFECT.OPEN)
        self.context.get_trade_proxy().submit_order(order)  
  
    def clear_order(self):
        """清空所有未执行订单"""     
        
        cur_date = self.context.now.date()
        ctp_trade_proxy = self.context.get_trade_proxy()
        ctp_orders = ctp_trade_proxy.query_order_info("")
        orders = self.sync_orders(cur_date,(None,None,ctp_orders))   
        for order in orders:
            # 针对未完成的进行撤单
            if order.status==ORDER_STATUS.ACTIVE:
                self.cancel_order(order,need_ref=False)
        
        
