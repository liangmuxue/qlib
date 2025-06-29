from rqalpha.apis import *
import rqalpha

from enum import Enum, unique

from trader.rqalpha.ml_context import MlIntergrate
from trader.rqalpha.ml_wf_context import MlWorkflowIntergrate
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.rqalpha.trade_entity import TradeEntity
from trader.utils.date_util import tradedays
from cus_utils.db_accessor import DbAccessor
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
import cus_utils.global_var as global_var
from cus_utils.log_util import AppLogger
logger = AppLogger()

@unique
class SellReason(Enum):
    """卖出类别，1.根据预测数据 2.止盈 3.止损 4.超期卖出"""
    PRED = 1 
    STOP_RAISE = 2
    STOP_FALL = 3
    EXPIRE_DAY = 4

# 交易信息表字段，分别为交易日期，股票代码，成交价格，成交量,总价格，成交状态，订单编号
TRADE_COLUMNS = ["trade_date","instrument","side","price","quantity","total_price","status","order_id"]

        
class BaseStrategy():
    """交易对象处理BASE类"""
    
    def __init__(self,proxy_name="rqalpha"):
        self.instruments_dict = self.create_instruments_dict() 
    
    def create_instruments_dict(self):
        
        dbaccessor = DbAccessor({})
        instruments_dict = {}  
        # 只处理具备数据的股票,因此生成库内股票字典
        result_rows = dbaccessor.do_query("select code,market from instrument_info")
        for row in result_rows:
            instruments_dict[int(row[0])] = {"code":row[0],"market":row[1]}        
        
        return instruments_dict
     
    def logger_debug(self,msg):  
        logger.debug("[{}] {}".format(self.context.now,msg))

    def logger_info(self,msg):  
        logger.info("[{}] {}".format(self.context.now,msg))        
        
    def build_proxy_with_name(self,proxy_name):
        return None
    
    def __build_with_context__(self,context,workflow_mode=False):
        self.context = context
        # 使用rqalpha模式下预设值的参数
        config_path = context.config.extra.context_vars.strategy_class.config_path
        provider_uri = context.config.extra.context_vars.strategy_class.provider_uri
        # 加载qlib上下文  
        if workflow_mode:
            task_config = global_var.get_value("task_config")
            context.ml_context = MlWorkflowIntergrate(task_config=task_config,provider_uri=provider_uri,ext_length=25
                                        ,task_id=context.config.extra.task_id,dump_path=context.config.extra.dump_path)
        else:
            context.ml_context = MlIntergrate(ext_length=25,config_path=config_path,provider_uri=provider_uri)
        self.strategy = context.config.extra.context_vars.strategy
        # 交易对象上下文
        save_path = context.config.extra.report_save_path
        data_save_path = save_path + "/trade_data.csv"
        log_save_path = save_path + "/trade_data_log.csv"
        self.trade_entity = self.create_trade_entity(save_path=data_save_path,log_save_path=log_save_path)
        # 注册订单事件
        context.fired = False
        subscribe_event(EVENT.TRADE, self.on_trade_handler)
        subscribe_event(EVENT.ORDER_CREATION_PASS, self.on_order_handler)     
        subscribe_event(EVENT.ORDER_CREATION_REJECT, self.on_order_handler)    
        subscribe_event(EVENT.ORDER_CANCELLATION_PASS, self.on_order_handler) 
        self.init_env()    
    
    def init_env(self):
        pass
    
    def first_period_of_day(self,dt,frequency=PeriodType.MIN5.value):
        if frequency==PeriodType.MIN5.value:
            if dt.minute==35:
                return True
            return False
        raise NotImplemented
    
    def before_trading(self,context):
        pass

    def handle_bar(self,context, bar_dict):
        pass


    ############################事件注册部分######################################
    def on_trade_handler(self,context, event):
        pass
    
    def on_order_handler(self,context, event):
        pass    