import time

from rqalpha.apis import *
import rqalpha
from rqalpha.const import SIDE,ORDER_STATUS
from trader.rqalpha.strategy_class.backtest_base import BaseStrategy,SellReason
from trader.rqalpha.dict_mapping import transfer_order_book_id,transfer_instrument
from trader.utils.date_util import tradedays
from trader.emulator.sim_strategy import SimStrategy
from data_extract.his_data_extractor import PeriodType
from cus_utils.log_util import AppLogger
logger = AppLogger()

class SimBacktestStrategy(SimStrategy):
    """仿真交易策略，分钟级别，继承回测基类"""
    
    def __init__(self,proxy_name="qidian"):
        super().__init__(proxy_name=proxy_name)
    
    def init_env(self):
        # 初始化交易代理对象
        emu_args = self.context.config.mod.ext_emulation_mod.emu_args
        # 根据标志，决定是否清空目录下的历史交易记录
        if emu_args["clear_data"]:
            self.trade_entity.clear_his_data()
         
    def before_trading(self,context):
        """交易前准备"""
        
        super().before_trading(context)
        
    def after_trading(self,context):
        logger.info("after_trading in")
        
    def open_auction(self,context, bar_dict):
        """集合竞价入口"""
        
        super().open_auction(context, bar_dict)
         
    def handle_bar(self,context, bar_dict):
        """主要的算法逻辑入口"""
        
        # 需要符合回测数据时间规范（5分钟数据）
        if not self.ds.valid_bar_date(context.now):
            return     
        # 首先进行撮合，然后进行策略
        env = Environment.get_instance()
        env.broker.trade_proxy.handler_bar(context.now)
        super().handle_bar(context,bar_dict)


    ###############################自有数据逻辑########################################  
    
    def get_candidate_list(self,pred_date,context=None):
        """取得候选列表，重载父类方法"""
        
        candidate_list = context.ml_context.filter_buy_candidate(pred_date)
        # candidate_list = ["000702"]
        
        # 检查是否缺失历史数据,如果缺失则剔除
        filter_list = []
        for item in candidate_list:
            symbol = str(item).zfill(6)
            # 只处理当日有开盘数据的
            if self.has_current_data(pred_date,symbol):        
                filter_list.append(item)
        return filter_list
    
    def has_current_data(self,day,symbol):
        """当日是否开盘交易"""

        env = Environment.get_instance()
        ds = env.data_proxy._data_source
                
        # 直接使用数据源
        try:
            item_df = ds.extractor.load_item_df(symbol,period=PeriodType.MIN5.value,institution=False)
        except Exception as e:
            return False
        item_df["datetime_dt"] = pd.to_datetime(item_df.datetime)
        today_df = item_df[item_df["datetime_dt"].dt.strftime("%Y%m%d").astype(int)==day]
        if today_df.shape[0]==0:
            return False          
        return True
    
    def get_last_price(self,order_book_id):
        """重载，这里为取得最近分钟行情"""
        
        env = Environment.get_instance()
        market_price = env.get_last_price(order_book_id)
        return market_price 
    
    