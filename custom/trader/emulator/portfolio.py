from datetime import date
from collections.abc import Mapping
from typing import Optional, Dict, List, Tuple, Union
import six
import numpy as np
import pandas as pd
import pickle
import os
from itertools import chain

from rqalpha.portfolio.account import Account
from rqalpha.portfolio.position import Position
from rqalpha.model import Order
from rqalpha.environment import Environment
from rqalpha.const import POSITION_DIRECTION, POSITION_EFFECT, DEFAULT_ACCOUNT_TYPE
from trader.emulator.futures_backtest_strategy import POS_COLUMNS

ACCOUNT_COLUMNS = ['id','cash','type','financing_rate']
POSITION_COLUMNS = ['account_id'] + POS_COLUMNS

class SimOrder(Order):
    
    def set_status(self,status):
        self._status = status
        
    def set_frozen_price(self,price):
        self._frozen_price = price
        
    
class SimPosition(object):
    __repr_properties__ = (
        "order_book_id", "direction", "quantity", "today_pos", "trading_pnl", "position_pnl", "last_price"
    )

    def __init__(self, order_book_id, direction, init_quantity=0, init_price=None,today_pos=False):
        self._env = Environment.get_instance()

        self._order_book_id = order_book_id
        self._direction = direction
        self._today_pos = today_pos

        self._quantity = init_quantity
        self._old_quantity = init_quantity
        self._logical_old_quantity = 0

        self._avg_price: float = init_price or 0
        self._trade_cost: float = 0
        self._transaction_cost: float = 0
        self._prev_close: Optional[float] = init_price
        self._last_price: Optional[float] = init_price

        self._direction_factor = 1 if direction == POSITION_DIRECTION.LONG else -1    

    @property
    def order_book_id(self):
        # type: () -> str
        return self._order_book_id

    @property
    def today_pos(self):
        # type: () -> str
        return self._today_pos
    
    @property
    def direction(self):
        # type: () -> POSITION_DIRECTION
        return self._direction

    @property
    def quantity(self):
        # type: () -> int
        return self._quantity

    @property
    def transaction_cost(self):
        # type: () -> float
        return self._transaction_cost

    @property
    def avg_price(self):
        # type: () -> float
        return self._avg_price

    @property
    def trading_pnl(self):
        # type: () -> float
        trade_quantity = self._quantity - self._logical_old_quantity
        return (trade_quantity * self.last_price - self._trade_cost) * self._direction_factor

    @property
    def position_pnl(self):
        # type: () -> float
        return self._logical_old_quantity * (self.last_price - self.prev_close) * self._direction_factor

    @property
    def pnl(self):
        # type: () -> float
        """
        返回该持仓的累积盈亏
        """
        return (self.last_price - self.avg_price) * self._quantity * self._direction_factor

    @property
    def market_value(self):
        # type: () -> float
        return self.last_price * self._quantity if self._quantity else 0

    @property
    def equity(self):
        # type: () -> float
        return self.last_price * self._quantity if self._quantity else 0
    
    def __repr__(self):
        fmt_str = "{}({})".format("SimPosition", ", ".join((str(p) + "={}") for p in self.__repr_properties__))
        return fmt_str.format(*(getattr(self, p, None) for p in self.__repr_properties__))
            
class SimAccount(Account):
    def __init__(
            self, account_type: str, total_cash: float, init_positions: Dict[str, Tuple[int, Optional[float]]],
            financing_rate: float,id=None
    ):
        self._type = account_type
        self._total_cash = total_cash  # 包含保证金的总资金
        self._env = Environment.get_instance()

        self._positions = {}
        self._backward_trade_set = set()
        self._frozen_cash = 0
        self._pending_deposit_withdraw: List[Tuple[date, float]] = []

        self._cash_liabilities = 0      # 现金负债

        self.register_event()

        self._management_fee_calculator_func = lambda account, rate: account.total_value * rate
        self._management_fee_rate = 0.0
        self._management_fees = 0.0

        # 融资利率/年
        self._financing_rate = financing_rate

        for order_book_id in init_positions.keys():
            self._positions[order_book_id] = init_positions[order_book_id]
        self.id = id
        
    def get_or_create_pos(
            self,
            order_book_id: str,
            direction: Union[POSITION_DIRECTION, str],
            init_quantity: float = 0,
            init_price : Optional[float] = None
    ) -> SimPosition:
        if order_book_id not in self._positions:
            pos = SimPosition(order_book_id, direction, init_quantity, init_price)
            if not init_price:
                last_price = self._env.get_last_price(order_book_id)
                pos.update_last_price(last_price)            
            self._positions[order_book_id] = pos
            positions = pos
        else:
            positions = self._positions[order_book_id]
        return positions
            
    def apply_trade(self, trade, order=None):
        # Do Nothing
        return
        
class Portfolio(object):
    
    def __init__(
            self,
            starting_cash: Dict[str, float],
            init_positions: List[Tuple[str, int]],
            financing_rate: float,  
            trade_date=None,
            data_proxy=None,
            persis_path=None     
    ):    
        account_args = {}
        account_args[DEFAULT_ACCOUNT_TYPE.FUTURE] = {
            "account_type": DEFAULT_ACCOUNT_TYPE.FUTURE.name, "total_cash": starting_cash, "init_positions": {}, "financing_rate": financing_rate
        }        
        for pos in init_positions:
            order_book_id = pos.order_book_id
            account_type = self.get_account_type(order_book_id)
            if account_type in account_args:
                account_args[account_type]["init_positions"][order_book_id] = pos
        
        self._accounts = {}      
        self._accounts = {account_type: SimAccount(**args) for account_type, args in account_args.items()}
        self._static_unit_net_value = 1
        # self._units = sum(account.total_value for account in six.itervalues(self._accounts))
        self.data_proxy = data_proxy
        self.persis_filepath = persis_path

    @classmethod
    def load_from_storage(cls,persis_path,trade_date=None,data_proxy=None):
        """从存储加载"""
        
        starting_cash = 0
        self._accounts = {}
        account_file_path = Portfolio.get_persis_account_path(persis_path)
        position_file_path = Portfolio.get_persis_position_path(persis_path)
        
        with open(account_file_path, "rb") as fin:
            account_list = pickle.load(fin)         
        with open(position_file_path, "rb") as fin:
            position_list = pickle.load(fin)        
        pos_list_total = []       
        for account in account_list:
            self.financing_rate = account['financing_rate']
            starting_cash += account.total_cash
            position_inner = position_list[position_list['account_id']==account['id']]
            # 初始化当前账号下的持仓信息
            pos_list = []
            for index,row in position_inner.iterrows():
                sim_pos = SimPosition(row['order_book_id'].values[0], row['direction'].values[0], init_quantity=row['quantity'].values[0], init_price=row['price'].values[0])
                pos_list.append(sim_pos)
            acc = SimAccount(account['type'],account['cash'],account['financing_rate'],pos_list,id=account['id'])
            self._accounts[account['type']] = acc
            pos_list_total += pos_list
            
        return cls(starting_cash,pos_list_total,self.financing_rate,data_proxy=data_proxy,trade_date=trade_date,persis_path=persis_path)

    @classmethod
    def load_from_ctp(cls,ctp_data,trade_date=None,data_proxy=None):
        """从CTP服务端数据对接"""
        
                                      
    def save_to_storage(self):
        """保存到存储"""
        
        # 保存账户信息
        account_list = []
        for account in self._accounts:
            account_list.append([account.id,account.type,account.total_cash,account.financing_rate])
        account_list = pd.DataFrame(np.array(account_list),columns=ACCOUNT_COLUMNS) 
        file_path = Portfolio.get_persis_account_path(self.persis_filepath)
        with open(file_path, "wb") as fout:
            pickle.dump(account_list, fout)              
        
        # 保存持仓信息
        postions_list = []
        for account in self._accounts:
            postions = account.get_positions()
            for pos in postions:
                pos_item = [account.id,pos.order_book_id,pos.quantity,pos.side,pos.direction,pos.avg_price,pos.datetime,pos.order_id]
                postions_list.append(pos_item)
        
        postions_list = pd.DataFrame(np.array(postions_list),columns=POSITION_COLUMNS)    
        file_path = Portfolio.get_persis_position_path(self.persis_filepath)
        with open(self.file_path, "wb") as fout:
            pickle.dump(postions_list, fout)             
        
    @classmethod
    def get_account_type(cls, order_book_id):
        return DEFAULT_ACCOUNT_TYPE.FUTURE
    
    @classmethod
    def get_persis_account_path(cls,persis_path):
        return os.path.join(persis_path,"account")
    @classmethod
    def get_persis_position_path(cls,persis_path):
        return os.path.join(persis_path,"position")
                
    def get_account(self, order_book_id):
        return self._accounts[self.get_account_type(order_book_id)]
    
    @property
    def accounts(self):
        # type: () -> Dict[DEFAULT_ACCOUNT_TYPE, Account]
        """
        账户字典
        """
        return self._accounts

    @property
    def start_date(self):
        """
        [datetime.datetime] 策略投资组合的开始日期
        """
        return Environment.get_instance().config.base.start_date

    @property
    def units(self):
        """
        [float] 份额
        """
        return self._units   
    
    @property
    def total_value(self):
        """
        [float]总权益
        """
        return sum(account.total_value for account in six.itervalues(self._accounts))    
    
    def get_positions(self):
        
        positions = []
        for key in self._accounts.keys():
            value = self._accounts[key]
            for pos_key in value.positions.keys():
                pos = value._positions[pos_key]
                positions.append(pos)
        return positions
    
    
    
    