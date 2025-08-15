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
from rqalpha.model import Order
from rqalpha.environment import Environment
from rqalpha.const import POSITION_DIRECTION, ORDER_STATUS, DEFAULT_ACCOUNT_TYPE,ORDER_TYPE
from rqalpha.portfolio.position import Position, PositionProxyDict
from trader.emulator.futures_backtest_strategy import POS_COLUMNS

ACCOUNT_COLUMNS = ['id','cash','type','financing_rate']
POSITION_COLUMNS = ['account_id'] + POS_COLUMNS

class SimOrder(Order):
    
    def set_status(self,status):
        self._status = status
        
    def set_frozen_price(self,price):
        self._frozen_price = price

    def set_trading_dt(self,trading_dt):
        self._trading_dt = trading_dt
        
    def set_filled_quantity(self,filled_quantity):
        self._filled_quantity = filled_quantity
                
class SimPosition(object):
    __repr_properties__ = (
        "order_book_id", "direction", "quantity","multiplier", "today_pos", "margin", "pnl", "position_cost", "avg_price","last_price"
    )

    def __init__(self, order_book_id, direction, init_quantity=0, init_price=None,margin=0,position_cost=0,multiplier=0,today_pos=False):
        self._env = Environment.get_instance()

        self._order_book_id = order_book_id
        self._direction = direction
        self._today_pos = today_pos

        self._quantity = init_quantity
        self._old_quantity = init_quantity
        self._logical_old_quantity = 0

        self._multiplier = multiplier
        self._trade_cost: float = 0
        self._margin = margin
        self._position_cost = position_cost
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
    def last_price(self):
        # type: () -> float
        return self._last_price
    
    @property
    def multiplier(self):
        """对应品种的乘数"""
        return self._multiplier

    @property
    def margin(self):
        """保证金"""
        return self._margin

    @property
    def position_cost(self):
        """持仓成本"""
        return self._position_cost
    
    @property
    def pnl(self):
        """返回该持仓的累积盈亏,当前价格*数量减去持仓成本，并考虑多空"""
        return (self.last_price * self._quantity * self._multiplier - self._position_cost) * self._direction_factor

    @property
    def avg_price(self):
        """持仓均价,使用持仓成本除以成交量来反向生成"""
        
        return self._position_cost/self._quantity/self._multiplier
    
    @property
    def market_value(self):
        """市值"""
        return self.last_price * self._quantity if self._quantity else 0

    @property
    def equity(self):
        # type: () -> float
        return self.last_price * self._quantity if self._quantity else 0
    
    def __repr__(self):
        fmt_str = "{}({})".format("SimPosition", ", ".join((str(p) + "={}") for p in self.__repr_properties__))
        return fmt_str.format(*(getattr(self, p, None) for p in self.__repr_properties__))
            
class SimAccount():
    def __init__(
            self, account_type: str, total_cash: float, frozen: float, margin: float, init_positions: Dict[str, Tuple[int, Optional[float]]],
            financing_rate: float,id=None
    ):
        self._type = account_type
        self._total_cash = total_cash  # 包含保证金的总资金
        self._env = Environment.get_instance()

        self._positions = {}
        self._backward_trade_set = set()
        self._frozen_cash = frozen
        self._margin = margin
        self._pending_deposit_withdraw: List[Tuple[date, float]] = []

        self._cash_liabilities = 0      # 现金负债
        
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
            init_quantity=0,
            init_price=0,
            margin=0
    ) -> SimPosition:
        if order_book_id not in self._positions:
            pos = SimPosition(order_book_id, direction, init_quantity, init_price,margin=margin)
            if not init_price:
                last_price = self._env.get_last_price(order_book_id)
                pos.update_last_price(last_price)            
            self._positions[order_book_id] = pos
            positions = pos
        else:
            positions = self._positions[order_book_id]
        return positions

    @property
    def positions(self):
        return PositionProxyDict(self._positions)
    
    @property
    def margin(self) -> float:
        """总保证金"""
        return self._margin
    
    def set_margin(self,margin):
        self._margin = margin

    @property
    def frozen_cash(self) -> float:
        """冻结资金"""
        return self._frozen_cash

    @property
    def cash(self) -> float:
        """可用资金"""
        
        return self._total_cash - self._margin - self._frozen_cash
        
    def set_frozen_cash(self,frozen_cash):
        self._frozen_cash = frozen_cash

    def get_positions(self):
        """获取所有持仓对象列表"""
        
        for position in self._iter_pos():
            if position.quantity == 0 and position.equity == 0:
                continue
            yield position
                               
    def apply_trade(self, trade, order=None):
        # Do Nothing
        return
        
class Portfolio(object):
    
    def __init__(
            self,
            balance,
            frozen,
            margin,
            init_positions,
            financing_rate,  
            trade_date=None,
            data_proxy=None,
            persis_path=None     
    ):    
        starting_cash = balance - frozen - margin
        account_args = {}
        account_args[DEFAULT_ACCOUNT_TYPE.FUTURE] = {
            "account_type": DEFAULT_ACCOUNT_TYPE.FUTURE.name, "total_cash": starting_cash, "frozen":frozen, "margin":margin,
            "init_positions": {}, "financing_rate": financing_rate
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

    @property
    def cash(self):
        """
        [float] 可用资金
        """
        return sum(account.cash for account in six.itervalues(self._accounts))
        
    def get_positions(self):
        
        positions = []
        for key in self._accounts.keys():
            value = self._accounts[key]
            for pos_key in value.positions.keys():
                pos = value._positions[pos_key]
                positions.append(pos)
        return positions
    
    
    
    