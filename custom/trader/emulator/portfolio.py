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
from rqalpha.environment import Environment
from rqalpha.const import POSITION_DIRECTION, POSITION_EFFECT, DEFAULT_ACCOUNT_TYPE
from trader.emulator.futures_backtest_strategy import POS_COLUMNS

ACCOUNT_COLUMNS = ['id','cash','type','financing_rate']
POSITION_COLUMNS = ['account_id'] + POS_COLUMNS

class SimPosition(Position):
    
    def __init__(self, order_book_id, direction, init_quantity=0, init_price=None):
        super().__init__(order_book_id, direction, init_quantity=init_quantity, init_price=init_price)
    
class SimAccount(Account):
    def __init__(
            self, account_type: str, total_cash: float, init_positions: Dict[str, Tuple[int, Optional[float]]],
            financing_rate: float,id=None
    ):
        super().__init__(account_type,total_cash,init_positions,financing_rate)  
        self.id = id
        
    def apply_trade(self, trade, order=None):
        # type: (Trade, Optional[Order]) -> None
        if trade.exec_id in self._backward_trade_set:
            return
        order_book_id = trade.order_book_id
        if order and trade.position_effect != POSITION_EFFECT.MATCH:
            if trade.last_quantity != order.quantity:
                self._frozen_cash -= trade.last_quantity / order.quantity * order.init_frozen_cash
            else:
                self._frozen_cash -= order.init_frozen_cash
        if trade.position_effect == POSITION_EFFECT.MATCH:
            delta_cash = self._get_or_create_pos(
                order_book_id, POSITION_DIRECTION.LONG
            ).apply_trade(trade) + self._get_or_create_pos(
                order_book_id, POSITION_DIRECTION.SHORT
            ).apply_trade(trade)
            self._total_cash += delta_cash
        else:
            delta_cash = self._get_or_create_pos(order_book_id, trade.position_direction).apply_trade(trade)
            self._total_cash += delta_cash
        self._backward_trade_set.add(trade.exec_id)
        
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
        last_trading_date = data_proxy.get_previous_trading_date(trade_date)
        for order_book_id, quantity in init_positions:
            account_type = self.get_account_type(order_book_id)
            if account_type in account_args:
                price = data_proxy.get_bar(order_book_id, last_trading_date).close
                account_args[account_type]["init_positions"][order_book_id] = quantity, price
        self._accounts = {account_type: SimAccount(**args) for account_type, args in account_args.items()}
        self._static_unit_net_value = 1
        self._units = sum(account.total_value for account in six.itervalues(self._accounts))
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
        return list(chain(*(a.get_positions() for a in six.itervalues(self._accounts))))    
    
    
    
    
    