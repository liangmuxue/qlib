from typing import List, Tuple, Dict
from rqalpha.utils.functools import lru_cache
from itertools import chain

import jsonpickle

from rqalpha.portfolio.account import Account
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.interface import AbstractBroker, Persistable
from rqalpha.utils.i18n import gettext as _
from rqalpha.core.events import EVENT, Event
from rqalpha.const import MATCHING_TYPE, ORDER_STATUS, POSITION_EFFECT, EXECUTION_PHASE, INSTRUMENT_TYPE
from rqalpha.model.order import Order
from rqalpha.environment import Environment
from rqalpha.mod.rqalpha_mod_sys_simulation.simulation_broker import SimulationBroker
from rqalpha.mod.rqalpha_mod_sys_simulation.matcher import DefaultTickMatcher

from trader.rqalpha.mod_ext_simulation.matcher import ExtBarMatcher
from trader.emulator.juejin.trade_proxy_juejin import JuejinTrade
from trader.rqalpha.trade_proxy_rqalpha import RqalphaTrade

class ExtSimulationBroker(SimulationBroker):
    """扩展broker代理，加入仿真环境"""
    
    def __init__(self, env, mod_config):
        super().__init__(env, mod_config)
        self.emu_channel = mod_config.emu_channel
        if mod_config.emu_channel=="juejin":
            emu_args = mod_config.emu_args
            # 掘金环境，初始化植入自身环境上下文
            self.trade_proxy = JuejinTrade(context=self,token=emu_args["token"],
                    end_point=emu_args["end_point"],account_id=emu_args["account_id"],account_alias=emu_args["account_alias"])
        if mod_config.emu_channel=="rqalpha":
            emu_args = mod_config.emu_args
            # RQALPHA环境，初始化植入自身环境上下文
            self.trade_proxy = RqalphaTrade(context=self,account_alias=emu_args["account_alias"])            

    @lru_cache(1024)
    def _get_matcher(self, order_book_id):
        # type: (str) -> AbstractMatcher
        instrument_type = self._env.data_proxy.instrument(order_book_id).type
        try:
            return self._matchers[instrument_type]
        except KeyError:
            if self._env.config.base.frequency == "tick":
                return self._matchers.setdefault(instrument_type, DefaultTickMatcher(self._env, self._mod_config))
            else:
                return self._matchers.setdefault(instrument_type, ExtBarMatcher(self._env, self._mod_config))

    def submit_order(self, order):
        self._check_subscribe(order)
        if order.position_effect == POSITION_EFFECT.MATCH:
            raise TypeError(_("unsupported position_effect {}").format(order.position_effect))
        account = self._env.get_account(order.order_book_id)
        self._env.event_bus.publish_event(Event(EVENT.ORDER_PENDING_NEW, account=account, order=order))
        if order.is_final():
            return
        if order.position_effect == POSITION_EFFECT.EXERCISE:
            return self._open_exercise_orders.append((account, order))
        if ExecutionContext.phase() == EXECUTION_PHASE.OPEN_AUCTION:
            self._open_auction_orders.append((account, order))
            self._open_auction_orders.clear()
        else:
            self._open_orders.append((account, order))
        # 使用具体的仿真（实盘）环境进行下单
        self.trade_proxy.submit_order(order)
        
    def cancel_order(self, order):
        account = self._env.get_account(order.order_book_id)

        self._env.event_bus.publish_event(Event(EVENT.ORDER_PENDING_CANCEL, account=account, order=order))

        order.mark_cancelled(_(u"{order_id} order has been cancelled by user.").format(order_id=order.order_id))
        self.trade_proxy.cancel_order(order)
        try:
            self._open_orders.remove((account, order))
        except ValueError:
            pass

    def before_trading(self, _):
        for account, order in self._open_orders:
            order.active()
            self._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))

    def after_trading(self, __):
        for account, order in self._open_orders:
            order.mark_rejected(_(u"Order Rejected: {order_book_id} can not match. Market close.").format(
                order_book_id=order.order_book_id
            ))
            self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=order))
        self._open_orders = []


    #############################################   回调部分   ########################################################
    def on_bar(self, _):
        for matcher in self._matchers.values():
            matcher.update()
        # self._match()

    def on_tick(self, event):
        tick = event.tick
        self._get_matcher(tick.order_book_id).update()
        self._match(tick.order_book_id)


    def _check_subscribe(self, order):
        if self._env.config.base.frequency == "tick" and order.order_book_id not in self._env.get_universe():
            raise RuntimeError(_("{order_book_id} should be subscribed when frequency is tick.").format(
                order_book_id=order.order_book_id))
