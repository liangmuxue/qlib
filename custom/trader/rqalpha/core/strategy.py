from functools import wraps

from rqalpha.core.events import EVENT, Event
from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.exception import ModifyExceptionFromType
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.const import EXECUTION_PHASE, EXC_TYPE
from rqalpha.environment import Environment


def run_when_strategy_not_hold(func):
    from rqalpha.utils.logger import system_log

    def wrapper(*args, **kwargs):
        if not Environment.get_instance().config.extra.is_hold:
            return func(*args, **kwargs)
        else:
            system_log.debug(_(u"not run {}({}, {}) because strategy is hold").format(func, args, kwargs))

    return wrapper


class Strategy(object):
    def __init__(self, event_bus, scope, ucontext):
        self._user_context = ucontext
        self._current_universe = set()

        self._init = scope.get('init', None)
        self._handle_bar = scope.get('handle_bar', None)
        self._handle_tick = scope.get('handle_tick', None)
        self._open_auction = scope.get("open_auction", None)
        func_before_trading = scope.get('before_trading', None)
        if func_before_trading is not None and func_before_trading.__code__.co_argcount > 1:
            self._before_trading = lambda context: func_before_trading(context, None)
            user_system_log.warn(_(u"deprecated parameter[bar_dict] in before_trading function."))
        else:
            self._before_trading = func_before_trading
        self._after_trading = scope.get('after_trading', None)

        if self._before_trading is not None:
            event_bus.add_listener(EVENT.BEFORE_TRADING, self.before_trading)
        if self._handle_bar is not None:
            event_bus.add_listener(EVENT.BAR, self.handle_bar)
        if self._handle_tick is not None:
            event_bus.add_listener(EVENT.TICK, self.handle_tick)
        if self._after_trading is not None:
            event_bus.add_listener(EVENT.AFTER_TRADING, self.after_trading)
        if self._open_auction is not None:
            event_bus.add_listener(EVENT.OPEN_AUCTION, self.open_auction)

    @property
    def user_context(self):
        return self._user_context

    def init(self):
        if self._init:
            with ExecutionContext(EXECUTION_PHASE.ON_INIT):
                with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                    self._init(self._user_context)

        Environment.get_instance().event_bus.publish_event(Event(EVENT.POST_USER_INIT))

    @run_when_strategy_not_hold
    def before_trading(self, event):
        with ExecutionContext(EXECUTION_PHASE.BEFORE_TRADING):
            with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                # 植入交易日期，用于后续对照
                self._user_context.trade_date = event.trade_date
                self._before_trading(self._user_context)

    @run_when_strategy_not_hold
    def handle_bar(self, event):
        bar_dict = event.bar_dict
        # with ExecutionContext(EXECUTION_PHASE.ON_BAR):
        #     with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
        #         self._handle_bar(self._user_context, bar_dict)
        self._handle_bar(self._user_context, bar_dict)

    @run_when_strategy_not_hold
    def open_auction(self, event):
        bar_dict = event.bar_dict
        with ExecutionContext(EXECUTION_PHASE.OPEN_AUCTION):
            with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                self._open_auction(self._user_context, bar_dict)

    @run_when_strategy_not_hold
    def handle_tick(self, event):
        tick = event.tick
        with ExecutionContext(EXECUTION_PHASE.ON_TICK):
            with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                self._handle_tick(self._user_context, tick)

    @run_when_strategy_not_hold
    def after_trading(self, event):
        with ExecutionContext(EXECUTION_PHASE.AFTER_TRADING):
            with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                self._after_trading(self._user_context)

    def wrap_user_event_handler(self, handler):
        @wraps(handler)
        def wrapped_handler(event):
            with ExecutionContext(EXECUTION_PHASE.GLOBAL):
                with ModifyExceptionFromType(EXC_TYPE.USER_EXC):
                    return handler(self._user_context, event)
        return wrapped_handler
