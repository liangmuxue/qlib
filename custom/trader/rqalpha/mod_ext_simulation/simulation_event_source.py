from datetime import timedelta, datetime, time
import time as tm

from rqalpha.environment import Environment
from rqalpha.interface import AbstractEventSource
from rqalpha.core.events import Event, EVENT
from rqalpha.utils.exception import patch_user_exc
from rqalpha.utils.datetime_func import convert_int_to_datetime
from rqalpha.const import DEFAULT_ACCOUNT_TYPE, INSTRUMENT_TYPE
from rqalpha.utils.i18n import gettext as _

from cus_utils.log_util import AppLogger
logger = AppLogger()

class SimulationEventSource(AbstractEventSource):
    def __init__(self, env):
        # type: (Environment) -> None
        self._env = env
        self._config = env.config
        self._universe_changed = False
        self._env.event_bus.add_listener(EVENT.POST_UNIVERSE_CHANGED, self._on_universe_changed)

        self._get_day_bar_dt = lambda date: date.replace(hour=15, minute=0)
        self._get_after_trading_dt = lambda date: date.replace(hour=15, minute=30)

    def _on_universe_changed(self, _):
        self._universe_changed = True

    def _get_universe(self):
        universe = self._env.get_universe()
        if len(universe) == 0 and DEFAULT_ACCOUNT_TYPE.STOCK.name not in self._config.base.accounts:
            raise patch_user_exc(RuntimeError(_(
                "Current universe is empty. Please use subscribe function before trade"
            )), force=True)
        return universe

    # [BEGIN] minute event helper
    def _get_stock_trading_minutes(self, trading_date):
        trading_minutes = set()

        current_dt = datetime.combine(trading_date, time(9, 31))
        am_end_dt = current_dt.replace(hour=11, minute=30)
        pm_start_dt = current_dt.replace(hour=13, minute=1)
        # 修改收盘时间用于仿真测试
        pm_end_dt = current_dt.replace(hour=18, minute=0)

        delta_minute = timedelta(minutes=1)
        while current_dt <= am_end_dt:
            trading_minutes.add(current_dt)
            current_dt += delta_minute

        current_dt = pm_start_dt
        while current_dt <= pm_end_dt:
            trading_minutes.add(current_dt)
            current_dt += delta_minute
        return trading_minutes

    def _get_future_trading_minutes(self, trading_date):
        trading_minutes = set()
        universe = self._get_universe()
        for order_book_id in universe:
            if self._env.get_account_type(order_book_id) == DEFAULT_ACCOUNT_TYPE.STOCK:
                continue
            trading_minutes.update(self._env.data_proxy.get_trading_minutes_for(order_book_id, trading_date))
        return set([convert_int_to_datetime(minute) for minute in trading_minutes])

    def _get_trading_minutes(self, trading_date):
        trading_minutes = set()
        for account_type in self._config.base.accounts:
            if account_type == DEFAULT_ACCOUNT_TYPE.STOCK:
                trading_minutes = trading_minutes.union(self._get_stock_trading_minutes(trading_date))
            elif account_type == DEFAULT_ACCOUNT_TYPE.FUTURE:
                trading_minutes = trading_minutes.union(self._get_future_trading_minutes(trading_date))
        return sorted(list(trading_minutes))
    # [END] minute event helper

    def events(self, start_date, end_date, frequency):
        trading_dates = self._env.data_proxy.get_trading_dates(start_date, end_date)
        frequency_sim = self._env.config.base.frequency_sim
        if frequency == "1d":
            # 根据起始日期和结束日期，获取所有的交易日，然后再循环获取每一个交易日
            for day in trading_dates:
                date = day.to_pydatetime()
                dt_before_trading = date.replace(hour=0, minute=0)

                dt_bar = self._get_day_bar_dt(date)
                dt_after_trading = self._get_after_trading_dt(date)

                yield Event(EVENT.BEFORE_TRADING, calendar_dt=dt_before_trading, trading_dt=dt_before_trading)
                yield Event(EVENT.OPEN_AUCTION, calendar_dt=dt_before_trading, trading_dt=dt_before_trading)
                yield Event(EVENT.BAR, calendar_dt=dt_bar, trading_dt=dt_bar)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt_after_trading, trading_dt=dt_after_trading)
        elif frequency == '1m' and not frequency_sim:
            for day in trading_dates:
                before_trading_flag = True
                date = day.to_pydatetime()
                last_dt = None
                done = False

                dt_before_day_trading = date.replace(hour=8, minute=30)

                while True:
                    if done:
                        break
                    exit_loop = True
                    trading_minutes = self._get_trading_minutes(date)
                    for calendar_dt in trading_minutes:
                        if last_dt is not None and calendar_dt < last_dt:
                            continue

                        if calendar_dt < dt_before_day_trading:
                            trading_dt = calendar_dt.replace(year=date.year, month=date.month, day=date.day)
                        else:
                            trading_dt = calendar_dt
                        if before_trading_flag:
                            before_trading_flag = False
                            yield Event(
                                EVENT.BEFORE_TRADING,
                                calendar_dt=calendar_dt - timedelta(minutes=30),
                                trading_dt=trading_dt - timedelta(minutes=30)
                            )
                            yield Event(
                                EVENT.OPEN_AUCTION,
                                calendar_dt=calendar_dt - timedelta(minutes=3),
                                trading_dt=trading_dt - timedelta(minutes=3),
                            )
                        if self._universe_changed:
                            self._universe_changed = False
                            last_dt = calendar_dt
                            exit_loop = False
                            break
                        # yield handle bar
                        yield Event(EVENT.BAR, calendar_dt=calendar_dt, trading_dt=trading_dt)
                    if exit_loop:
                        done = True

                dt = self._get_after_trading_dt(date)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt, trading_dt=dt)
        elif frequency == "tick":
            data_proxy = self._env.data_proxy
            for day in trading_dates:
                date = day.to_pydatetime()
                last_tick = None
                last_dt = None
                dt_before_day_trading = date.replace(hour=8, minute=30)
                while True:
                    for tick in data_proxy.get_merge_ticks(self._get_universe(), date, last_dt):
                        # find before trading time

                        calendar_dt = tick.datetime

                        if calendar_dt < dt_before_day_trading:
                            trading_dt = calendar_dt.replace(year=date.year, month=date.month, day=date.day)
                        else:
                            trading_dt = calendar_dt

                        if last_tick is None:
                            last_tick = tick

                            """
                            这里区分时间主要是为了对其之前，之前对获取tick数据的时间有限制，期货的盘前时间是20:30，股票是09:00。
                            在解除获取tick数据的限制后，股票的tick的开始时间是09:15，而期货则是20:59
                            """
                            if self._env.get_instrument(tick.order_book_id).type == INSTRUMENT_TYPE.FUTURE:
                                yield Event(
                                    EVENT.BEFORE_TRADING,
                                    calendar_dt=calendar_dt - timedelta(minutes=30),
                                    trading_dt=trading_dt - timedelta(minutes=30),
                                )
                            else:
                                yield Event(
                                    EVENT.BEFORE_TRADING,
                                    calendar_dt=calendar_dt - timedelta(minutes=15),
                                    trading_dt=trading_dt - timedelta(minutes=15),
                                )

                        if self._universe_changed:
                            self._universe_changed = False
                            break

                        last_dt = calendar_dt
                        yield Event(EVENT.TICK, calendar_dt=calendar_dt, trading_dt=trading_dt, tick=tick)

                    else:
                        break

                dt = self._get_after_trading_dt(date)
                yield Event(EVENT.AFTER_TRADING, calendar_dt=dt, trading_dt=dt)
        elif frequency == "1m" and frequency_sim:
            # 实时模拟仿真模式,每秒检查，分钟推送
            before_trading_flag = True
            date = trading_dates[0].to_pydatetime()
            last_dt = None
            done = False

            dt_before_day_trading = date.replace(hour=8, minute=30)

            while True:
                if done:
                    break
                exit_loop = True
                trading_minutes = self._get_trading_minutes(date)
                for calendar_dt in trading_minutes:
                    if last_dt is not None and calendar_dt < last_dt:
                        continue

                    if calendar_dt < dt_before_day_trading:
                        trading_dt = calendar_dt.replace(year=date.year, month=date.month, day=date.day)
                    else:
                        trading_dt = calendar_dt
                    if before_trading_flag:
                        before_trading_flag = False
                        yield Event(
                            EVENT.BEFORE_TRADING,
                            calendar_dt=calendar_dt - timedelta(minutes=30),
                            trading_dt=trading_dt - timedelta(minutes=30)
                        )
                        yield Event(
                            EVENT.OPEN_AUCTION,
                            calendar_dt=calendar_dt - timedelta(minutes=3),
                            trading_dt=trading_dt - timedelta(minutes=3),
                        )
                    if self._universe_changed:
                        self._universe_changed = False
                        last_dt = calendar_dt
                        exit_loop = False
                        break
                    # 与当前实际时间比较，如果不一致则需要对齐
                    while True:
                        now_time = datetime.now()
                        # 交易时间大于当前时间，等待
                        if calendar_dt.hour>now_time.hour or (calendar_dt.hour==now_time.hour and calendar_dt.minute>now_time.minute):
                            logger.debug("need wait,calendar_dt.hour:{},calendar_dt.minute:{}".format(calendar_dt.hour,calendar_dt.minute))
                            tm.sleep(3)
                            continue
                        # 交易时间小于当前时间，跳入下一交易时间
                        if calendar_dt.hour>now_time.hour or (calendar_dt.hour==now_time.hour and calendar_dt.minute>now_time.minute):
                            logger.debug("need next,calendar_dt.hour:{},calendar_dt.minute:{}".format(calendar_dt.hour,calendar_dt.minute))
                            break                      
                        # 交易时间等于当前时间，执行bar事件
                        yield Event(EVENT.BAR, calendar_dt=calendar_dt, trading_dt=trading_dt)
                        break
                if exit_loop:
                    done = True

            dt = self._get_after_trading_dt(date)
            yield Event(EVENT.AFTER_TRADING, calendar_dt=dt, trading_dt=dt)            
            
        else:
            raise NotImplementedError(_("Frequency {} is not support.").format(frequency))
