import six
from rqalpha.core.events import EVENT
from rqalpha.utils.logger import user_system_log

from rqalpha.interface import AbstractMod
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.exception import patch_user_exc
from rqalpha.const import MATCHING_TYPE, RUN_TYPE

from trader.rqalpha.mod_ext_simulation.simulation_broker import ExtSimulationBroker
from rqalpha.mod.rqalpha_mod_sys_simulation.signal_broker import SignalBroker
from rqalpha.mod.rqalpha_mod_sys_simulation.simulation_event_source import SimulationEventSource

class SimulationMod(AbstractMod):
    def __init__(self):
        self._env = None

    def start_up(self, env, mod_config):
        self._env = env
        self.emu_channel = mod_config.emu_channel
        if env.config.base.run_type == RUN_TYPE.LIVE_TRADING:
            return

        mod_config.matching_type = self.parse_matching_type(mod_config.matching_type, env.config.base.frequency)

        if env.config.base.margin_multiplier <= 0:
            raise patch_user_exc(ValueError(_(u"invalid margin multiplier value: value range is (0, +∞]")))

        if env.config.base.frequency == "tick":
            if mod_config.matching_type not in [
                MATCHING_TYPE.NEXT_TICK_LAST,
                MATCHING_TYPE.NEXT_TICK_BEST_OWN,
                MATCHING_TYPE.NEXT_TICK_BEST_COUNTERPARTY,
                MATCHING_TYPE.COUNTERPARTY_OFFER,
            ]:
                raise RuntimeError(_("Not supported matching type {}").format(mod_config.matching_type))
        else:
            if mod_config.matching_type not in [
                MATCHING_TYPE.NEXT_BAR_OPEN,
                MATCHING_TYPE.VWAP,
                MATCHING_TYPE.CURRENT_BAR_CLOSE,
            ]:
                raise RuntimeError(_("Not supported matching type {}").format(mod_config.matching_type))

        if env.config.base.frequency == "1d" and mod_config.matching_type == MATCHING_TYPE.NEXT_BAR_OPEN:
            mod_config.matching_type = MATCHING_TYPE.CURRENT_BAR_CLOSE
            user_system_log.warn(_(u"matching_type = 'next_bar' is abandoned when frequency == '1d',"
                                   u"Current matching_type is 'current_bar'."))

        if mod_config.signal:
            env.set_broker(SignalBroker(env, mod_config))
        else:
            env.set_broker(ExtSimulationBroker(env, mod_config))

        if mod_config.management_fee:
            env.event_bus.add_listener(EVENT.POST_SYSTEM_INIT, self.register_management_fee_calculator)

        event_source = SimulationEventSource(env)
        env.set_event_source(event_source)

    def tear_down(self, code, exception=None):
        pass

    @staticmethod
    def parse_matching_type(me_str, frequency):
        if me_str is None:
            # None 表示根据 frequency 自动选择
            if frequency in ["1d", "1m"]:
                me_str = "current_bar"
            elif frequency == "tick":
                me_str = "last"
            else:
                raise ValueError("frequency only support ['1d', '1m', 'tick']")

        assert isinstance(me_str, six.string_types)
        me_str = me_str.lower()
        if me_str == "current_bar":
            return MATCHING_TYPE.CURRENT_BAR_CLOSE
        if me_str == "vwap":
            return MATCHING_TYPE.VWAP
        elif me_str == "next_bar":
            return MATCHING_TYPE.NEXT_BAR_OPEN
        elif me_str == "last":
            return MATCHING_TYPE.NEXT_TICK_LAST
        elif me_str == "best_own":
            return MATCHING_TYPE.NEXT_TICK_BEST_OWN
        elif me_str == "best_counterparty":
            return MATCHING_TYPE.NEXT_TICK_BEST_COUNTERPARTY
        elif me_str == "counterparty_offer":
            return MATCHING_TYPE.COUNTERPARTY_OFFER
        else:
            raise NotImplementedError

    def register_management_fee_calculator(self, event):
        management_fee = self._env.config.mod.sys_simulation.management_fee
        accounts = self._env.portfolio.accounts
        for _account_type, v in management_fee:
            _account_type = _account_type.upper()
            if _account_type not in accounts:
                all_account_type = list(accounts.keys())
                raise ValueError(_("NO account_type = ({}) in {}").format(_account_type, all_account_type))
            accounts[_account_type].set_management_fee_rate(float(v))
