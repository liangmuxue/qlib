# coding=utf-8
from __future__ import unicode_literals, print_function, absolute_import

import os
import sys
import threading
from datetime import date as Date, datetime as Datetime
from importlib import import_module

import six
from typing import Text, List, Dict, NoReturn, Any, Union, Iterable

from gmtrade.__version__ import __version__
from gmtrade.api.storage import ctx, Account
from gmtrade.csdk.c_sdk import py_gmi_place_order, py_gmi_get_unfinished_orders, py_gmi_get_orders, \
    py_gmi_cancel_all_orders, py_gmi_close_all_positions, py_gmi_cancel_order, py_gmi_get_execution_reports, \
    c_status_fail, py_gmi_strerror, py_gmi_set_token, py_gmi_set_serv_addr, py_gmi_login, \
    py_gmi_set_data_callback, gmi_get_c_version, py_gmi_get_cash, py_gmi_get_positions, py_gmi_start, py_gmi_stop
from gmtrade.enum import OrderQualifier_Unknown, OrderDuration_Unknown
from gmtrade.pb.account_pb2 import ExecRpt, Order, Orders, OrderStyle_Volume, OrderStyle_Value, OrderStyle_Percent, \
    OrderStyle_TargetVolume, OrderStyle_TargetValue, OrderStyle_TargetPercent, ExecRpts, Cash, Cashes, Positions, \
    Position
from gmtrade.pb.trade_pb2 import GetUnfinishedOrdersReq, GetOrdersReq, CancelAllOrdersReq, CloseAllPositionsReq, \
    GetExecrptsReq, GetCashReq, GetPositionsReq
from gmtrade.utils import load_to_list, gmtradelogger

from trader.emulator.juejin.callback import callback_controller

GmDate = Union[Text, Datetime, Date]  # 自定义gm里可表示时间的类型
TextNone = Union[Text, None]  # 可表示str或者None类型
AccountTextNone = Union[Account, Text, None]  # 可表示Account或者Text(account_id or account_alias)或者None类型

running = False
_biglock = threading.Lock()


def set_default_account(acc):
    # type: (Account) -> NoReturn
    """
    设置默认帐号. 如果不设置, 默认第一个成功登陆的Account为默认帐户
    """
    if acc:
        ctx.default_account = acc


def get_version():
    # type: () ->Text
    return __version__


def get_strerror(error_code):
    # type: (int) -> Text
    return py_gmi_strerror(error_code)


def set_token(token):
    # type: (Text) ->NoReturn
    """
    设置用户的token，用于身份认证
    """
    py_gmi_set_token(token)


def set_endpoint(serv_addr="localhost:7001"):
    # type: (Text) -> NoReturn
    """
    设置终端服务地址
    """
    py_gmi_set_serv_addr(serv_addr)


def login(*args):
    """
    登录一个或一组账户。用户应该首先调用login指定要登录的账户，然后才能调用交易业务函数。如果需要接收交易事件，则login应在start函数之前调用
    """
    accounts = []
    for item in args:
        if isinstance(item, Iterable):
            for it in item:
                if isinstance(it, Account):
                    accounts.append(it)
        else:
            if isinstance(item, Account):
                accounts.append(item)
    account_ids = [acc.account_id.strip() for acc in accounts]
    status = py_gmi_login(",".join(account_ids))
    if c_status_fail(status, 'gmi_login'):
        gmtradelogger.warn('account登陆出错, %s', account_ids)
    else:
        for acc in accounts:
            ctx.logined_accounts[acc.account_id] = acc
            if acc.account_alias and acc.account_alias not in ctx.logined_accounts:
                ctx.logined_accounts[acc.account_alias] = acc
            if ctx.default_account is None:
                ctx.default_account = acc


def start(filename=None):
    """
    启动eventloop，接收交易事件并触发回调函数. 返回int值, 0表示成功, 非0为失败
    修改原来的函数绑定模式，改为对象类绑定模式
    """
    
    # 处理用户传入 __file__这个特殊变量的情况
    if filename is None:
        # 认为用户就是一个文件,取当前文件
        filename = sys.argv[0]

    syspathes = set(s.replace('\\', '/') for s in sys.path)
    commonpaths = [os.path.commonprefix([p, filename]) for p in syspathes]
    commonpaths.sort(key=lambda s: len(s), reverse=True)
    maxcommonpath = commonpaths[0]
    filename = filename.replace(maxcommonpath, '')  # type: str
    if filename.startswith('/'):
        filename = filename[1:]

    if filename.endswith(".py"):
        filename = filename[:-3]
    filename = filename.replace("/", ".")
    filename = filename.replace('\\', ".")
    fmodule = import_module(filename)

    # 把gmtrade.api里的所有的符号都导出到当前策略文件(fmodule)的命令空间, 方便使用
    from gmtrade import api
    for name in api.__all__:
        if name not in fmodule.__dict__:
            fmodule.__dict__[name] = getattr(api, name)

    # 调用户文件的init
    ctx.inside_file_module = fmodule

    ctx.on_execution_report_fun = getattr(fmodule, 'on_execution_report', None)
    ctx.on_order_status_fun = getattr(fmodule, 'on_order_status', None)
    ctx.on_account_status_fun = getattr(fmodule, 'on_account_status', None)

    ctx.on_trade_data_connected_fun = getattr(fmodule, 'on_trade_data_connected', None)
    ctx.on_trade_data_disconnected_fun = getattr(fmodule, 'on_trade_data_disconnected', None)

    ctx.on_error_fun = getattr(fmodule, 'on_error', None)
    ctx.on_shutdown_fun = getattr(fmodule, 'on_shutdown', None)

    py_gmi_set_data_callback(callback_controller)  # 设置事件处理的回调函数

    splash_msgs = [
        '-' * 60,
        'python gmtrade sdk version: {}'.format(__version__),
        'c_sdk version: {}'.format(gmi_get_c_version()),
        '-' * 60,
    ]

    print(os.linesep.join(splash_msgs))
    global running
    status = py_gmi_start()  # type: int
    if c_status_fail(status, 'gmi_start'):
        running = False
        return status
    else:
        running = True
    return status


def stop():
    """
    停止eventloop，不再接收交易事件
    """
    global running
    running = False
    py_gmi_stop()
    sys.exit(2)


def get_cash(account=None):
    # type: (Account) -> Union[Cash, None]
    """
    返回指定账户的资金
    """
    req = GetCashReq()
    req.account_id = _get_account_id(account)
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_get_cash(req_b)
    if c_status_fail(status, 'py_gmi_get_cash') or not result:
        return None
    res = Cashes()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return res.data[0]
    else:
        return None


def get_positions(account=None):
    # type: (Account) -> List[Position]
    """
    返回指定账户的全部持仓
    """
    req = GetPositionsReq()
    req.account_id = _get_account_id(account)
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_get_positions(req_b)
    if c_status_fail(status, 'py_gmi_get_positions') or not result:
        return []
    res = Positions()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []


def _inner_place_order(o):
    # type: (Order) ->List[Order]
    """
    下单并返回order的信息. 同步调用, 在下单返回的order 的 client_order_id 将会有值.
    下单出错的话, 返回空的list
    """
    orders = Orders()
    orders.data.extend([o])

    req_b = orders.SerializeToString()
    with _biglock:
        status, result = py_gmi_place_order(req_b)

    if c_status_fail(status, 'py_gmi_place_order') or not result:
        return []

    res = Orders()
    res.ParseFromString(result)

    return [item for item in res.data]


def _get_account_id(val):
    # type: (Union[Text, Account]) ->Text
    if val is None and ctx.default_account is not None:
        return ctx.default_account.account_id

    if isinstance(val, Account):
        for v in six.itervalues(ctx.logined_accounts):
            if v == val:
                return val.account_id

    acc = ctx.logined_accounts.get(val, None)
    # 都没有匹配上, 等着后端去拒绝
    return acc.account_id if acc else val


def account(account_id='', account_alias=''):
    # type: (Text, Text) -> Account
    """
    创建一个账户对象，可以指定account_id或account_alias。账户对象可用于后续的交易API中，表明对哪一个账户做操作
    """
    return Account(account_id, account_alias)


def order_volume(symbol, volume, side, order_type, position_effect,
                 price=0, order_duration=OrderDuration_Unknown, order_qualifier=OrderQualifier_Unknown, account=None):
    # type: (Text, float, int, int, int, float, int, int, AccountTextNone) ->List[Order]
    """
    按指定量委托
    """
    order_style = OrderStyle_Volume
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.volume = volume
    o.price = price
    o.side = side
    o.order_type = order_type
    o.position_effect = position_effect
    o.order_style = order_style
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.account_id = account_id

    return _inner_place_order(o)


def order_value(symbol, value, side, order_type, position_effect,
                price=0, order_duration=OrderDuration_Unknown, order_qualifier=OrderQualifier_Unknown, account=None):
    # type:(Text, float, int, int, int, float, int, int, AccountTextNone) ->List[Order]
    """
    按指定价值委托
    """
    order_style = OrderStyle_Value
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.value = value
    o.price = price
    o.side = side
    o.order_type = order_type
    o.position_effect = position_effect
    o.order_style = order_style
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.account_id = account_id

    return _inner_place_order(o)


def order_percent(symbol, percent, side, order_type, position_effect,
                  price=0, order_duration=OrderDuration_Unknown, order_qualifier=OrderQualifier_Unknown, account=None):
    # type: (Text, float, int, int, int, float, int, int, AccountTextNone)->List[Order]
    """
    按指定比例委托
    """
    order_style = OrderStyle_Percent
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.percent = percent
    o.price = price
    o.side = side
    o.order_type = order_type
    o.position_effect = position_effect
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.order_style = order_style
    o.account_id = account_id

    return _inner_place_order(o)


def order_target_volume(symbol, volume, position_side, order_type, price=0, order_duration=OrderDuration_Unknown,
                        order_qualifier=OrderQualifier_Unknown, account=None):
    # type: (Text, float, int, int, float, int, int, AccountTextNone) ->List[Order]
    """
    调仓到目标持仓量
    """
    order_style = OrderStyle_TargetVolume
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.target_volume = volume
    o.price = price
    o.position_side = position_side
    o.order_type = order_type
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.order_style = order_style
    o.account_id = account_id

    return _inner_place_order(o)


def order_target_value(symbol, value, position_side, order_type, price=0,
                       order_duration=OrderDuration_Unknown, order_qualifier=OrderQualifier_Unknown, account=None):
    # type: (Text, float, int, int, float, int, int, AccountTextNone) ->List[Order]
    """
    调仓到目标持仓额
    """
    order_style = OrderStyle_TargetValue
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.target_value = value
    o.price = price
    o.position_side = position_side
    o.order_type = order_type
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.order_style = order_style
    o.account_id = account_id

    return _inner_place_order(o)


def order_target_percent(symbol, percent, position_side, order_type, price=0, order_duration=OrderDuration_Unknown,
                         order_qualifier=OrderQualifier_Unknown, account=None):
    # type: (Text, float, int, int, float, int, int, AccountTextNone) ->List[Order]
    """
    调仓到目标持仓比例
    """
    order_style = OrderStyle_TargetPercent
    account_id = _get_account_id(account)

    o = Order()
    o.symbol = symbol
    o.target_percent = percent
    o.price = price
    o.position_side = position_side
    o.order_type = order_type
    o.order_qualifier = order_qualifier
    o.order_duration = order_duration
    o.order_style = order_style
    o.account_id = account_id

    return _inner_place_order(o)


def get_unfinished_orders(account=None):
    # type: (AccountTextNone)->List[Order]
    """
    返回指定账户的日内未完成委托
    """

    req = GetUnfinishedOrdersReq()
    req.account_id = _get_account_id(account)
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_get_unfinished_orders(req_b)
    if c_status_fail(status, 'py_gmi_get_unfinished_orders') or not result:
        return []

    res = Orders()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []


def get_orders(account=None):
    # type: (AccountTextNone) ->List[Order]
    """
    返回指定账户的日内委托
    """
    req = GetOrdersReq()
    req.account_id = _get_account_id(account)
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_get_orders(req_b)
    if c_status_fail(status, 'py_gmi_get_orders') or not result:
        return []

    res = Orders()
    res.ParseFromString(result)

    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []


def order_cancel_all(account=None):
    # type: (AccountTextNone) ->NoReturn
    """
    取消该帐户的所有委托
    """
    req = CancelAllOrdersReq()
    req.account_ids.extend([_get_account_id(account)])
    req_b = req.SerializeToString()
    py_gmi_cancel_all_orders(req_b)


def order_close_all(account=None):
    # type: (AccountTextNone) ->List[Order]
    """
    平当前帐号所有可平持仓
    """
    req = CloseAllPositionsReq()
    req.account_ids.extend([_get_account_id(account)])
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_close_all_positions(req_b)
    if c_status_fail(status, 'py_gmi_close_all_positions') or not result:
        return []

    res = Orders()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []


def order_cancel(wait_cancel_orders):
    # type: (Union[Dict[Text,Any], List[Dict[Text, Any]]]) ->NoReturn
    """
    撤销委托. 传入单个字典. 或者list字典. 每个字典包含key: cl_ord_id, account_id. 如果不包含account_id, 则取默认的account_id
    """
    wait_cancel_orders = load_to_list(wait_cancel_orders)
    default_account_id = None
    if ctx.default_account is not None:
        default_account_id = ctx.default_account.account_id

    orders = Orders()
    for wait_cancel_order in wait_cancel_orders:
        order = orders.data.add()
        order.cl_ord_id = wait_cancel_order.get('cl_ord_id')
        order.account_id = wait_cancel_order.get('account_id', default_account_id)

    req_b = orders.SerializeToString()
    py_gmi_cancel_order(req_b)


def order_batch(order_infos, account=None):
    # type: (Iterable[Dict[Text, Any]], AccountTextNone) -> List[Order]
    """
    批量委托接口
    """
    orders = Orders()
    for order_info in order_infos:
        order_info['account_id'] = _get_account_id(account)
        order = orders.data.add()
        [setattr(order, k, order_info[k]) for k in order_info]
    req_b = orders.SerializeToString()
    with _biglock:
        status, result = py_gmi_place_order(req_b)
    if c_status_fail(status, 'py_gmi_place_order') or not result:
        return []

    res = Orders()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []


def get_execution_reports(account=None):
    # type: (AccountTextNone) -> List[ExecRpt]
    """
    返回指定账户的日内成交
    """
    req = GetExecrptsReq()
    req.account_id = _get_account_id(account)
    req_b = req.SerializeToString()
    with _biglock:
        status, result = py_gmi_get_execution_reports(req_b)
    if c_status_fail(status, 'py_gmi_get_execution_reports') or not result:
        return []

    res = ExecRpts()
    res.ParseFromString(result)
    if len(res.data) > 0:
        return [item for item in res.data]
    else:
        return []
