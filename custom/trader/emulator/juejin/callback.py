# coding=utf-8
"""
回调任务分发
"""
from __future__ import unicode_literals, print_function, absolute_import

from typing import Text, NoReturn

from trader.emulator.juejin.storage import ctx
from gmtrade.constant import CALLBACK_TYPE_EXECRPT, CALLBACK_TYPE_ORDER, CALLBACK_TYPE_CASH, \
    CALLBACK_TYPE_POSITION, CALLBACK_TYPE_ERROR, CALLBACK_TYPE_STOP, CALLBACK_TYPE_TRADE_CONNECTED, \
    CALLBACK_TYPE_ACCOUNTSTATUS, CALLBACK_TYPE_TRADE_DISCONNECTED
from gmtrade.pb.account_pb2 import ExecRpt, Order, Cash, Position, AccountStatus
from gmtrade.pb_to_dict import protobuf_to_dict
from gmtrade.utils import gmtradelogger


def excerpt_callback(data):
    if ctx.on_execution_report_fun is not None:
        excerpt = ExecRpt()
        excerpt.ParseFromString(data)
        if ctx.on_execution_report_fun is not None:
            ctx.on_execution_report_fun(excerpt)
            return


def order_callback(data):
    if ctx.on_order_status_fun is not None:
        order = Order()
        order.ParseFromString(data)
        if ctx.on_order_status_fun is not None:
            ctx.on_order_status_fun(order)
            return


def cash_callback(data):
    # fixme 这里先直接返回
    return
    cash = Cash()
    cash.ParseFromString(data)
    cash = protobuf_to_dict(cash, including_default_value_fields=True)
    account_id = cash['account_id']
    accounts = ctx.accounts
    accounts[account_id].cash = cash


def position_callback(data):
    # fixme 这里先直接返回
    return
    position = Position()
    position.ParseFromString(data)
    position = protobuf_to_dict(position, including_default_value_fields=True)
    symbol = position['symbol']
    side = position['side']
    account_id = position['account_id']
    accounts = ctx.accounts
    position_key = '{}.{}'.format(symbol, side)
    accounts[account_id].inside_positions[position_key] = position

    if not position.get('volume'):
        if accounts[account_id].inside_positions.get(position_key):
            return accounts[account_id].inside_positions.pop(position_key)


def default_err_callback(code, info):
    # type: (Text, Text) -> NoReturn
    gmtradelogger.warning(
        '发生错误, 调用默认的处理函数, error code=%s, info=%s.  你可以在自定义on_error函数接管它.',
        code, info
    )


def err_callback(data):
    """
    遇到错误时回调, 错误代码跟错误信息的对应关系参考: https://www.myquant.cn/docs/cpp/170
    """
    if ctx.on_error_fun is None:
        ctx.on_error_fun = default_err_callback

    try:
        data_unicode = data.decode('utf8')
        sparr = data_unicode.split('|', 1)
        if len(sparr) == 1:
            code, info = "code解析不出来", sparr[0]
        else:
            code, info = sparr
        ctx.on_error_fun(code, info)
    except Exception as e:
        gmtradelogger.exception("字符编码解析错误", e)
        ctx.on_error_fun("1011", data)


def stop_callback(data):
    if ctx.on_shutdown_fun is not None:
        ctx.on_shutdown_fun()

    from gmtrade.api import stop
    print("!~~~~~~~~~~~!停止!~~~~~~~~~~~!")
    stop()


def trade_connected_callback():
    gmtradelogger.info("连接交易服务成功")
    if ctx.on_trade_data_connected_fun is not None:
        ctx.on_trade_data_connected_fun()


def account_status_callback(data):
    if ctx.on_account_status_fun is not None:
        account_status = AccountStatus()
        account_status.ParseFromString(data)
        if ctx.on_account_status_fun is not None:
            ctx.on_account_status_fun(account_status)
            return


def trade_disconnected_callback():
    if ctx.on_trade_data_disconnected_fun is not None:
        ctx.on_trade_data_disconnected_fun()


def callback_controller(msg_type, data):
    """
    回调任务控制器
    """
    try:
        # python 3 传过来的是bytes 类型， 转成str
        if isinstance(msg_type, bytes):
            msg_type = bytes.decode(msg_type)

        if msg_type == CALLBACK_TYPE_ERROR:
            return err_callback(data)

        if msg_type == CALLBACK_TYPE_EXECRPT:
            return excerpt_callback(data)

        if msg_type == CALLBACK_TYPE_ORDER:
            return order_callback(data)

        if msg_type == CALLBACK_TYPE_CASH:
            return cash_callback(data)

        if msg_type == CALLBACK_TYPE_POSITION:
            return position_callback(data)

        if msg_type == CALLBACK_TYPE_STOP:
            return stop_callback(data)

        if msg_type == CALLBACK_TYPE_TRADE_CONNECTED:
            return trade_connected_callback()

        if msg_type == CALLBACK_TYPE_TRADE_DISCONNECTED:
            return trade_disconnected_callback()

        if msg_type == CALLBACK_TYPE_ACCOUNTSTATUS:
            return account_status_callback(data)

        gmtradelogger.warn("没有处理消息:%s的处理函数", msg_type)

    except SystemExit:
        gmtradelogger.info('^^--------SystemExit---------^^')
        from gmtrade.api import stop
        stop()

    except BaseException as e:
        gmtradelogger.exception("^^--------------遇到exception--------------^^")
        from gmtrade.api import stop
        stop()
