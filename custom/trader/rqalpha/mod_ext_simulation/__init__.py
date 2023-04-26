
import click

from rqalpha import cli

__config__ = {
    # 开启信号模式：该模式下，所有通过风控的订单将不进行撮合，直接产生交易
    "signal": False,
    # 撮合方式，其中：
    #   日回测的可选值为 "current_bar"|"vwap"（以当前 bar 收盘价｜成交量加权平均价撮合）
    #   分钟回测的可选值有 "current_bar"|"next_bar"|"vwap"（以当前 bar 收盘价｜下一个 bar 的开盘价｜成交量加权平均价撮合)
    #   tick 回测的可选值有 "last"|"best_own"|"best_counterparty"（以最新价｜己方最优价｜对手方最优价撮合）和 "counterparty_offer"（逐档撮合）
    #   matching_type 为 None 则表示根据回测频率自动选择。日/分钟回测下为 current_bar , tick 回测下为 last
    "matching_type": None,
    # 开启对于处于涨跌停状态的证券的撮合限制
    "price_limit": True,
    # 开启对于对手盘无流动性的证券的撮合限制（仅在 tick 回测下生效）
    "liquidity_limit": False,
    # 开启成交量限制
    #   开启该限制意味着每个 bar 的累计成交量将不会超过该时间段内市场上总成交量的一定比值（volume_percent）
    #   开启该限制意味着每个 tick 的累计成交量将不会超过当前tick与上一个tick的市场总成交量之差的一定比值
    "volume_limit": True,
    # 每个 bar/tick 可成交数量占市场总成交量的比值，在 volume_limit 开启时生效
    "volume_percent": 0.25,
    # 滑点模型，可选值有 "PriceRatioSlippage"（按价格比例设置滑点）和 "TickSizeSlippage"（按跳设置滑点）
    #    亦可自己实现滑点模型，选择自己实现的滑点模型时，此处需传入包含包和模块的完整类路径
    #    滑点模型类需继承自 rqalpha.mod.rqalpha_mod_sys_simulation.slippage.BaseSlippage
    "slippage_model": "PriceRatioSlippage",
    # 设置滑点值，对于 PriceRatioSlippage 表示价格的比例，对于 TickSizeSlippage 表示跳的数量
    "slippage": 0,
    # 开启对于当前 bar 无成交量的标的的撮合限制（仅在日和分钟回测下生效）
    "inactive_limit": True,
    # 账户每日计提的费用，需按照(账户类型，费率)的格式传入，例如[("STOCK", 0.0001), ("FUTURE", 0.0001)]
    "management_fee": [],
}


def load_mod():
    from trader.rqalpha.mod_ext_simulation.mod import SimulationMod
    return SimulationMod()


"""
注入 --signal option: 实现信号模式回测
注入 --slippage option: 实现设置滑点
注入 --commission-multiplier options: 实现设置手续费乘数
注入 --matching-type: 实现选择回测引擎
"""
cli_prefix = "mod__sys_simulation__"

cli.commands['run'].params.append(
    click.Option(
        ('--signal', cli_prefix + "signal"),
        is_flag=True, default=None,
        help="[sys_simulation] exclude match engine",
    )
)

cli.commands['run'].params.append(
    click.Option(
        ('-sp', '--slippage', cli_prefix + "slippage"),
        type=click.FLOAT,
        help="[sys_simulation] set slippage"
    )
)

cli.commands['run'].params.append(
    click.Option(
        ('--slippage-model', cli_prefix + "slippage_model"),
        type=click.STRING,
        help="[sys_simulation] set slippage model"
    )
)

cli.commands['run'].params.append(
    click.Option(
        ('-mt', '--matching-type', cli_prefix + "matching_type"),
        type=click.Choice(
            ['current_bar', 'next_bar', 'last', 'best_own', 'best_counterparty', 'vwap', 'counterparty_offer']),
        help="[sys_simulation] set matching type"
    )
)

cli.commands['run'].params.append(
    click.Option(
        ('--inactive-limit', cli_prefix + "inactive_limit"),
        type=click.BOOL,
        help="[sys_simulation] Limit transaction when volume is 0"
    )
)

cli.commands["run"].params.append(
    click.Option(
        ('--management-fee', cli_prefix + "management_fee",),
        type=click.STRING, nargs=2, multiple=True,
        help="[sys_simulation] Account management rate. eg '--management-fee stock 0.0002' "
    )
)
