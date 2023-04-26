import struct
import sys
import sysconfig

from gmtrade.__version__ import __version__

class ExtContext(object):
    """拓展原Context模式，注意需要单例应用"""

    context = None
    own_caller = None
    
    on_error_fun = None
    on_shutdown_fun = None

    on_execution_report_fun = None
    on_order_status_fun = None
    on_account_status_fun = None

    on_trade_data_connected_fun = None
    on_trade_data_disconnected_fun = None

    token = None  # type: Text
    sdk_lang = "python{}.{}".format(sys.version_info.major, sys.version_info.minor)  # type: Text
    sdk_version = __version__  # type: Text
    sdk_arch = str(struct.calcsize("P") * 8)  # type: Text
    sdk_os = sysconfig.get_platform()  # type: Text

    # 已登陆的所有帐号, 用accountid或account_alias做为key值
    logined_accounts = {}  # type: Dict[Text, Account]
    # 默认帐号
    default_account = None  # type: Union[Account, None]
    # 缓存的订单
    cached_orders = []
    
    
ctx = ExtContext()  