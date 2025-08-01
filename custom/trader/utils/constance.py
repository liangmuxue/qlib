

"""订单交易涉及的枚举常量"""

from enum import Enum, unique

@unique
class CtpQueryType(Enum):
    """CTP查询请求名称列表"""
    
    QryAccount = "QryAccount"
    QryPosition = "QryPosition"
    QryOrder = "QryOrder"

@unique
class OrderStatusType(Enum):
    """订单状态，0全部成交 1部分成交 3未成交 5已撤销"""
    
    NotBegin = -1 
    AllTraded = 0 
    PartTradedQueueing = 1
    UnClosed = 3
    Canceled = 5


@unique
class CtpSyncFlag(Enum):
    """本地存储与CTP环境同步后的状态标志"""
    
    ACCORD = 0
    NOT_EXISTS = 1    