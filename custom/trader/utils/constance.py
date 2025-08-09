

"""订单交易涉及的枚举常量"""

from enum import Enum, unique



@unique
class OrderOffsetType(Enum):
    """开仓平仓标志 0,开仓;1,平仓;2,强平;3,平今;4,平昨;5,强减"""
    
    OPEN = "0"
    CLOSE = "1"
    CLOSE_TODAY = "3"
    CLOSE_PREV = "5"
    
@unique
class CtpQueryType(Enum):
    """CTP查询请求名称列表"""
    
    QryAccount = "QryAccount"
    QryPosition = "QryPosition"
    QryOrder = "QryOrder"

@unique
class OrderStatusType(Enum):
    """订单状态，a:已提交 0全部成交 1部分成交 3未成交 5已撤销"""
    
    NotBegin = "-1" 
    HasCommit = "a" 
    AllTraded = "0" 
    PartTradedQueueing = "1"
    UnClosed = "3"
    Canceled = "5"


@unique
class CtpSyncFlag(Enum):
    """本地存储与CTP环境同步后的状态标志"""
    
    ACCORD = 0
    NOT_EXISTS = 1    