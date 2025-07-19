

"""订单交易涉及的枚举常量"""

from enum import Enum, unique

@unique
class OrderStatusType(Enum):
    """订单状态，0全部成交 1部分成交"""
    
    NotBegin = -1 
    AllTraded = 0 
    PartTradedQueueing = 1

