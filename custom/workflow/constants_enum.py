"""工作流涉及的枚举常量"""

from enum import Enum, unique

@unique
class WorkflowType(Enum):
    """工作流类别，1回测 2实盘"""
    backtest = 1 
    offer = 0

@unique
class WorkflowStatus(Enum):
    """工作流状态，0 未执行 1 运行中 2 运行失败 3 运行成功"""
    created = 0 
    running = 1  
    fail = 2 
    success = 3   
    
@unique
class WorkflowSubStatus(Enum):
    """子工作流状态，0 未执行 1 运行中 2 运行失败 3 运行成功"""
    created = 0 
    running = 1  
    fail = 2 
    success = 3      
    
@unique
class FrequencyType(Enum):
    """任务频次类别，1 日 2 周 3 月 4 季度 5 年"""
    
    DAY = 1 
    WEEK = 2  
    MONTH = 3 
    QUARTER = 4      
    YEAR = 5    
    
@unique
class LocalDataSourceType(Enum):
    """本地数据存储类别，1 qlib 2 pandas dataframe"""
    
    qlib = 1 
    dataframe = 2     
        