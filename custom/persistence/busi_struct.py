from cus_utils.db_accessor import DbAccessor

class BasePersistence(object):
    """数据库持久化处理"""
    
    def __init__(self):
        self.dbaccessor = DbAccessor({})
        
