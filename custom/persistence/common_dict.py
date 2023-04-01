from enum import Enum, unique

from .busi_struct import BasePersistence
from cus_utils.db_accessor import DbAccessor

@unique
class CommonDictType(Enum):
    """数据字典类别"""
    
    INDUSTRY_TYPE = "industry_type" # 行业类别

    
class CommonDict(BasePersistence):
    """数据库字典处理"""
    
    def __init__(self):
        
        super().__init__()
    
    def get_dict_by_id(self,dict_id):
        query_sql = "select id,code,value from common_dict where id=%s"
        row = self.dbaccessor.do_query(query_sql, (dict_id))[0] 
        return {"code":row[1],"value":row[2]}  
    
    def get_or_insert_dict(self,common_dict):
        """根据传入参数取得数据库字典信息，如果没有则添加
            Params：
                common_dict 数据字典数据，Dict类型
            Return:
                数据库内查询或者新增的数据字典信息
        """
        
        type = common_dict["type"]
        code = common_dict["code"]
        value = common_dict["value"]
        type_id = self.dbaccessor.do_query("select id from common_dict_type where code=%s", (type))[0][0]
        query_sql = "select id,code,value from common_dict where code=%s and type=%s"
        results = self.dbaccessor.do_query(query_sql, (code,type_id))
        # 如果没有此字典，则插入新的字典数据
        if len(results)==0:
            insert_sql = "insert into common_dict(code,value,type) values(%s,%s,%s)"
            self.dbaccessor.do_inserto_withparams(insert_sql,(code,value,type_id))
            results = self.dbaccessor.do_query(query_sql, (code,type_id))
        row = results[0]
        rtn = {"id":row[0],"code":row[1],"value":row[2]}   
        return rtn 
