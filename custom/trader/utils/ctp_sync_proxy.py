import asyncio
import time

class CtpSyncProxy(object):
    """CTP交易流程中异步和同步的处理"""
    
    def __init__(self,caller):
        self.caller = caller
        self.results_content = {}

    def process_qry_result(self,qry_bunder,results):
        """接收处理异步结果,qry_bunder为字符串，表示异步结果对应的调用者"""
        
        if qry_bunder in self.results_content:
            # 如果结果集里有同类数据，则需要检查问题
            raise Exception("same content:{}".format(qry_bunder))
        
        # 放入结果集合中，以提供给异步数据处理程序使用
        self.results_content = results
        
    async def async_func(self,func_name,args=None):
        """异步转同步，无限循环体内检查数据是否到达，到达后转同步"""
        
        if args is None:
            self.caller.__getattribute__(func_name)
        else:
            self.caller.__getattribute__(func_name)(args)
        counter = 0
        while True:
            # 这里调用方法名需要和绑定的回调类别名一致
            if func_name in self.results_content:
                # 获取数据的同时，删除结果集对应内容
                results = self.results_content.pop(func_name)
                return results
            time.sleep(1)
            counter += 1
            # 如果长时间没有结果，则返回空
            if counter>10:
                return None

    def qry_sync_func(self,func_name,args):
        """接受业务调用，并进行异步同步转换"""
        
        result = asyncio.run(self.async_func(func_name,args))
        return result
