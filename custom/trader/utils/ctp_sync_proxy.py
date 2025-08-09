import asyncio
import time
import copy

from cus_utils.log_util import AppLogger
logger = AppLogger()

class CtpSyncProxy(object):
    """CTP交易流程中异步和同步的处理"""
    
    def __init__(self,caller):
        self.caller = caller
        self.results_content = {}

    def process_qry_result(self,qry_bunder,results):
        """接收处理异步结果,qry_bunder为字符串，表示异步结果对应的调用者"""
        
        # logger.debug("process_qry_result in,qry_bunder:{},and results:{}".format(qry_bunder,results))
        if qry_bunder in self.results_content:
            # 有可能分多条返回，在这里累加
            ori_data = self.results_content[qry_bunder]
            ori_data.append(results)
            self.results_content[qry_bunder] = ori_data
        else:
            # 放入结果集合中，以提供给异步数据处理程序使用
            if results is None:
                self.results_content[qry_bunder] = None
            else:
                self.results_content[qry_bunder] = [results]
        
    async def async_func(self,func_name,args=None,wait_time=3,multiple=False):
        """异步转同步，循环体内检查数据是否到达，到达后转同步"""
        
        method = getattr(self.caller.api, func_name)
        # logger.debug("begin call:{}".format(method))
        if args is None:
            method()
        else:
            method(args)
        # logger.debug("end call:{}".format(method))
        counter = 0
        # 根据指定时间，等待后延时获取结果
        while counter<=wait_time:
            time.sleep(1)
            counter += 1
        if func_name not in self.results_content:
            return None
        results = self.results_content.pop(func_name)
        # 如果当前业务函数只返回一条结果，则脱掉外层数组包装，返回对象
        if not multiple and results is not None:
            results = results[0]
        return results

    def qry_sync_func(self,func_name,args=None,wait_time=3,multiple=False):
        """接受业务调用，并进行异步同步转换
            @params:
               func_name 字符串类型 方法名  
               args 方法的参数
               wait_time 回调时需要等待的秒数
               multiple 回调是否包含多条结果
        """
        
        result = asyncio.run(self.async_func(func_name,args=args,wait_time=wait_time,multiple=multiple))
        return result
