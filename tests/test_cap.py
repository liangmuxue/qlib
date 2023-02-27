import signal,functools
class TimeoutError(Exception):pass 
def time_out(seconds,error_msg='TIME_OUT_ERROR:No connection were found in limited time!'):
    def decorated(func):
        result = ''
        def signal_handler(signal_num,frame):
            global result
            result = error_msg
            raise TimeoutError(error_msg) 
        
        def wrapper(*args,**kwargs): 
            global result
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                result = func(*args,**kwargs) 
            finally:
                signal.alarm(0) 
                print('finish')
                return result
        return functools.wraps(func)(wrapper) #return wrapper 
    return decorated

import time

@time_out(5) 
def func():
    print("g")
    time.sleep(3) 
    print("t")
    return

func()