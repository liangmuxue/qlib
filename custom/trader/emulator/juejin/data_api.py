from __future__ import print_function, absolute_import
from gm.api import set_token,history,ADJUST_PREV



if __name__ == "__main__":
    set_token('your token_id')
    
    data = history(symbol='SHSE.600000', frequency='1d', start_time='2020-01-01 09:00:00', end_time='2020-12-31 16:00:00',
                   fields='open,high,low,close', adjust=ADJUST_PREV, adjust_end_time='2020-12-31', df=True)
    print(data)