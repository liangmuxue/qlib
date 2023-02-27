from datetime import datetime,timedelta
from chinese_calendar import is_holiday
    
def tradedays(start,end):
    '''
    计算两个日期间的工作日数量
    start:开始日期
    end:结束日期
    '''

    if type(start) == str:
        start = datetime.strptime(start,'%Y%m%d').date()
    if type(end) == str:
        end = datetime.strptime(end,'%Y%m%d').date()
    if start > end:
        start,end = end,start
        
    counts = 0
    while True:
        if start > end:
            break
        if is_holiday(start) or start.weekday()==5 or start.weekday()==6:
            start += timedelta(days=1)
            continue
        counts += 1
        start += timedelta(days=1)
    return counts