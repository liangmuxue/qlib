import datetime as dt
from datetime import datetime,timedelta
import calendar
from chinese_calendar import is_holiday

def is_working_day(day):
    """判断是否节假日"""
    
    if type(day) == str:
        date = datetime.strptime(day,'%Y%m%d').date()   
    else:
        date = day 
    return not is_holiday(date)
     
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


def get_tradedays_dur(start_date,duration):
    '''
    计算指定日期之前(或之后)的工作日
    start:开始日期
    end:结束日期
    '''

    if type(start_date) == str:
        start_date = datetime.strptime(start_date,'%Y%m%d').date()

    target_date = start_date
    counts = 0
    while counts!=duration:
        if is_holiday(target_date) or target_date.weekday()==5 or target_date.weekday()==6:
            if duration>0:
                target_date += timedelta(days=1)
            else:
                target_date -= timedelta(days=1)
            continue
        if duration>0:
            counts += 1
        else:
            counts -= 1
        if duration>0:
            target_date += timedelta(days=1)
        else:
            target_date -= timedelta(days=1)
    return target_date


def date_string_transfer(ori_date,direction=1):
    '''日期格式转换 YYYYMMDD与YYYY-MM-DD互转
        Params：
           direction 转换方向 1 YYYYMMDD转YYYY-MM-DD 2 YYYY-MM-DD转YYYYMMDD
    '''
    
    if direction==1:
        target_date = ori_date[:4] + "-" + ori_date[4:6] + "-" + ori_date[6:]
    else:
        arr = ori_date.split("-")
        target_date = arr[0] + arr[1] + arr[2]
    return target_date

def get_first_and_last_day(year,month):
    """取得每个月第一天和最后一天"""
    
    weekDay,monthCountDay = calendar.monthrange(year,month)
    firstDay = dt.date(year,month,day=1)
    lastDay = dt.date(year,month,day=monthCountDay)
    return firstDay,lastDay


def get_tradedays(start,end):
    '''
    计算指定日期之间的工作日
    start:开始日期
    end:结束日期
    '''

    cal_list = []
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
        cal_list.append(datetime.strftime(start,'%Y%m%d'))
        counts += 1
        start += timedelta(days=1)
    return cal_list