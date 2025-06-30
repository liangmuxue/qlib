import time
import datetime as dt
from datetime import datetime,timedelta
import calendar
from chinese_calendar import is_holiday
from dateutil.relativedelta import relativedelta

def get_next_working_day(day):
    """取得指定日期的下一工作日"""
    
    cur_day = day
    while(True):
        next_day = cur_day + timedelta(days=1)
        if is_working_day(next_day):
            return next_day
        cur_day = next_day

def get_previous_day(day):
    """取得指定日期的上一日"""
    
    return day + timedelta(days=-1)
    
def is_working_day(day):
    """判断是否节假日"""
    
    if type(day) == str:
        date = datetime.strptime(day,'%Y%m%d').date()   
    else:
        date = day 
    # 周末倒休的工作日，大陆股市也休息
    if date.weekday()==5 or date.weekday()==6:
        return False
    if is_holiday(date):
        return False    
    return True
     
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
        if len(start_date)==8:
            start_date = datetime.strptime(start_date,'%Y%m%d').date()
        else:
            start_date = datetime.strptime(start_date,'%Y-%m-%d').date()
        
    counts = 0
    # if is_holiday(start_date):  
    #     if duration>0:
    #         counts -= 
    #     else:
    #         counts += 1
    target_date = start_date
      
    while True:
        if counts==duration:
            break
        if duration>0:
            next_date = target_date + timedelta(days=1)
        else:
            next_date = target_date - timedelta(days=1)
        if is_holiday(next_date) or next_date.weekday()==5 or next_date.weekday()==6:
            target_date = next_date
        else:
            if duration>0:
                counts += 1
            else:
                counts -= 1
            target_date = next_date
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

def get_first_and_last_datetime(day):
    """取得指定日期的第一分钟，和最后一分钟的时间戳"""
    
    first_time = dt.datetime(day.year,day.month,day.day,0,0,0)
    first_timestamp = time.mktime(first_time.timetuple())
    last_time = dt.datetime(day.year,day.month,day.day,23,59,59)
    last_timestamp = time.mktime(last_time.timetuple())
    
    return first_timestamp,last_timestamp

def get_previous_month(date):
    """取得上个月月份字符串"""
    
    first = date.replace(day=1)
    last_month = first - dt.timedelta(days=1)
    last_month_str = last_month.strftime("%Y%m")
    return last_month_str

def get_next_month(date,next=1):
    """取得下个月月份字符串"""
    
    week, days_num = calendar.monthrange(date.year, date.month)
    month_later = date + relativedelta(months=next)
    return month_later
    
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

def get_trade_min_dur(trade_time,period_number=5):
    """按照A股交易规则，计算两个时间的交易间隔数"""
    
    trade_day = trade_time.strftime('%Y%m%d')
    # 生成当天上午及下午开盘时间
    m_begin = datetime(trade_time.year,trade_time.month,trade_time.day,9,30)
    m_end = datetime(trade_time.year,trade_time.month,trade_time.day,11,30)
    a_begin = datetime(trade_time.year,trade_time.month,trade_time.day,13,0)
    a_end = datetime(trade_time.year,trade_time.month,trade_time.day,15,0)
    dur_time = None
    if trade_time<m_begin:
        return 0
    if trade_time<m_end:
        dur_time = trade_time - m_begin
    if trade_time>m_end and trade_time<a_begin:
        dur_time = m_end - m_begin
    if trade_time>a_begin and trade_time<a_end:
        dur_time = (m_end - m_begin) + (trade_time - a_begin)      
    if trade_time>a_end:
        dur_time = (m_end - m_begin) + (a_end - a_begin)   
    dur_number = dur_time.seconds//(60*period_number) + 1
    # day_item_number = 4 * 60 / period_number
    # dur_number = day_item_number - dur_number
    return int(dur_number)
