import time
import datetime as dt
from datetime import datetime,timedelta,date
import chinese_calendar as ccal
from chinese_calendar import is_holiday
import pandas_market_calendars as mcal
from dateutil.relativedelta import relativedelta
import pandas as pd

def get_end_of_month(trade_days):
    """取得月末日期"""

    # 1、每月最后交易日（月末锚点）
    month_last = trade_days.to_series().resample("M").last().dropna()
    # 2、月末效应：月末最后3个交易日（你做月末效应核心）
    month_effect_days = []
    for dt in month_last:
        pos = trade_days.get_indexer([dt], method="ffill")[0]
        slice_day = trade_days[max(0, pos-2):pos+1]
        month_effect_days.extend(slice_day)
    
    # 3、季度末最后3个交易日（季末资金效应）
    q_last = trade_days.to_series().resample("Q").last().dropna()
    q_effect_days = []
    for dt in q_last:
        pos = trade_days.get_indexer([dt], method="ffill")[0]
        slice_day = trade_days[max(0, pos-2):pos+1]
        q_effect_days.extend(slice_day)
        
    return month_effect_days,q_effect_days
    
def get_long_holiday_eve(start_year, end_year):
    """取得长假前一天"""
    
    holiday_names = [
        "New Year's Day", "Spring Festival", "Tomb-sweeping Day", "Labour Day", "Dragon Boat Festival", "National Day","Mid-autumn Festival"
    ]  
        
    start = date(start_year,1,1)
    end = date(end_year,12,31)
    dt_range = pd.date_range(start, end, freq="D")
    eve_dates = []

    for d in dt_range:
        cur_date = d.date()
        next_date = cur_date + timedelta(days=1)

        # 1.当天是工作日（包含周末调休补班）
        if not ccal.is_workday(cur_date):
            continue

        # 2.次日是否【法定节假日（带节日名，排除普通周末）】
        is_legal_hol, hol_name = ccal.get_holiday_detail(next_date)
        # 必须存在节日名称且在法定节日列表，普通周末is_legal_hol=False/无名称
        if is_legal_hol and hol_name in holiday_names:
            eve_dates.append(d)

    df_ref = pd.DataFrame({"datetime":eve_dates, "is_holiday_eve":1})
    df_ref["datetime"] = pd.to_datetime(df_ref["datetime"])
    return df_ref

def get_holiday_eves(start_year: int, end_year: int) -> pd.DataFrame:
    """
    获取指定年份范围内，所有国内长假前一天
    长假包含：春节、清明节、劳动节、端午节、中秋节、国庆节
    返回：DataFrame（含 假期名称、假期开始日、假期前一天）
    """
    # 1. 生成目标年份所有日期
    dates = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D"
    )
    df = pd.DataFrame({"datetime": dates})

    # 2. 标记所有法定节假日 & 假期类型
    df["is_holiday"] = df["datetime"].apply(chinese_calendar.is_holiday)
    df["is_workday"] = df["datetime"].apply(chinese_calendar.is_workday)
    df["holiday_name"] = df["datetime"].apply(
        lambda d: chinese_calendar.get_holiday_detail(d)[1]
    )

    # 3. 筛选出【长假开始日】（排除调休、周末、重复日期）
    # holiday_names = [
    #     "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节"
    # ]
    holiday_names = [
        "New Year's Day", "Spring Festival", "Tomb-sweeping Day", "Labour Day", "Dragon Boat Festival", "National Day","Mid-autumn Festival"
    ]    
    holiday_starts = df[
        (df["holiday_name"].isin(holiday_names))
        & (df["is_holiday"] == True)
    ].copy()

    # 去重：每个假期只保留第一天
    holiday_starts = holiday_starts.drop_duplicates(subset=["holiday_name", "datetime"], keep="first")

    # 4. 计算【长假前一天】
    holiday_starts["holiday_eve"] = holiday_starts["datetime"] - pd.Timedelta(days=1)

    # 5. 整理输出
    # result = holiday_starts[["holiday_name", "datetime", "holiday_eve"]].rename(
    #     columns={"datetime": "假期开始日", "holiday_eve": "长假前一天"}
    # )
    return holiday_starts.reset_index(drop=True)

def get_next_working_day(day):
    """取得指定日期的下一工作日"""
    
    cur_day = day
    while(True):
        next_day = cur_day + timedelta(days=1)
        if is_working_day(next_day):
            return next_day
        cur_day = next_day

def get_prev_working_day(day):
    """取得指定日期的上一工作日"""
    
    cur_day = day
    while(True):
        prev_day = cur_day - timedelta(days=1)
        if is_working_day(prev_day):
            return prev_day
        cur_day = prev_day
        
def get_previous_day(day):
    """取得指定日期的上一日"""
    return day + timedelta(days=-1)

def get_next_day(day):
    """取得指定日期的下一日"""
    return day + timedelta(days=1)
    
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
        if start >= end:
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

def get_months_ago(date,months=1):
    """取得指定日期前几个月的时间日期"""
    
    ago_time = date - relativedelta(months=months)
    target_month = ago_time.strftime('%Y%m')
    return target_month

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
    
def get_tradedays(start,end,date_format=False):
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
        if date_format:
            cal_list.append(start)
        else:
            cal_list.append(start.strftime('%Y%m%d'))
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

def get_nowtime_working_day():
    """取得当前交易日期，注意从15点开始，就算作下一交易日"""
    
    now_time = datetime.now()
    now_date = now_time.date()
    # 如果当日为非工作日，则返回下一个工作日
    if not is_working_day(now_time.strftime("%Y%m%d")):
        return get_next_working_day(now_date)
    
    if (now_time.hour<15):
        return now_date
    next_day = get_next_working_day(now_date)
    
    return next_day

