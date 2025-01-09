import six
import datetime
from datetime import date,datetime as dt_obj
import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from rqalpha.data.base_data_source import BaseDataSource
from rqalpha.model.tick import TickObject
from rqalpha.apis import instruments,get_previous_trading_date
from rqalpha.const import TRADING_CALENDAR_TYPE
from trader.utils.date_util import get_tradedays_dur
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from data_extract.juejin_futures_extractor import JuejinFuturesExtractor
from data_extract.akshare_extractor import AkExtractor
from trader.rqalpha.dict_mapping import judge_market,transfer_instrument

_STOCK_FIELD_NAMES = [
    'datetime', 'open', 'high', 'low', 'close', 'vol', 'amount'
]

class FuturesDataSource(BaseDataSource):
    """期货自定义数据源"""
    
    def __init__(self, path,stock_data_path,frequency_sim=True):
        super(FuturesDataSource, self).__init__(path,{})
        self.extractor = JuejinFuturesExtractor(savepath=stock_data_path)
        self.extractor_ak = AkExtractor(savepath=stock_data_path)
        # 是否实时模式
        self.frequency_sim = frequency_sim 

    def get_k_data(self, instrument, start_dt, end_dt,frequency=None,need_prev=True,institution=False):
        """从已下载的文件中，加载K线数据"""
        
        # 可以加载不同的频次类型数据
        if frequency=="5m":
            period = PeriodType.MIN5.value
            extractor = self.extractor
        if frequency=="1m":
            period = PeriodType.MIN1.value
            extractor = self.extractor            
        # 日线数据使用akshare的数据源
        if frequency=="1d":
            period = PeriodType.DAY.value    
            extractor = self.extractor_ak   
            start_dt = dt_obj(start_dt.year, start_dt.month, start_dt.day) 
            end_dt = dt_obj(end_dt.year, end_dt.month, end_dt.day) 
        
        # 筛选对应日期以及股票的相关数据
        item_data = extractor.load_item_df(instrument.order_book_id.split(".")[0],period=period,institution=institution)
        item_data = item_data[(item_data["datetime"]>=start_dt)&(item_data["datetime"]<=end_dt)]
        if item_data.shape[0]==0:
            return None
        # 改变为rqalpha格式
        item_data["last"] = item_data["close"]
        # 取得前一个交易时段收盘
        item_data["prev_close"]= np.NaN
        if need_prev:
            item_data["prev_close"] = self._get_prev_close(instrument, start_dt,frequency=frequency)   
        return item_data

    def get_bar(self, instrument, dt, frequency,need_prev=True):
        if frequency != '1m' and frequency != '1d':
            return super(FuturesDataSource, self).get_bar(instrument, dt, frequency)
        
        if dt.hour<=9 and dt.minute<35:
            # 
            dt = get_tradedays_dur(dt,-1)
            frequency = "1d"
            bar_data = self.get_k_data(instrument, dt, dt,frequency=frequency,need_prev=need_prev)
        else:
            bar_data = self.get_k_data(instrument, dt, dt,frequency=frequency,need_prev=need_prev)

        if bar_data is None or bar_data.empty:
            return None
        else:
            return bar_data.iloc[0].to_dict()

    def history_bars(self, instrument, bar_count, frequency, fields, dt, skip_suspended=True,include_now=False, adjust_type=None, adjust_orig=None):
        """历史数据查询"""
        
        cal_df_index = self.get_trading_calendars()[TRADING_CALENDAR_TYPE.EXCHANGE]
        start_dt_loc = cal_df_index.get_loc(dt.replace(hour=0, minute=0, second=0, microsecond=0)) - bar_count + 1
        start_dt = cal_df_index[start_dt_loc]

        bar_data = self.get_k_data(instrument, start_dt, dt,frequency=frequency)

        if bar_data is None or bar_data.empty:
            return None
        else:
            if isinstance(fields, six.string_types):
                fields = [fields]
            fields = [field for field in fields if field in bar_data.columns]

            return bar_data[fields].values

    def valid_bar_date(self,dt):
        if dt.minute % 5 > 0:
            # 检查是否5分钟间隔
            return False
        return True        
    
    def current_snapshot(self,instrument, frequency, dt):
        """取得指定股票的当前交易信息快照"""
        
        order_book_id = instrument.order_book_id
        symbol = instrument.trading_code
        
        if frequency!="1m":
            return super(TdxDataSource, self).current_snapshot(instrument, frequency, dt)
        
        if self.frequency_sim:   
            # 如果实时模式，则取得实时数据      
            market = judge_market(order_book_id)    
            bar = self.get_real_data(symbol,market)
        else:
            # 目前rqalpha只支持1分钟和1天，而本数据源以5分钟为主，在此进行判别
            if not self.valid_bar_date(dt):
                # 如果不是5分钟间隔，则返回空
                return None            
            bar = self.get_bar(instrument,dt,frequency)
        tick_obj = TickObject(instrument, bar)
        return tick_obj

    def _get_prev_close(self, instrument, dt,frequency=None):
        """取得上一交易时段的收盘价"""
        
        # 根据当前时间点，取得上一时间点
        if frequency=="1m":
            # 分钟级别，使用datetime进行直接计算
            prev_datetime = dt - datetime.timedelta(minutes=5)  
        if frequency=="1d":
            # 日期级别，使用api方法取得上一交易日
            prev_datetime = get_previous_trading_date(dt)
            
        bar = self.get_bar(instrument,prev_datetime,frequency=frequency,need_prev=False)
        if bar is None or len(bar) < 1:
            return np.nan
        return bar["last"]

    def get_open_auction_bar(self, instrument, dt):
        """重载原方法"""
        
        # 使用昨收盘模拟开盘价
        prev_date = get_tradedays_dur(dt,-1)
        day_bar = self.get_bar(instrument, prev_date, "1d")
        if day_bar is None:
            bar = dict.fromkeys(self.OPEN_AUCTION_BAR_FIELDS, np.nan)
        else:
            # 此处修改为对字典数据的检查
            bar = {k: day_bar[k] if k in day_bar else np.nan for k in self.OPEN_AUCTION_BAR_FIELDS}
        bar["last"] = bar["open"]
        return bar
                
    def available_data_range(self, frequency):
        # 修改原方法，有效日期的结束日期设置为后一天（原来为前一天）
        return date(2004, 1, 1), date.today() + relativedelta(days=1)
    
    def get_real_data(self,instrument,market):
        real_data = self.extractor.get_real_data(instrument,market)
        return real_data.iloc[0].to_dict()        
        
        