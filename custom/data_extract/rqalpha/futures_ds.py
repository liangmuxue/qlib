import os
import datetime
import time
from datetime import date, timedelta
from datetime import datetime as dt_obj
import numpy as np
import pandas as pd

from rqalpha.data.base_data_source.storages import (ExchangeTradingCalendarStore)
from rqalpha.const import TRADING_CALENDAR_TYPE
from rqalpha.data.base_data_source import BaseDataSource
from rqalpha.model.tick import TickObject
from rqalpha.model.instrument import Instrument
from trader.utils.date_util import get_tradedays_dur,get_tradedays,get_prev_working_day,get_next_day
from cus_utils.db_accessor import DbAccessor
from cus_utils.string_util import find_first_digit_position
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from data_extract.juejin_futures_extractor import JuejinFuturesExtractor
from data_extract.akshare_extractor import AkExtractor
from trader.rqalpha.dict_mapping import judge_market,transfer_instrument
from data_extract.rqalpha.fur_ds_proxy import FutureInfoStore
from dateutil.relativedelta import relativedelta
from data_extract.akshare_futures_extractor import AkFuturesExtractor

_STOCK_FIELD_NAMES = [
    'datetime', 'open', 'high', 'low', 'close', 'vol', 'amount'
]

class FuturesDataSource(BaseDataSource):
    """期货自定义数据源"""
    
    def __init__(self, path,stock_data_path=None,sim_path=None,frequency_sim=True):
        # super(FuturesDataSource, self).__init__(path,{})

        def _p(name):
            return os.path.join(path, name)
        
        self._calendar_providers = {
            TRADING_CALENDAR_TYPE.EXCHANGE: ExchangeTradingCalendarStore(_p("trading_dates.npy"))
        }
        self._future_info_store = FutureInfoStore(_p("future_info.json"), None)              
          
        self.dbaccessor = DbAccessor({})
        self.busi_columns = ["code","datetime","open","close","high","low","volume","hold","settle"]
        self.day_contract_columns = ["symbol","contract","volume"]
        
        self.extractor = JuejinFuturesExtractor(savepath=stock_data_path,sim_path=sim_path)
        self.extractor_ak = AkFuturesExtractor(savepath=stock_data_path)
        # 是否实时模式
        self.frequency_sim = frequency_sim 
        self.load_all_trading_variety()
    
    def load_all_trading_variety(self):
        """预加载所有品种基础数据"""
        
        trading_variety_data = self.get_all_trading_variety()
        trading_variety_mapping = {}
        for index,row in trading_variety_data.iterrows():
            trading_variety_mapping[row['code']] = row
        self.trading_variety_mapping = trading_variety_mapping
    
    def get_trading_variety_info(self,order_book_id):
        
        symbol_code = self.get_instrument_code_from_contract_code(order_book_id)
        symbol_obj = self.trading_variety_mapping[symbol_code]
        return symbol_obj
    
    def time_inject(self,code_name=None,begin=False):
        if begin:
            self.time_begin = time.time()
        else:
            elapsed_time = time.time() - self.time_begin   
            self.time_begin = time.time()
            # print("{} Elapsed time: {} seconds".format(code_name,elapsed_time))  
            
    def get_trading_calendars(self):
        return {TRADING_CALENDAR_TYPE.EXCHANGE: self.get_trading_calendar()}
                        
    def load_sim_data(self,simdata_date):
        self.extractor.load_sim_data(simdata_date,dataset=self.dataset)

    def get_trading_calendar(self, trading_calendar_type=None):
        """取得交易日历"""
        
        begin_date = "20240101"
        end_date = "20251231"
        trade_days = get_tradedays(begin_date,end_date)
        
        return trade_days

    def get_instruments(self, id_or_syms=None, types=None):
        # type: (Optional[Iterable[str]], Optional[Iterable[INSTRUMENT_TYPE]]) -> Iterable[Instrument]
        if id_or_syms is None:
            return []
        instruments = []
        for sym in id_or_syms:
            contract_info = self.get_contract_info(sym)
            if contract_info.shape[0]==0:
                continue
            params = {"order_book_id":contract_info.iloc[0]['code'],"symbol":contract_info.iloc[0]['name'],
                      "type":"Future","trading_hours":contract_info.iloc[0]['trading_hours'],"de_listed_date":None}
            ins = Instrument(params)
            instruments.append(ins)
        return instruments
        
    def build_trading_contract_mapping(self,date):  
        """生成对应日期的所有可交易合约的对照表"""
        
        contract_info = self.get_contract_info()
        results = []
        for i in range(contract_info.shape[0]):
            item = contract_info.iloc[i]
            # 根据日期，取得对应品种的主力合约以及次主力合约
            contract_names = self.extractor_ak.get_likely_main_contract_names(item['code'], date)
            for symbol in contract_names:
                # 查看合约当日的日线数据，如果有则记录
                contract_data = self.get_contract_data_by_day(symbol, date)     
                if contract_data is not None:
                    results.append([item['code'],symbol,contract_data['volume'].values[0]])
        if len(results)==0:
            return None
        results = pd.DataFrame(np.array(results),columns=self.day_contract_columns)
        results['date'] = date.date()
        return results
        
    def has_current_data(self,day,symbol):
        """当日是否开盘交易"""

        # 直接使用数据源,检查当日是否有分时数据
        item_df = self.extractor.get_time_data_by_day(symbol,day,period=PeriodType.MIN1.value)
        if item_df is None or item_df.shape[0]==0:
            return False
        return True
    
    def rectification_order_book_id(self,symbol,inverse=False):
        """校正合约编码转换问题"""
        
        instrument_code = self.get_instrument_code_from_contract_code(symbol)
        exchange_code = self.get_exchange_from_instrument(instrument_code)
        # 郑商所有特殊编码规则，需要转换,并且其他交易所都是小写，只有郑商所大写
        if exchange_code=="CZCE":
            if inverse:
                symbol = symbol[:2] + "2" + symbol[2:]
                symbol = symbol.upper()
            else:
                symbol = symbol[:2] + symbol[3:]
        else:
            if inverse:
                symbol = symbol.upper()
            else:
                symbol = symbol.lower()
        return symbol
        
    def transfer_futures_order_book_id(self,symbol,date):    
        """品种代码转化为合约代码"""
        
        main_contract_code = self.get_main_contract_name(symbol, date)
        return main_contract_code
        
    def get_time_data_by_day(self,day,symbol):
        """取得指定品种和对应日期的分时交易记录,TODO：注意需要加入前一天晚盘"""

        return self.extractor.get_time_data_by_day(day,symbol)

    def get_instrument_code_from_contract_code(self,contract_code):
        """根据合约编码，取得品种代码"""
        
        return self.extractor.get_instrument_code_from_contract_code(contract_code)
        
    def get_main_contract_name(self,instrument,date):
        """根据品种编码和指定日期，根据交易情况，确定对应的主力合约"""
        
        #取得有可能的所有合约名称
        if isinstance(date,str):
            date_obj = dt_obj.strptime(str(date), '%Y%m%d')
        else:
            date_obj = date
        contract_names = self.extractor_ak.get_likely_main_contract_names(instrument, date_obj,ref=False)
        # 检查潜在合约的上一日成交金额，如果超出10%则进行合约切换
        contract_mapping = {}
        volume_main = 0
        main_name = None
        for symbol in contract_names:
            item_df = self.load_item_day_data(symbol,date_obj)
            if item_df is None or item_df.shape[0]==0:
                continue
            cur_volume = item_df['volume'].values[0]
            if volume_main<cur_volume:
                volume_main = cur_volume
                main_name = item_df['code'].values[0]
            
        return main_name       

    
    def get_k_data(self, order_book_id, start_dt, end_dt,frequency=None,need_prev=False):
        """从已下载的文件中，加载K线数据"""
        
        self.time_inject(begin=True)
        # 可以加载不同的频次类型数据
        if frequency=="1m":
            period = PeriodType.MIN1.value
            # item_data = self.extractor.load_item_df(instrument,period=period)   
            item_data = self.extractor.load_item_by_time(order_book_id,start_dt,period=period)   
            if item_data.shape[0]==0:
                return None                
            item_data["last"] = item_data["close"]      
        # 日线数据使用akshare的数据源
        if frequency=="1d":
            start_dt = dt_obj(start_dt.year, start_dt.month, start_dt.day).date() 
            end_dt = dt_obj(end_dt.year, end_dt.month, end_dt.day).date()
            item_data = self.load_item_day_data(order_book_id,start_dt)
            if item_data is None or item_data.shape[0]==0:
                return None                
            # 使用结算价作为当日最终价格
            item_data["last"] = item_data["settle"]    
            # 字段和rq统一
            item_data['symbol'] = item_data['code']        
            item_data = item_data.rename(columns={"date":"datetime"})

        # 取得前一个交易时段收盘
        item_data["prev_close"]= np.NaN
        if need_prev:
            item_data["prev_close"] = self._get_prev_close(order_book_id, start_dt,frequency=frequency)
        # item_data = item_data.iloc[0].to_dict()
        return item_data

    def get_bar(self, order_book_id, dt, frequency,need_prev=False):
        if frequency != '1m' and frequency != '1d':
            return super(FuturesDataSource, self).get_bar(order_book_id, dt, frequency)
        
        if frequency == '1m' and not self.is_trade_opening_for_contract(order_book_id,dt):
            # 如果不在交易时间，则不出数
            return None
        else:
            bar_data = self.get_k_data(order_book_id, dt, dt,frequency=frequency,need_prev=need_prev)

        if bar_data is None or bar_data.empty:
            return None
        else:
            return bar_data.iloc[0].to_dict()

    def history_bars(self, order_book_id, bar_count, frequency, fields=None, dt=None):
        """历史数据查询"""
        
        start_dt = get_tradedays_dur(dt,-bar_count)
        bar_data = self.get_k_data(order_book_id, start_dt, dt,frequency=frequency)
        if bar_data is None or bar_data.empty:
            return None
        else:
            return bar_data[fields].values
    
    def current_snapshot(self,instrument, frequency, dt):
        """取得指定品种的当前交易信息快照"""
        
        order_book_id = instrument.order_book_id
        symbol = instrument.trading_code

        if self.frequency_sim:   
            # 如果实时模式，则取得实时数据      
            market = judge_market(order_book_id)    
            bar = self.get_real_data(symbol,market)
        else:
            bar = self.get_bar(instrument,dt,frequency)
            
        if not bar:
            return None
            
        def tick_fields_for(ins):
            _STOCK_FIELD_NAMES = [
                'datetime', 'open', 'high', 'low', 'last', 'volume', 'total_turnover', 'prev_close',
                'limit_up', 'limit_down'
            ]
            _FUTURE_FIELD_NAMES = _STOCK_FIELD_NAMES + ['open_interest', 'prev_settlement']

            if ins.type == 'Future':
                return _STOCK_FIELD_NAMES
            else:
                return _FUTURE_FIELD_NAMES
                        
        d = {k: bar[k] for k in tick_fields_for(instrument) if k in list(bar.keys())}
        d['last'] = bar['close']
        d['prev_close'] = bar['prev_close'] 
        # if 'code' in bar:
        #     d['symbol'] = bar['code'] 
           
        tick_obj = TickObject(instrument, d)
        return tick_obj

    def _get_prev_close(self, order_book_id, dt,frequency=None):
        """取得上一交易时段的收盘价"""
        
        # 根据当前时间点，取得上一时间点
        if frequency=="1m":
            # 分钟级别，使用datetime进行直接计算
            prev_datetime = dt - datetime.timedelta(minutes=1)  
        if frequency=="1d":
            # 日期级别，使用api方法取得上一交易日
            prev_datetime = get_prev_working_day(dt)
            prev_datetime = dt_obj.combine(prev_datetime, dt_obj.min.time())
            
        bar = self.get_bar(order_book_id,prev_datetime,frequency=frequency,need_prev=False)
        if bar is None or len(bar) < 1:
            return np.nan
        return bar["close"]

    def get_commission_info(self, instrument):
        """计算交易手续费"""
        return self._future_info_store.get_future_info(instrument)
    
    
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

    def load_item_day_data(self,symbol,date):  
        """加载指定日期和合约名称的日线数据"""
        
        item_sql = "select UPPER(code),date,open,close,high,low,volume,hold,settle from dominant_real_data where code='{}' " \
            "and date='{}'".format(symbol,date.strftime('%Y-%m-%d'))
        result_rows = self.dbaccessor.do_query(item_sql)   
        if len(result_rows)==0:
            return None
        result = pd.DataFrame(np.expand_dims(np.array(list(result_rows[0])),0),columns=self.busi_columns) 
        return result       

    def load_item_allday_data(self,symbol):  
        """加载指定合约名称的所有日线数据"""
        
        item_sql = "select code,date,open,close,high,low,volume,hold,settle from dominant_real_data where code='{}' ".format(symbol)
        result_rows = self.dbaccessor.do_query(item_sql)   
        if len(result_rows)==0:
            return None
        result = pd.DataFrame(np.array(list(result_rows)),columns=self.busi_columns) 
        return result     

    def get_contract_data_by_day(self,symbol,date):  
        """加载指定合约,指定日期的日线数据"""
        
        item_sql = "select code,date,open,close,high,low,volume,hold,settle from dominant_real_data" \
            " where code='{}' and date='{}'".format(symbol,date.strftime('%Y-%m-%d'))
        result_rows = self.dbaccessor.do_query(item_sql)   
        if len(result_rows)==0:
            return None
        result = pd.DataFrame(np.array(list(result_rows)),columns=self.busi_columns) 
        return result  
        
    def get_contract_info(self,contract_code=None):     
        """取得合约相关信息"""
        
        if contract_code is not None:
            item_code = self.get_instrument_code_from_contract_code(contract_code)
            item_sql = "select name,code,multiplier,limit_rate,magin_radio,price_range,COALESCE(day_time_range, ''),day_time_range,night_time_range  from trading_variety where code='{}' ".format(item_code)
        else:
            item_sql = "select name,code,multiplier,limit_rate,magin_radio,price_range,COALESCE(day_time_range, ''),day_time_range,night_time_range  from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(item_sql)  
        result_arr = np.array(result_rows)         
        if result_arr.shape[0]==0:
            return result_arr
        result_arr = pd.DataFrame(result_arr,columns=["name","code","multiplier","limit_rate","magin_radio","price_range","trading_hours","day_time_range","night_time_range"])
        
        return result_arr
        
    def get_all_contract_names(self,date=None):
        """取得所有合约名称（用于订阅）"""
        
        contract_info = self.get_contract_info()
        contract_names = []
        for i in range(contract_info.shape[0]):
            item = contract_info.iloc[i]
            # 根据日期，取得对应品种的主力合约
            name = self.get_main_contract_name(item['code'],date)
            if name is None:
                continue
            contract_names.append(name)
                
        return contract_names
    
    def get_trading_minutes_list(self,instrument, trading_dt):
        """取得对应合约的交易时间段,返回排序列表"""
        
        trading_minutes = self.get_trading_minutes_for(instrument, trading_dt)
        return sorted(trading_minutes)
        
        
    def get_trading_minutes_for(self,instrument, trading_dt):
        """取得对应合约的交易时间段,注意如果包含晚盘，则晚盘开始作为开盘"""
        
        if isinstance(instrument,str):
            contract_symbol = instrument
        else:
            contract_symbol = instrument.order_book_id
        code = self.get_instrument_code_from_contract_code(contract_symbol)
        # 从数据库取得对应品种的交易时间范围
        sql = "select code,ac_time_range,day_time_range,night_time_range from trading_variety where code='{}' ".format(code)
        result_rows = self.dbaccessor.do_query(sql)  
        if len(result_rows)==0:
            return None
        day_time_range = result_rows[0][2]
        night_time_range = result_rows[0][3]
        # 从时间范围定义拆解为分钟列表
        trading_minutes = set()
        split_range = []
        if not ',' in day_time_range:
            split_range.append(day_time_range)
        else:
            split_range += day_time_range.split(",")
        if not ',' in night_time_range:
            split_range.append(night_time_range)
        else:
            split_range += night_time_range.split(",")    
        
        # 遍历所有切分段，拆解为每个分钟
        for item in split_range:
            if len(item)<3:
                continue
            begin_end = item.split("-")
            start_time = begin_end[0].strip()
            end_time = begin_end[1].strip()
            # 如果当前阶段为晚盘，则日期前提一天
            if start_time=="21:00":
                prev_trading_dt = get_prev_working_day(trading_dt)
                start = prev_trading_dt.strftime('%Y-%m-%d ') + start_time
                # 如果为跨天的夜盘，需要使用下一个自然日作为结束日期
                if end_time.startswith("0"):
                    end = get_next_day(prev_trading_dt).strftime('%Y-%m-%d ') + end_time
                else:
                    end = prev_trading_dt.strftime('%Y-%m-%d ') + end_time
            else:            
                start = trading_dt.strftime('%Y-%m-%d ') + start_time
                end = trading_dt.strftime('%Y-%m-%d ') + end_time
            min_list = pd.date_range(start=start, end=end,freq='min')
            trading_minutes.update(set(min_list.to_pydatetime().tolist()))
            
        return trading_minutes
    
    def get_exchange_from_instrument(self,instrument_code):
        
        sql = "select e.code from trading_variety v,futures_exchange e where v.exchange_id=e.id and v.code='{}' ".format(instrument_code)
        result_rows = self.dbaccessor.do_query(sql)  
        if len(result_rows)==0:
            return None        
        return result_rows[0][0]

    def get_all_trading_variety(self):
        """取得所有可交易品种"""
        
        data = self.get_contract_info()
        return data
    
    ############################### Busi Data Logic ###########################################
    
    def get_hot_key(self,date_time,symbol):
        hot_key_str = "{}_{}".format(symbol,datetime.datetime.strftime(date_time, '%H%M'))
        return hot_key_str
    
    def build_hot_loading_data(self,now_date,symbol_list,reset=False):
        """生成热加载数据，提升查询性能"""
        
        if reset:
            self.hot_data_mappings = {}
            self.hot_data_symbol_mappings = {}
        
        def hot_data_create(symbol,date):
            # 从存储中拿到当天所有分钟数据，并放入热区
            item_df = self.get_time_data_by_day(str(date),symbol)
            self.hot_data_symbol_mappings[symbol] = item_df
            for row in item_df.iterrows():
                hot_key_str = self.get_hot_key(row[1]['datetime'],symbol)
                self.hot_data_mappings[hot_key_str] = row                          
        # 分别对待开仓、待平仓、持仓数据进行热区加载，提前把所有1分钟数据加载到键值对数据中，加速后续查询
        for key in symbol_list:
            hot_data_create(key,now_date)
            
    def is_trade_opening(self,dt):
        """检查当前是否已开盘"""
        
        # 出于数据获取考虑，9点01才算开盘
        open_timing = datetime.datetime(dt.year,dt.month,dt.day,9,0,0)
        if dt>open_timing:
            return True
        return False

    def is_trade_opening_for_contract(self,order_book_id,trading_dt):
        
        symbol_obj = self.get_trading_variety_info(order_book_id)
        return self.is_trade_opening_for_symbol(symbol_obj,trading_dt)
        
    def is_trade_opening_for_symbol(self,symbol_obj,trading_dt):
        """检查指定时间下的指定品种是否可以交易"""
        
        day_time_range = symbol_obj['day_time_range']
        night_time_range = symbol_obj['night_time_range']
        
        split_range = []
        if not ',' in day_time_range:
            split_range.append(day_time_range)
        else:
            split_range += day_time_range.split(",")
        if not ',' in night_time_range:
            split_range.append(night_time_range)
        else:
            split_range += night_time_range.split(",")    
        
        def between_time(start,end,dt):
            start_time = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
            if start_time<=dt<=end_time:
                return True
            return False
                
        # 遍历所有切分段，拆解为每个分钟
        for item in split_range:
            if len(item)<3:
                return False
            begin_end = item.split("-")
            start_time = begin_end[0].strip()
            end_time = begin_end[1].strip()
            # 如果当前阶段为晚盘，则日期前提一天
            if start_time=="21:00":
                # 如果为跨天的夜盘，则从0点计算开始时间
                if end_time.startswith("0"):
                    start = trading_dt.strftime('%Y-%m-%d ') + "00:00:00"
                    end = trading_dt.strftime('%Y-%m-%d ') + end_time + ":00"
                    if between_time(start,end,trading_dt):
                        return True           
                    # 追加到当日24点的交易时间段     
                    start = trading_dt.strftime('%Y-%m-%d ') + start_time + ":00"
                    end = trading_dt.strftime('%Y-%m-%d ') + "23:59:59"
                    if between_time(start,end,trading_dt):
                        return True                            
                else:
                    start = trading_dt.strftime('%Y-%m-%d ') + start_time + ":00"
                    end = trading_dt.strftime('%Y-%m-%d ') + end_time + ":00"
                    if between_time(start,end,trading_dt):
                        return True                    
            else:            
                start = trading_dt.strftime('%Y-%m-%d ') + start_time + ":00"
                end = trading_dt.strftime('%Y-%m-%d ') + end_time + ":00"
                if between_time(start,end,trading_dt):
                    return True
            
        return False 
               
    def get_prev_price(self,order_book_id,now_dt):
        """取得上一分钟交易数据，注意需要考虑中间的休市时间以及晚盘的时间"""
        
        if now_dt.hour==10 and now_dt.minute==30:
            last_dt = datetime.datetime(now_dt.year,now_dt.month,now_dt.day,10,15,0)
        elif now_dt.hour==13 and now_dt.minute==30:
            last_dt = datetime.datetime(now_dt.year,now_dt.month,now_dt.day,11,30,0)
        elif now_dt.hour==21 and now_dt.minute==0:
            last_dt = datetime.datetime(now_dt.year,now_dt.month,now_dt.day,15,0,0)            
        else:
            last_dt = now_dt + timedelta(minutes=-1)       
        price = self.get_last_price(order_book_id, last_dt)
        return price

    def get_last_price(self,order_book_id,dt,fileds="close"):
        """取得指定标的最近报价信息"""

        # 如果未开盘则取昨日收盘价,开盘后取当前时间点收盘价
        if not self.is_trade_opening_for_contract(order_book_id,dt):
            prev_day = get_tradedays_dur(dt, -1) 
            bar = self.get_bar(order_book_id,prev_day,"1d")
            if bar is None:
                return None
            return bar[fileds]
        
        # 首先从热区加载，如果没有再从存储中读取
        hot_key_str = self.get_hot_key(dt,order_book_id)
        if hot_key_str in self.hot_data_mappings:
            return self.hot_data_mappings[hot_key_str][1]['close']
        # 分钟数据键值对没有，再从当日品种热区里找
        if order_book_id in self.hot_data_symbol_mappings:
            item_data = self.hot_data_symbol_mappings[order_book_id]
            target_data = item_data[item_data['datetime']<=pd.to_datetime(dt)]
            if target_data.shape[0]>0:
                target_data = target_data.sort_values(by=["datetime"],ascending=False)
                return target_data['close'].values[0]
        # 都没有，再从存储中查
        recent_datetime = self.get_recent_date_by_date(order_book_id,dt)
        if recent_datetime is None:
            return None        
        # 取得当日距离指定时间最近的交易
        bar = self.get_bar(order_book_id,recent_datetime,"1m")
        if bar is None:
            return None
        return bar['close']
            