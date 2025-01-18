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
from cus_utils.db_accessor import DbAccessor
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from data_extract.juejin_futures_extractor import JuejinFuturesExtractor
from data_extract.akshare_extractor import AkExtractor
from trader.rqalpha.dict_mapping import judge_market,transfer_instrument
import cus_utils.global_var as global_var
from data_extract.akshare_futures_extractor import AkFuturesExtractor

_STOCK_FIELD_NAMES = [
    'datetime', 'open', 'high', 'low', 'close', 'vol', 'amount'
]

class FuturesDataSource(BaseDataSource):
    """期货自定义数据源"""
    
    def __init__(self, path,stock_data_path=None,sim_path=None,frequency_sim=True):
        super(FuturesDataSource, self).__init__(path,{})
        
        self.dbaccessor = DbAccessor({})
        self.busi_columns = ["code","datetime","open","high","low","close","volume","hold","settle"]
        self.day_contract_columns = ["symbol","contract"]
        
        self.extractor = JuejinFuturesExtractor(savepath=stock_data_path,sim_path=sim_path)
        self.extractor_ak = AkFuturesExtractor(savepath=stock_data_path)
        # 是否实时模式
        self.frequency_sim = frequency_sim 
        # 从全局变量中取得主流程透传的主体模型,以及对应数据集对象
        model = global_var.get_value("model")
        dataset = model.dataset
        dataset.build_series_data(no_series_data=True)
        self.dataset = dataset
        
    def load_sim_data(self,simdata_date):
        self.extractor.load_sim_data(simdata_date,dataset=self.dataset)
      
    def build_trading_contract_mapping(self,date):  
        """生成对应日期的所有可交易合约的对照表"""
        
        contract_info = self.get_contract_info()
        results = []
        for i in range(contract_info.shape[0]):
            item = contract_info.iloc[i]
            # 根据日期，取得对应品种的主力合约
            symbol = self.get_main_contract_name(item['code'],date)
            # 查看主力合约当日的日线数据，如果有则记录
            contract_data = self.get_contract_data_by_day(symbol, date)     
            if contract_data is not None:
                results.append([item['code'],symbol])
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
    
    def transfer_furtures_order_book_id(self,symbol,date):    
        """品种简写编码转化为合约代码"""
        
        main_contract_code = self.get_main_contract_name(symbol, date)
        return main_contract_code
        
        
    def get_main_contract_name(self,instrument,date):
        """根据品种编码和指定日期，根据交易情况，确定对应的主力合约"""
        
        #取得当前月，下个月，下下个月3个月份合约名称
        contract_names = self.extractor.get_likely_main_contract_names(instrument, date)
        # 检查潜在合约的上一日成交金额，如果超出10%则进行合约切换
        contract_mapping = {}
        volume_main = 0
        main_name = None
        for symbol in contract_names:
            item_df = self.load_item_day_data(symbol,date)
            if item_df is None or item_df.shape[0]==0:
                continue
            cur_volume = item_df['volume'].values[0]
            if volume_main<cur_volume:
                volume_main = cur_volume
                main_name = item_df['code'].values[0]
            
        return main_name       
        
        
    def get_k_data(self, instrument, start_dt, end_dt,frequency=None,need_prev=True,institution=False):
        """从已下载的文件中，加载K线数据"""
        
        # 可以加载不同的频次类型数据
        if frequency=="1m":
            period = PeriodType.MIN1.value
            item_data = self.extractor.load_item_df(instrument,period=period)   
        # 日线数据使用akshare的数据源
        if frequency=="1d":
            start_dt = dt_obj(start_dt.year, start_dt.month, start_dt.day).date() 
            end_dt = dt_obj(end_dt.year, end_dt.month, end_dt.day).date()
            item_data = self.load_item_allday_data(instrument.order_book_id)
            item_data = item_data.rename(columns={"date":"datetime"})
        # 筛选对应日期以及合约的相关数据
        item_data = item_data[(item_data["datetime"]>=start_dt)&(item_data["datetime"]<=end_dt)]
        if item_data.shape[0]==0:
            return None
        # 改变为rqalpha格式
        item_data["last"] = item_data["close"]
        # 取得前一个交易时段收盘
        item_data["prev_close"]= np.NaN
        if need_prev:
            item_data["prev_close"] = self._get_prev_close(instrument, start_dt,frequency=frequency)
        # item_data = item_data.iloc[0].to_dict()
        return item_data

    def get_bar(self, instrument, dt, frequency,need_prev=True):
        if frequency != '1m' and frequency != '1d':
            return super(FuturesDataSource, self).get_bar(instrument, dt, frequency)
        
        if dt.hour<=9 and dt.minute<35:
            # 如果不在交易时间，则取上一日，否则取上一个分时
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
            return bar_data[fields].values
    
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
            bar = self.get_bar(instrument,dt,frequency)
        tick_obj = TickObject(instrument, bar)
        return tick_obj

    def _get_prev_close(self, instrument, dt,frequency=None):
        """取得上一交易时段的收盘价"""
        
        # 根据当前时间点，取得上一时间点
        if frequency=="1m":
            # 分钟级别，使用datetime进行直接计算
            prev_datetime = dt - datetime.timedelta(minutes=1)  
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

    def load_item_day_data(self,symbol,date):  
        """加载指定日期和合约名称的日线数据"""
        
        item_sql = "select code,date,open,close,high,low,volume,hold,settle from dominant_real_data where code='{}' " \
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
            item_code = contract_code[:-4]
            item_sql = "select name,code,multiplier,limit_rate,magin_radio from trading_variety where code='{}' ".format(item_code)
        else:
            item_sql = "select name,code,multiplier,limit_rate,magin_radio from trading_variety where isnull(magin_radio)=0"
        result_rows = self.dbaccessor.do_query(item_sql)  
        result_arr = np.array(result_rows)         
        return pd.DataFrame(result_arr,columns=["name","code","multiplier","limit_rate","magin_radio"])
        
    def get_all_contract_names(self,date):
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
    
    def get_trading_minutes_for(self,instrument, trading_dt):
        """取得对应合约的交易时间段"""
        
        contract_symbol = instrument.order_book_id
        code = contract_symbol[:-4]
        # 从数据库取得对应品种的交易时间范围
        sql = "select code,ac_time_range,day_time_range,night_time_range from trading_variety where code='{}' ".format(code)
        result_rows = self.dbaccessor.do_query(sql)  
        if len(result_rows)==0:
            return None
        ac_time_range = result_rows[0][1]
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
            start = trading_dt.strftime('%Y-%m-%d ') + begin_end[0].strip()
            end = trading_dt.strftime('%Y-%m-%d ') + begin_end[1].strip()
            min_list = pd.date_range(start=start, end=end,freq='min')
            trading_minutes.update(set(min_list.to_pydatetime().tolist()))
            
        return trading_minutes
        