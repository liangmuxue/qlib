from data_extract.rqalpha.futures_ds import FuturesDataSource

import datetime
import time
from datetime import date
from datetime import datetime as dt_obj
import numpy as np
import pandas as pd
import pickle
import os

from trader.utils.date_util import get_tradedays_dur
from cus_utils.db_accessor import DbAccessor
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType
from data_extract.juejin_futures_extractor import JuejinFuturesExtractor
from data_extract.akshare_extractor import AkExtractor
from trader.rqalpha.dict_mapping import judge_market,transfer_instrument
import cus_utils.global_var as global_var
from data_extract.akshare_futures_extractor import AkFuturesExtractor
from data_extract import akshare_futures_extractor
from gunicorn import instrument

class FuturesRealDataSource(FuturesDataSource):
    """期货自定义实时数据源"""
    
    def __init__(self,stock_data_path=None,sim_path=None,frequency_sim=True):
        
        self.dbaccessor = DbAccessor({})
        self.busi_columns = ["code","datetime","open","high","low","close","volume","hold","settle"]
        self.day_contract_columns = ["symbol","contract","volume"]
        
        self.extractor = JuejinFuturesExtractor(savepath=stock_data_path,sim_path=sim_path)
        self.extractor_ak = AkFuturesExtractor(savepath=stock_data_path)
        # 是否实时模式
        self.frequency_sim = frequency_sim 
        
    
    # def load_all_contract_conv(self):  
    #
    #     ret = self.extractor_ak.get_day_contract()
    #     self.all_contracts = pd.DataFrame(ret,columns='')
    
    def load_all_contract(self,cache_mode=True):
        """预加载所有合约"""
        
        contract_data_file = "custom/data/results/all_contract.pkl"
        
        if cache_mode and os.path.exists(contract_data_file):
            # 临时缓存模式
            with open(contract_data_file, "rb") as fin:
                all_contracts = pickle.load(fin)    
        else:          
            all_contracts = []
            contract_info = self.extractor_ak.get_day_contract()
            for symbol in contract_info:
                main_code = symbol[:-4]
                all_contracts.append([main_code,symbol])
            all_contracts = pd.DataFrame(np.array(all_contracts),columns=['code','main_contract_code'])
            
        self.all_contracts = all_contracts
        if cache_mode:
            with open(contract_data_file, "wb") as fout:
                pickle.dump(all_contracts, fout)              
    
    def get_last_price(self,order_book_id,dt=None):
        """取得指定标的最近报价信息"""

        instrument_code = order_book_id[:-4]
        exchange_code = self.get_exchange_from_instrument(instrument_code)
        realtime_data = self.extractor_ak.get_realtime_data(order_book_id, exchange_code)
        if realtime_data is None:
            return None
        return realtime_data['close'].values[0]    

    def get_last_bar(self,order_book_id,dt=None):
        """取得指定标的最近报价信息"""

        instrument_code = order_book_id[:-4]
        exchange_code = self.get_exchange_from_instrument(instrument_code)
        realtime_data = self.extractor_ak.get_realtime_data(order_book_id, exchange_code)
        if realtime_data is None:
            return None
        return realtime_data.iloc[0].to_dict()
    
    def get_main_contract_name(self,instrument,date):
        """根据品种编码，确定当前对应的主力合约"""
        
        item = self.all_contracts[self.all_contracts['code']==instrument]
        if item.shape[0]==0:
            return None
        return item['main_contract_code'].values[0]
            
    def get_all_contract_names(self,date):
        """取得所有合约名称（用于订阅）"""
        
        contract_names = self.extractor_ak.get_day_contract()
        return contract_names        
            