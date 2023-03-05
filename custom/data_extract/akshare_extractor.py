from .his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from .his_data_extractor import HisDataExtractor,PeriodType,MarketType

import akshare as ak
from akshare.stock_feature.stock_hist_em import (
    code_id_map_em
)

import numpy as np
import pandas as pd
import requests

from cus_utils.log_util import AppLogger

logger = AppLogger()

class AkExtractor(HisDataExtractor):
    """akshare数据源"""

    def __init__(self, backend_channel="ak",savepath=None):
        
        super().__init__(backend_channel=backend_channel,savepath=savepath)
        self.session = requests.Session()
        self.session.mount("http://", TimeoutHTTPAdapter(timeout=(5,10)))
      
    def extract_code_data(self):  
        """取得所有股票代码"""
        
        all_sz = ak.stock_info_sz_name_code(indicator="A股列表")
        all_sh = ak.stock_info_sh_name_code(symbol="主板A股")   
        all_sz["market"] = MarketType.SZ 
        all_sh["market"] = MarketType.SH 
        data_sz = all_sz[["A股代码","A股简称","market"]].values
        data_sh = all_sh[["证券代码","证券简称","market"]].values
        return np.concatenate((data_sh,data_sz),axis=0)
          
    def extract_item_data(self,instrument_code,start_date=None,end_date=None,period=None,market=MarketType.SH.value):  
        """取得单个股票历史行情数据"""
        
        # AKSHARE目前只支持按照日导入
        if period!=PeriodType.DAY.value:
            raise NotImplementedError
        
        # 取得日线数据，后复权
        if start_date is None:
            start_date = "19700101"
        if end_date is None:
            end_date = "20500101"     
             
        def get_data():
            i = 0
            # 超时策略
            while i < 3:
                try:
                    item_data = self.stock_zh_a_hist(symbol=instrument_code, period="daily", start_date=start_date,end_date=end_date,adjust="hfq")
                    return item_data
                except requests.exceptions.RequestException:
                    logger.warning("request timeout:{},try again".format(instrument_code)) 
                    i += 1    
            logger.warning("request end with timeout:{}".format(instrument_code)) 
            return None
                    
        item_data = get_data()
        if item_data is None or item_data.shape[0]==0:
            return None
        item_data.insert(loc=0, column='code', value=instrument_code)
        # 中文标题改英文 
        item_data.columns = self.busi_columns
        ori_len = item_data.shape[0]
        item_data = self.data_clean(item_data)
        if ori_len>item_data.shape[0]:
            logger.debug("clean some:{},less:{}".format(instrument_code,ori_len-item_data.shape[0]))
                 
        return item_data

    def data_clean(self,item_data):
        """数据清洗"""
        
        # 收盘排查
        item_data_clean = item_data[item_data["close"]>0]
        item_data_clean = item_data_clean[item_data_clean["close"]/item_data_clean["open"]-1<0.2]
        # 高低价格排查
        item_data_clean = item_data_clean[(item_data_clean["high"]-item_data_clean["low"])>=0]
        
        return item_data_clean
    
    def show_item_data(self,instrument_code,start_date=None,end_date=None):   
        """显示单个股票历史行情数据"""
        
        # 取得日线数据，后复权
        item_data = ak.stock_zh_a_hist(symbol=instrument_code, period="daily", start_date=start_date,end_date=end_date,adjust="hfq")
        item_data.insert(loc=0, column='code', value=instrument_code)
        print(item_data)

    def stock_zh_a_hist(
        self,
        symbol: str = "000001",
        period: str = "daily",
        start_date: str = "19700101",
        end_date: str = "20500101",
        adjust: str = "",
    ) -> pd.DataFrame:
        """重载原方法，改进连接问题"""
        code_id_dict = code_id_map_em()
        adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
        period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": period_dict[period],
            "fqt": adjust_dict[adjust],
            "secid": f"{code_id_dict[symbol]}.{symbol}",
            "beg": start_date,
            "end": end_date,
            "_": "1623766962675",
        }
        r = self.session.get(url, params=params)
        data_json = r.json()
        if not (data_json["data"] and data_json["data"]["klines"]):
            return pd.DataFrame()
        temp_df = pd.DataFrame(
            [item.split(",") for item in data_json["data"]["klines"]]
        )
        temp_df.columns = [
            "日期",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "振幅",
            "涨跌幅",
            "涨跌额",
            "换手率",
        ]
        temp_df.index = pd.to_datetime(temp_df["日期"])
        temp_df.reset_index(inplace=True, drop=True)
    
        temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
        temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
        temp_df["最高"] = pd.to_numeric(temp_df["最高"])
        temp_df["最低"] = pd.to_numeric(temp_df["最低"])
        temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
        temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
        temp_df["振幅"] = pd.to_numeric(temp_df["振幅"])
        temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"])
        temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"])
        temp_df["换手率"] = pd.to_numeric(temp_df["换手率"])
    
        return temp_df
                 
        
        
            