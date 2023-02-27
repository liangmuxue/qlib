# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import requests
import datetime
from tqdm import tqdm
import akshare as ak
import pandas as pd

from data_extract.http_capable import TimeoutHTTPAdapter

from cus_utils.log_util import AppLogger
logger = AppLogger()

from akshare.stock_feature.stock_hist_em import (
    code_id_map_em
)

class HisDataExtractor:
    """历史证券数据采集"""

    def __init__(self, backend_channel="ak",item_savepath=None):
        """

        Parameters
        ----------
        backend_channel : 采集源    ak: akshare数据源
        """
        
        self.CODE_DATA_SAVEPATH = "./custom/data/stock_data/"
        if item_savepath is None:
            self.ITEM_DATA_SAVEPATH = "./custom/data/stock_data/item"
        else:
            self.ITEM_DATA_SAVEPATH = item_savepath
        if backend_channel=="ak":
            self.source_url = ""
            
        self.session = requests.Session()
        self.session.mount("http://", TimeoutHTTPAdapter(timeout=(5,10)))

    def get_code_data(self,create=True):
        """取得所有股票代码"""
        
        if create is True:
            all_sz = ak.stock_info_sz_name_code(indicator="A股列表")
            all_sh1 = ak.stock_info_sh_name_code(indicator="主板A股")   
            df_sz = all_sz.iloc[:,1]
            df_sh = all_sh1.iloc[:,0]      
            self.stock_sz = np.hstack([np.array(df_sz), '399107'])
            self.stock_sh = np.hstack([np.array(df_sh), '000001'])                        
        else:
            # 直接读取已经保存的code文件
            all_sz = pd.read_csv(self.CODE_DATA_SAVEPATH + "/sz.csv")  
            all_sh1 = pd.read_csv(self.CODE_DATA_SAVEPATH + "/sh.csv")
            df_sz = all_sz
            df_sh = all_sh1    
            self.stock_sz = np.hstack([np.array(df_sz)[:,0], '399107'])
            self.stock_sh = np.hstack([np.array(df_sh)[:,0], '000001'])       
        
        # 保存代码数据到文件
        if create is True:
            sh_savepath = self.CODE_DATA_SAVEPATH + "/sh.csv"
            sz_savepath = self.CODE_DATA_SAVEPATH + "/sz.csv"
            df_sh.to_csv(sh_savepath,index=False)
            df_sz.to_csv(sz_savepath,index=False)
               
    def load_all_data(self,load_record=False):
        """取得所有股票历史行情数据"""
        
        record_path = self.CODE_DATA_SAVEPATH + "/item_record.npy"
        if load_record:
            record_arr = np.load(record_path)
        else:
            record_list = []
            record_arr = np.array(record_list)
        stock_item = {'sz': self.stock_sz, 'sh': self.stock_sh}
        stock_item = {'sh': self.stock_sh}
        
        index = 0
        for key, value in stock_item.items():
            for single_stock in value:
                index += 1
                # 如果已经导入过，则忽略
                if np.any(np.isin([single_stock],record_arr)):
                    # logger.debug("has record,ignore:{}".format(key))
                    continue
                try:
                    item_data = self.load_item_data(single_stock)
                    if item_data is None:
                        logger.warning("load item None:{}".format(single_stock))
                    else:
                        logger.info("save item data ok:{}".format(single_stock))
                        record_arr = np.append(record_arr,single_stock)
                except Exception as e:
                    logger.warning("load item {} err:{}".format(single_stock,e))
                
                if index%3==0:
                    # 保存已导入列表，用于后续断点续传
                    np.save(record_path,record_arr)
          
    def load_item_data(self,instrument_code,start_date=None,end_date=None):   
        """取得单个股票历史行情数据"""
        
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
        if item_data is None:
            return None
        item_data.insert(loc=0, column='code', value=instrument_code)
        # 中文标题改英文 
        item_data.columns = ["code","date","open","close","high","low","volume","amount","amplitude","flu_range","flu_amount","turnover"]
        ori_len = item_data.shape[0]
        item_data = self.data_clean(item_data)
        if ori_len>item_data.shape[0]:
            logger.debug("clean some:{},less:{}".format(instrument_code,ori_len-item_data.shape[0]))
            
        # 每个股票分别保存
        save_path = "{}/{}.csv".format(self.ITEM_DATA_SAVEPATH,instrument_code)
        item_data.to_csv(save_path, index=False)        
        return item_data

    def show_item_data(self,instrument_code,start_date=None,end_date=None):   
        """显示单个股票历史行情数据"""
        
        # 取得日线数据，后复权
        item_data = ak.stock_zh_a_hist(symbol=instrument_code, period="daily", start_date=start_date,end_date=end_date,adjust="hfq")
        item_data.insert(loc=0, column='code', value=instrument_code)
        print(item_data)
     
    def data_clean(self,item_data):
        """数据清洗"""
        
        # 收盘排查
        item_data_clean = item_data[item_data["close"]>0]
        item_data_clean = item_data_clean[item_data_clean["close"]/item_data_clean["open"]-1<0.2]
        # 高低价格排查
        item_data_clean = item_data_clean[(item_data_clean["high"]-item_data_clean["low"])>=0]
        
        return item_data_clean

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
           
if __name__ == "__main__":    
    extractor = HisDataExtractor(item_savepath="./custom/data/stock_data/item")   
    # extractor.load_item_data("600520",start_date="2008-05-21",end_date="2008-05-28")    
    # extractor.load_item_data("600060")
    # extractor.show_item_data("600010",start_date="2020-05-07",end_date="2020-05-29")
    # extractor = HisDataExtractor()
    # extractor.download_data(file_type="qyspjg")
    extractor.get_code_data(create=False)
    extractor.load_all_data(load_record=True)
    
        
