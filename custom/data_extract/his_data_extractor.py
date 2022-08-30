# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import requests
import datetime
from tqdm import tqdm
import akshare as ak
import pandas as pd

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

    def get_code_data(self,create=True):
        """取得所有股票代码"""
        
        if create is True:
            all_sz = ak.stock_info_sz_name_code(indicator="A股列表")
            all_sh1 = ak.stock_info_sh_name_code(indicator="主板A股")   
            df_sz = all_sz.iloc[:,1]
            df_sh = all_sh1.iloc[:,2]      
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
            # df_sh.to_csv(sh_savepath,index=False)
            # df_sz.to_csv(sz_savepath,index=False)
               
    def load_all_data(self):
        """取得所有股票历史行情数据"""
        
        stock_item = {'sz': self.stock_sz, 'sh': self.stock_sh}
        stock_item = {'sh': self.stock_sh}
        for key, value in stock_item.items():
            for single_stock in value:
                try:
                    item_data = self.load_item_data(single_stock)
                    print("save item data ok:{}".format(single_stock))
                except Exception as e:
                    print("load item {} err".format(single_stock),e)
                
    def load_item_data(self,instrument_code,start_date=None,end_date=None):   
        """取得单个股票历史行情数据"""
        
        # 取得日线数据，后复权
        item_data = ak.stock_zh_a_hist(symbol=instrument_code, period="daily", start_date=start_date,end_date=end_date,adjust="hfq")
        item_data.insert(loc=0, column='code', value=instrument_code)
        # 中文标题改英文 
        item_data.columns = ["code","date","open","close","high","low","volume","amount","amplitude","flu_range","flu_amount","turnover"]
        ori_len = item_data.shape[0]
        item_data = self.data_clean(item_data)
        if ori_len>item_data.shape[0]:
            print("clean some:{},less:{}".format(instrument_code,ori_len-item_data.shape[0]))
            
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
           
if __name__ == "__main__":    
    extractor = HisDataExtractor(item_savepath="./custom/data/stock_data/item")   
    # extractor.load_item_data("600520",start_date="2008-05-21",end_date="2008-05-28")    
    extractor.load_item_data("600007",start_date="20161227",end_date="20161230")
    # extractor.show_item_data("600010",start_date="2020-05-07",end_date="2020-05-29")
    # extractor = HisDataExtractor()
    # extractor.download_data(file_type="qyspjg")
    # extractor.get_code_data(create=False)aq
    # extractor.load_all_data()
    
        
