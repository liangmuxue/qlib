from data_extract.his_data_extractor import HisDataExtractor
from cus_utils.http_capable import TimeoutHTTPAdapter
from data_extract.his_data_extractor import HisDataExtractor,PeriodType,MarketType

import math
import pickle
import os

import akshare as ak
from akshare.stock_feature.stock_hist_em import (
    code_id_map_em
)
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import requests

from cus_utils.log_util import AppLogger

logger = AppLogger()

class AkExtractor(HisDataExtractor):
    """akshare数据源"""

    def __init__(self, backend_channel="ak",savepath=None,**kwargs):
        
        super().__init__(backend_channel=backend_channel,savepath=savepath,kwargs=kwargs)
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
          
    def extract_item_data(self,instrument_code,start_date=None,end_date=None,period=None,market=MarketType.SH.value,institution=False,no_total_file=False):  
        """取得单个股票历史行情数据"""
        
        # AKSHARE目前只支持按照日导入
        if period!=PeriodType.DAY.value:
            raise NotImplementedError
        
        # 取得日线数据，后复权
        if start_date is None:
            start_date = "19700101"
        if end_date is None:
            end_date = "20500101"     
        
        # 根据参数决定是否取得复权数据
        if institution:
            adjust="hfq"
        else:
            adjust=""
        def get_data():
            i = 0
            # 超时策略
            while i < 3:
                try:
                    item_data = self.stock_zh_a_hist(symbol=instrument_code, period="daily", start_date=start_date,end_date=end_date,adjust=adjust)
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

    def stock_individual_info_em(self,symbol) -> pd.DataFrame:
        """
        东方财富-个股-股票信息
        https://quote.eastmoney.com/concept/sh603777.html?from=classic
        :param symbol: 股票代码
        :type symbol: str
        :return: 股票信息
        :rtype: pandas.DataFrame
        """
        code_id_dict = code_id_map_em()
        url = "http://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            "fltt": "2",
            "invt": "2",
            "fields": "f120,f121,f122,f174,f175,f59,f163,f43,f57,f58,f169,f170,f46,f44,f51,f168,f47,f164,f116,f60,f45,f52,f50,f48,f167,f117,f71,f161,f49,f530,f135,f136,f137,f138,f139,f141,f142,f144,f145,f147,f148,f140,f143,f146,f149,f55,f62,f162,f92,f173,f104,f105,f84,f85,f183,f184,f185,f186,f187,f188,f189,f190,f191,f192,f107,f111,f86,f177,f78,f110,f262,f263,f264,f267,f268,f255,f256,f257,f258,f127,f199,f128,f198,f259,f260,f261,f171,f277,f278,f279,f288,f152,f250,f251,f252,f253,f254,f269,f270,f271,f272,f273,f274,f275,f276,f265,f266,f289,f290,f286,f285,f292,f293,f294,f295",
            "secid": f"{code_id_dict[symbol]}.{symbol}",
            "_": "1640157544804",
        }
        r = self.session.get(url, params=params)
        data_json = r.json()
        temp_df = pd.DataFrame(data_json)
        temp_df.reset_index(inplace=True)
        del temp_df["rc"]
        del temp_df["rt"]
        del temp_df["svr"]
        del temp_df["lt"]
        del temp_df["full"]
        code_name_map = {
            "f57": "股票代码",
            "f58": "股票简称",
            "f84": "总股本",
            "f85": "流通股",
            "f127": "行业",
            "f116": "总市值",
            "f117": "流通市值",
            "f189": "上市时间",
        }
        temp_df["index"] = temp_df["index"].map(code_name_map)
        temp_df = temp_df[pd.notna(temp_df["index"])]
        if "dlmkts" in temp_df.columns:
            del temp_df["dlmkts"]
        temp_df.columns = [
            "item",
            "value",
        ]
        temp_df.reset_index(inplace=True, drop=True)
        return temp_df
                 
    def extract_base_info(self,instrument):  
        """取得股票的基础信息"""

        def get_data():
            i = 0
            # 超时策略
            while i < 3:
                try:
                    item_data = self.stock_individual_info_em(symbol=instrument)  
                    return item_data
                except requests.exceptions.RequestException:
                    logger.warning("request timeout:{},try again".format(instrument)) 
                    i += 1    
            logger.warning("request end with timeout:{}".format(instrument)) 
            return None
                    
        stock_individual_info_em_df = get_data()
        if stock_individual_info_em_df is None or stock_individual_info_em_df.shape[0]==0:
            return None
               
        base_data = {}
        base_data["industry"] = stock_individual_info_em_df.values[2][1]
        base_data["total_capital"] = stock_individual_info_em_df.values[6][1]
        base_data["tradable_shares"] = stock_individual_info_em_df.values[7][1]
        return base_data

    def clear_redun_data(self,ori_data_item,date):
        """清除重复部分的数据"""
        
        data_item = ori_data_item[ori_data_item["datetime"]<date]
        return data_item  


    def sw_index_second_info(self) -> pd.DataFrame:
        """手工解析乐股网二级行业分类数据 https://legulegu.com/stockdata/sw-industry-overview"""
    
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        }       
        url = "https://legulegu.com/stockdata/sw-industry-overview"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, features="lxml")
        code_raw = soup.find(name="div", attrs={"id": "level2Items"}).find_all(
            name="div", attrs={"class": "lg-industries-item-chinese-title"}
        )
        name_raw = soup.find(name="div", attrs={"id": "level2Items"}).find_all(
            name="div", attrs={"class": "lg-industries-item-number"}
        )
        parent_name_raw = soup.find(name="div", attrs={"id": "level2Items"}).find_all(
            name="span", attrs={"class": "parent-industry-name"}
        )     
        value_raw = soup.find(name="div", attrs={"id": "level2Items"}).find_all(
            name="div", attrs={"class": "lg-sw-industries-item-value"}
        )
        code = [item.get_text() for item in code_raw]
        name = [item.get_text().split("(")[0] for item in name_raw]
        parent_name = [item.get_text()[1:-1] for item in parent_name_raw]
        num = [item.get_text().split("(")[1].split(")")[0] for item in name_raw]
        num_1 = [
            item.find_all("span", attrs={"class": "value"})[0].get_text().strip()
            for item in value_raw
        ]
        num_2 = [
            item.find_all("span", attrs={"class": "value"})[1].get_text().strip()
            for item in value_raw
        ]
        num_3 = [
            item.find_all("span", attrs={"class": "value"})[2].get_text().strip()
            for item in value_raw
        ]
        num_4 = [
            item.find_all("span", attrs={"class": "value"})[3].get_text().strip()
            for item in value_raw
        ]
        temp_df = pd.DataFrame([code, name,parent_name, num, num_1, num_2, num_3, num_4]).T
        temp_df.columns = [
            "行业代码",
            "行业名称",
            "上级行业名称",
            "成份个数",
            "静态市盈率",
            "TTM(滚动)市盈率",
            "市净率",
            "静态股息率",
        ]
        temp_df["成份个数"] = pd.to_numeric(temp_df["成份个数"], errors="coerce")
        temp_df["静态市盈率"] = pd.to_numeric(temp_df["静态市盈率"], errors="coerce")
        temp_df["TTM(滚动)市盈率"] = pd.to_numeric(
            temp_df["TTM(滚动)市盈率"], errors="coerce"
        )
        temp_df["市净率"] = pd.to_numeric(temp_df["市净率"], errors="coerce")
        temp_df["静态股息率"] = pd.to_numeric(temp_df["静态股息率"], errors="coerce")
        return temp_df
    
    def sw_index_third_info(self) -> pd.DataFrame:
        """手工解析乐股网三级行业分类数据 https://legulegu.com/stockdata/sw-industry-overview"""
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        }        
        url = "https://legulegu.com/stockdata/sw-industry-overview"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, features="lxml")
        code_raw = soup.find(name="div", attrs={"id": "level3Items"}).find_all(
            name="div", attrs={"class": "lg-industries-item-chinese-title"}
        )
        name_raw = soup.find(name="div", attrs={"id": "level3Items"}).find_all(
            name="div", attrs={"class": "lg-industries-item-number"}
        )
        parent_name_raw = soup.find(name="div", attrs={"id": "level3Items"}).find_all(
            name="span", attrs={"class": "parent-industry-name"}
        )        
        value_raw = soup.find(name="div", attrs={"id": "level3Items"}).find_all(
            name="div", attrs={"class": "lg-sw-industries-item-value"}
        )
        code = [item.get_text() for item in code_raw]
        name = [item.get_text().split("(")[0] for item in name_raw]
        parent_name = [item.get_text()[1:-1] for item in parent_name_raw]
        num = [item.get_text().split("(")[1].split(")")[0] for item in name_raw]
        num_1 = [
            item.find_all("span", attrs={"class": "value"})[0].get_text().strip()
            for item in value_raw
        ]
        num_2 = [
            item.find_all("span", attrs={"class": "value"})[1].get_text().strip()
            for item in value_raw
        ]
        num_3 = [
            item.find_all("span", attrs={"class": "value"})[2].get_text().strip()
            for item in value_raw
        ]
        num_4 = [
            item.find_all("span", attrs={"class": "value"})[3].get_text().strip()
            for item in value_raw
        ]
        temp_df = pd.DataFrame([code, name,parent_name, num, num_1, num_2, num_3, num_4]).T
        temp_df.columns = [
            "行业代码",
            "行业名称",
            "上级行业名称",
            "成份个数",
            "静态市盈率",
            "TTM(滚动)市盈率",
            "市净率",
            "静态股息率",
        ]
        temp_df["成份个数"] = pd.to_numeric(temp_df["成份个数"], errors="coerce")
        temp_df["静态市盈率"] = pd.to_numeric(temp_df["静态市盈率"], errors="coerce")
        temp_df["TTM(滚动)市盈率"] = pd.to_numeric(
            temp_df["TTM(滚动)市盈率"], errors="coerce"
        )
        temp_df["市净率"] = pd.to_numeric(temp_df["市净率"], errors="coerce")
        temp_df["静态股息率"] = pd.to_numeric(temp_df["静态股息率"], errors="coerce")
        return temp_df
 
    def download_sw_index(self):   
        """获取申银万国股票指数数据"""
     
        # 分别下载3级分类，并合并到一个数据文件
        sw_index_first_info_df = ak.sw_index_first_info()
        index_analysis_monthly_sw_df = ak.index_analysis_monthly_sw(symbol="市场表征", date="20240329")
        # 保存到数据库
        insert_sql = "insert into sw_index(code,name) values(%s,%s)"
        for idx,row in index_analysis_monthly_sw_df.iterrows():
            self.dbaccessor.do_inserto_withparams(insert_sql, (row['指数代码'],row['指数名称']))     

    def get_sw_index_day_data(self,indus_file_path=None):   
        """取得申万指数的历史交易数据"""
     
        # 从之前存储中取得指数类别数据
        sql = "select code,name from sw_index"
        result_rows = self.dbaccessor.do_query(sql)        
        
        # 遍历各类别，并逐个获取对应日K数据
        upt_sql = "update instrument_info set sw_industry=%s where code=%s"
        code_stats = []
        for result in result_rows:
            code = result[0]
            print("process code:{}".format(code))
            # 调用api，获取当前行业类别下的成分股
            try:
                index_hist_sw_df = ak.index_hist_sw(symbol=code, period="day")
            except Exception:
                print("index_hist_sw api fail for:{}".format(code))
                continue
            file_path = os.path.join(indus_file_path,code+".csv")   
            # 中文标题改英文 
            index_hist_sw_df.columns = self.sw_indus_k_columns    
            # 统一顺序
            index_hist_sw_df = index_hist_sw_df[self.busi_columns[:8]]     
            index_hist_sw_df.to_csv(file_path,index=False)  
                       
    def download_industry_data(self):   
        """生成行业分类数据（申银万国）"""
     
        # 分别下载3级分类，并合并到一个数据文件
        sw_index_first_info_df = ak.sw_index_first_info()
        # 二级和三级使用自定义解析，用于补充上级分类
        sw_index_second_info_df = self.sw_index_second_info()
        sw_index_third_info_df = self.sw_index_third_info()  
        # 手动添加级别数 
        sw_index_first_info_df['level'] = 1    
        sw_index_first_info_df['上级行业名称'] = 0
        sw_index_second_info_df['level'] = 2    
        sw_index_third_info_df['level'] = 3  
        # 合并
        total_df = pd.concat([sw_index_first_info_df,sw_index_second_info_df,sw_index_third_info_df])
        # 保存到数据库
        insert_sql = "insert into sw_industry(code,name,parent_name,level,cons_num,static_pe,dynamic_pe,pb,yield) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        for idx,row in total_df.iterrows():
            self.dbaccessor.do_inserto_withparams(insert_sql, 
                        (row['行业代码'],row['行业名称'],row['上级行业名称'],row['level'],row['成份个数'],row['静态市盈率'],row['TTM(滚动)市盈率'],row['市净率'],row['静态股息率']))     
        # 统一挂接上级行业
        upt_sql = "update sw_industry swi,sw_industry sw_ass set swi.parent_code=sw_ass.code where swi.parent_name=sw_ass.name and swi.level=2"
        self.dbaccessor.do_updateto(upt_sql)
        upt_sql = "update sw_industry swi,sw_industry sw_ass set swi.parent_code=sw_ass.code where swi.parent_name=sw_ass.name and swi.level=3"
        self.dbaccessor.do_updateto(upt_sql)
                
    def create_industry_cons_data(self,indus_file_path=None):   
        """下载生成行业分类成分股数据（申银万国）"""
     
        # 从之前存储中取得行业类别数据，第三级别
        sql = "select code,name,level,cons_num,static_pe,dynamic_pe,pb,yield from sw_industry where level=3"
        result_rows = self.dbaccessor.do_query(sql)        
        
        # 遍历第三级分类，并逐个获取对应成分股
        upt_sql = "update instrument_info set sw_industry=%s where code=%s"
        code_stats = []
        for result in result_rows:
            code = result[0]
            print("process code:{}".format(code))
            # 调用api，获取当前行业类别下的成分股
            try:
                sw_index_third_cons_df = ak.sw_index_third_cons(symbol=code)
            except Exception:
                print("api fail for:{}".format(code))
                continue
            # 去除后缀，使股票代码保持一致
            for idx,row in sw_index_third_cons_df.iterrows():
                instrument_code = row['股票代码']
                # 去除前缀，和本地编码保持一致
                instrument_code = instrument_code[:-3]
                code_stats.append(int(instrument_code))
                self.dbaccessor.do_inserto_withparams(upt_sql, (code,instrument_code))      
                        
        code_stats_unique = list(set(code_stats))  
        print("code_stats len:{},code_stats_unique len:{}".format(len(code_stats),len(code_stats_unique)))
        
    def get_industry_day_data(self,indus_file_path=None):   
        """取得申万行业的历史交易数据"""
     
        # 从之前存储中取得行业类别数据
        sql = "select code,name,level,cons_num,static_pe,dynamic_pe,pb,yield from sw_industry"
        result_rows = self.dbaccessor.do_query(sql)        
        
        # 遍历各类别，并逐个获取对应日K数据
        upt_sql = "update instrument_info set sw_industry=%s where code=%s"
        code_stats = []
        for result in result_rows:
            code = result[0]
            code = code[:-3]
            print("process code:{}".format(code))
            # 调用api，获取当前行业类别下的成分股
            try:
                index_hist_sw_df = ak.index_hist_sw(symbol=code, period="day")
            except Exception:
                print("index_hist_sw api fail for:{}".format(code))
                continue
            file_path = os.path.join(indus_file_path,code+".csv")   
            # 中文标题改英文 
            index_hist_sw_df.columns = self.sw_indus_k_columns    
            # 统一顺序
            index_hist_sw_df = index_hist_sw_df[self.busi_columns[:8]]     
            index_hist_sw_df.to_csv(file_path,index=False)            
         


            