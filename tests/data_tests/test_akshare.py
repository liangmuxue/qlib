import akshare as ak
from trader.utils.date_util import get_tradedays_dur
import datetime
import pandas as pd

def test_day():
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="600528", period="daily", start_date="20080101", end_date='20221231', adjust="hfq")
    print(stock_zh_a_hist_df.datetime.min())

def test_minute():
    
    # stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", 
    #                                 start_date="2023-02-01 09:32:00", end_date="2023-02-06 09:32:00", period='5', adjust='')
    # ak_hist_df = ak.stock_zh_a_hist_min_em(symbol='sh000300',
    #                                    start_date='2023-05-04 10:30:00',
    #                                    end_date='2023-06-02 15:00:00', period='30')
    stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol='sh000300', period='1')
    print(stock_zh_a_minute_df)

def test_base_info():
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol="600519")
    print(stock_individual_info_em_df)

def test_trade_date():
    dt = datetime.datetime(2022,1,31) 
    prev_date = get_tradedays_dur(dt,-1) 
    print("prev_date:",prev_date)

def test_industry():
    # stock_industry_clf_hist_sw_df = ak.stock_industry_clf_hist_sw()
    # print(stock_industry_clf_hist_sw_df)
    sw_index_first_info_df = ak.sw_index_first_info()
    print(sw_index_first_info_df)
    sw_index_first_info_df.to_csv("custom/data/results/sw_index_first_info_df.csv", encoding="gbk", index=False)    
    # sw_index_second_info_df = ak.sw_index_second_info()
    # print(sw_index_second_info_df)
    # sw_index_second_info_df.to_csv("custom/data/results/sw_index_second_info_df.csv", encoding="gbk", index=False)
    # sw_index_third_info_df = ak.sw_index_third_info()
    # print(sw_index_third_info_df)    
    # sw_index_third_info_df.to_csv("custom/data/results/sw_index_third_info_df.csv", encoding="gbk", index=False)
    # sw_index_third_cons_df = ak.sw_index_third_cons(symbol="801016.SI")
    # print(sw_index_third_cons_df)
    # sw_index_third_cons_df.to_csv("custom/data/results/sw_index_third_cons_df.csv", encoding="gbk", index=False)
    # stock_industry_category_cninfo_df = ak.stock_industry_category_cninfo(symbol="申银万国行业分类标准")
    # print(stock_industry_category_cninfo_df)    
    # stock_industry_category_cninfo_df.to_csv("custom/data/results/shenyin_industry.csv", encoding="gbk", index=False)
    # stock_industry_clf_hist_sw_df = ak.stock_industry_clf_hist_sw()
    # print(stock_industry_clf_hist_sw_df)
 
def test_his_shenwan():
    index_hist_sw_df = ak.index_hist_sw(symbol="801193", period="day")
    print(index_hist_sw_df)   
    index_hist_sw_df.to_csv("custom/data/results/801193.csv", encoding="gbk", index=False)
    
def test_bao():
    import baostock as bs
    
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    
    # rs = bs.query_history_k_data_plus("sz.000001",
    #     "date,time,code,open,high,low,close,volume,amount,adjustflag",
    #     start_date='2017-07-01', end_date='2023-12-31',
    #     frequency="5", adjustflag="3")
    # print('query_history_k_data_plus respond error_code:'+rs.error_code)
    # print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    #
    # data_list = []
    # while (rs.error_code == '0') & rs.next():
    #     data_list.append(rs.get_row_data())
    # result = pd.DataFrame(data_list, columns=rs.fields)
    
    # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    # print(result)
    
    rs = bs.query_stock_industry()
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    result.to_csv("custom/data/results/bao_stock_industry.csv", encoding="gbk", index=False)
    print(result)    
    
    bs.logout()
    
if __name__ == "__main__":
    # test_minute()
    # test_bao()
    # test_industry()
    test_his_shenwan()
    # test_base_info()
    # test_day()
    # test_trade_date()