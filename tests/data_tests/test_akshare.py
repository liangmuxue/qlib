import akshare as ak
from trader.utils.date_util import get_tradedays_dur
from datetime import datetime
def test_day():
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="600528", period="daily", start_date="20080101", end_date='20221231', adjust="hfq")
    print(stock_zh_a_hist_df.datetime.min())

def test_minute():
    
    stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date="2023-02-01 09:32:00", end_date="2023-02-06 09:32:00", period='5', adjust='')
    print(stock_zh_a_hist_min_em_df)

def test_base_info():
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol="600519")
    print(stock_individual_info_em_df)

def test_trade_date():
    dt = datetime(2023,5,1) 
    prev_date = get_tradedays_dur(dt,-1) 
    print("prev_date:",prev_date)
       
if __name__ == "__main__":
    # test_minute()
    # test_base_info()
    # test_day()
    test_trade_date()