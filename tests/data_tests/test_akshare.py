import akshare as ak

# stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="600528", period="daily", start_date="20170301", end_date='20210907', adjust="hfq")
# print(stock_zh_a_hist_df)

def test_minute():
    
    stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(symbol="000001", start_date="2023-02-01 09:32:00", end_date="2023-02-06 09:32:00", period='5', adjust='')
    print(stock_zh_a_hist_min_em_df)


if __name__ == "__main__":
    test_minute()