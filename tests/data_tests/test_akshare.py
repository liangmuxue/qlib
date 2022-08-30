import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="600528", period="daily", start_date="20170301", end_date='20210907', adjust="hfq")
print(stock_zh_a_hist_df)