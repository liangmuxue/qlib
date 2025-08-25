import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from chinese_calendar import is_holiday
from trader.utils.date_util import get_tradedays_dur,get_next_working_day,get_prev_working_day

from data_extract.rqalpha.futures_ds import FuturesDataSource
from trader.emulator.futures_real_ds import FuturesDataSourceSql

# 回测整合记录项：date--日期，instrument--品种代码，order_book_id--合约编码，
#       open_price--开仓价,close_price--平仓价,volume--成交量,direction--成交方向(LONG-多方，SHORT-空方)

BACKTEST_RESULT_COLUMNS = ["date","instrument","order_book_id","open_price",
                       "close_price","volume","direction"]     

# 预测整合记录项：date--日期，instrument--品种代码，pred_trend--预测趋势，prev_close--预测前收盘价，
#       firstday_open,firstday_close,firstday_high,firstday_low,--第1天开盘、收盘、最高、最低
#       secondday_open,secondday_close,secondday_high,secondday_low,--第2天开盘、收盘、最高、最低

PRED_RESULT_COLUMNS = ["date","instrument","pred_trend","prev_close",
                       "firstday_open","firstday_close","firstday_high","firstday_low",
                       "secondday_open","secondday_close","secondday_high","secondday_low"]  

class DataStats(object):
    """针对预测、回测、模拟、交易的综合统计"""
    
    def __init__(self,work_dir=None,backtest_dir=None):
        
        self.work_dir = work_dir
        self.backtest_dir = backtest_dir
        
        stock_data_path = "/home/qdata/futures_data/juejin/main_1min"
        self.ds = FuturesDataSourceSql("/home/liang/.rqalpha/bundle",stock_data_path=stock_data_path,sim_path=stock_data_path,frequency_sim=False)        

    def combine_pred_result(self):
        """预测数据中整合实际结果"""

        step2_file = os.path.join(self.work_dir,"step2_rs.pkl")
        with open(step2_file, "rb") as fin:
            instrument_result_data = pickle.load(fin)  
                    
        date_list = instrument_result_data['date'].sort_values().unique()
        results = []
        for date in date_list:
            cur_day = datetime.strptime(str(date), "%Y%m%d")
            prev_day = get_prev_working_day(cur_day)
            second_day = get_next_working_day(cur_day)
            date_item = instrument_result_data[instrument_result_data['date']==date]
            for index,item in date_item.iterrows():
                instrument = item['instrument']
                pred_trend = item['trend_flag']
                prev_bar = self.ds.get_continue_data_by_day(instrument, prev_day.strftime("%Y%m%d")).iloc[0]
                first_bar = self.ds.get_continue_data_by_day(instrument, cur_day.strftime("%Y%m%d")).iloc[0]
                second_bar = self.ds.get_continue_data_by_day(instrument, second_day.strftime("%Y%m%d")).iloc[0]
                item = [date,instrument,pred_trend,prev_bar['settle'],
                        first_bar['open'],first_bar['settle'],first_bar['high'],first_bar['low'],
                        second_bar['open'],second_bar['settle'],second_bar['high'],second_bar['low']]
                results.append(item)
        results = pd.DataFrame(np.array(results),columns=PRED_RESULT_COLUMNS) 
        print(results)

    def combine_backtest_result(self):
        """回测数据中整合实际结果"""

        trading_file = os.path.join(self.backtest_dir,"trade_data.csv")
        trade_data_df = pd.read_csv(trading_file,parse_dates=['trade_date'],infer_datetime_format=True)   
        trade_data_df = trade_data_df.sort_values(by=["trade_date","order_book_id"])
                    
        date_list = instrument_result_data['date'].sort_values().unique()
        results = []
        for date in date_list:
            cur_day = datetime.strptime(str(date), "%Y%m%d")
            prev_day = get_prev_working_day(cur_day)
            second_day = get_next_working_day(cur_day)
            date_item = instrument_result_data[instrument_result_data['date']==date]
            for index,item in date_item.iterrows():
                instrument = item['instrument']
                pred_trend = item['trend_flag']
                prev_bar = self.ds.get_continue_data_by_day(instrument, prev_day.strftime("%Y%m%d")).iloc[0]
                first_bar = self.ds.get_continue_data_by_day(instrument, cur_day.strftime("%Y%m%d")).iloc[0]
                second_bar = self.ds.get_continue_data_by_day(instrument, second_day.strftime("%Y%m%d")).iloc[0]
                item = [date,instrument,pred_trend,prev_bar['settle'],
                        first_bar['open'],first_bar['settle'],first_bar['high'],first_bar['low'],
                        second_bar['open'],second_bar['settle'],second_bar['high'],second_bar['low']]
                results.append(item)
        results = pd.DataFrame(np.array(results),columns=PRED_RESULT_COLUMNS) 
        print(results)
                
    def check_step_output(self):
        
        step1_file = os.path.join(self.work_dir,"step1_rs.pkl")
        step2_file = os.path.join(self.work_dir,"step2_rs.pkl")
        with open(step1_file, "rb") as fin:
            indus_result_data = pickle.load(fin)          
        with open(step2_file, "rb") as fin:
            instrument_result_data = pickle.load(fin)  
        print(indus_result_data)    
        
    def mock_pred_data(self):      
        
        result_file_path = "/home/qdata/workflow/fur_sim_flow_2025/task/162/dump_data/pred_result.pkl"
        with open(result_file_path, "rb") as fin:
            result_data = pickle.load(fin)    
        result_data['date'] = 20250824
        # result_data.loc[result_data['instrument']=='ZC','instrument'] = 'AP'
        # result_data.loc[result_data['instrument']=='RR','instrument'] = 'CJ'
        # result_data.loc[result_data['instrument']=='PK','instrument'] = 'JD'
        with open(result_file_path, "wb") as fout:
            pickle.dump(result_data, fout)          
        print(result_data)
        
if __name__ == "__main__":
    stats = DataStats(work_dir="custom/data/results/stats",backtest_dir="/home/qdata/workflow/fur_backtest_flow/trader_data/08")  
    # stats.check_step_output()
    # stats.combine_pred_result()
    stats.mock_pred_data()
    
    
    
    
    