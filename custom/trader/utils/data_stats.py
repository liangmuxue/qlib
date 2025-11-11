import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from chinese_calendar import is_holiday,get_workdays
from trader.utils.date_util import get_tradedays_dur,get_next_working_day,get_prev_working_day

from rqalpha.const import ORDER_STATUS,SIDE
from trader.emulator.futures_sql_ds import FuturesDataSourceSql
from backtrader.utils import date

RESULT_FILE_PATH = "custom/data/results/stats"
RESULT_FILE_STEP1 = "step1_rs.pkl"
RESULT_FILE_STEP2 = "step2_rs.pkl"
RESULT_FILE_VIEW = "coll_result.csv"
INTER_RS_FILEPATH = "pred_step1_rs.pkl"

# 回测整合记录项：date--日期，instrument--品种代码，trade_flag--交易标志('trade'-交易,'ignore'-忽略,'noclose'-无平仓,'lock'-锁定), order_book_id--合约编码，
#       open_price--开仓价,close_price--平仓价,volume--成交量,side--成交方向(BUY-多方,SELL-空方)

BACKTEST_RESULT_COLUMNS = ["date","instrument","order_book_id","trade_flag","open_price",
                       "close_price","quantity","side"]     

# 预测整合记录项：date--日期，instrument--品种代码，pred_trend--预测趋势，prev_close--预测前收盘价，
#       firstday_open,firstday_close,firstday_high,firstday_low,--第1天开盘、收盘、最高、最低
#       secondday_open,secondday_close,secondday_high,secondday_low,--第2天开盘、收盘、最高、最低

PRED_RESULT_COLUMNS = ["date","instrument","pred_trend","prev_close",
                       "firstday_open","firstday_close","firstday_high","firstday_low",
                       "thirdday_open","thirdday_close","thirdday_high","thirdday_low"]  

class DataStats(object):
    """针对预测、回测、模拟、交易的综合统计"""
    
    def __init__(self,work_dir=None,backtest_dir=None):
        
        self.work_dir = work_dir
        self.backtest_dir = backtest_dir
        self.combine_val_filepath = "val_combine.csv"
        self.combine_pred_filepath = "pred_combine.csv"
        self.combine_backtest_filepath = "backtest_combine.csv"
        self.combine_simulation_filepath = "simulation_combine.csv"
        self.backtest_analysys_filepath = "backtest_analysys.csv"
        
        stock_data_path = "/home/qdata/futures_data/juejin/main_1min"
        self.ds = FuturesDataSourceSql("/home/liang/.rqalpha/bundle",stock_data_path=stock_data_path,sim_path=stock_data_path,frequency_sim=False)        

    def combine_val_result(self,date_range=None):
        """验证数据中整合实际结果"""

        val_data_file = os.path.join(self.work_dir,RESULT_FILE_VIEW) 
        col_data_types = {"top_index":int,"instrument":str,"yield_rate":float,"result":int,"trend_value":int,"date":int}   
        instrument_result_data = pd.read_csv(val_data_file,dtype=col_data_types)
        results = self.compute_val_result(instrument_result_data, date_range)
        combine_data_file = os.path.join(self.work_dir,self.combine_val_filepath)
        results.to_csv(combine_data_file,index=False)  
    
    def compute_val_result(self,val_data,date_range=None):
        
        date_list = val_data['date'].sort_values().unique()
        results = []
        for date in date_list:
            cur_day = datetime.strptime(str(date), "%Y%m%d")
            prev_day = get_prev_working_day(cur_day)
            second_day = get_next_working_day(cur_day)
            third_day = get_next_working_day(second_day)
            date_item = val_data[val_data['date']==date]
            for index,item in date_item.iterrows():
                instrument = item['instrument']
                pred_trend = item['pred_trend']
                prev_bar = self.ds.get_continue_data_by_day(instrument, prev_day.strftime("%Y%m%d"))
                if prev_bar.shape[0]==0:
                    print("{} prev_bar has no data in {}".format(instrument,date))
                    continue
                prev_bar = prev_bar.iloc[0]
                first_bar = self.ds.get_continue_data_by_day(instrument, cur_day.strftime("%Y%m%d"))
                if first_bar.shape[0]==0:
                    print("{} first_bar has no data in {}".format(instrument,date))
                    continue
                first_bar = first_bar.iloc[0]
                second_bar = self.ds.get_continue_data_by_day(instrument, second_day.strftime("%Y%m%d"))
                third_bar = self.ds.get_continue_data_by_day(instrument, third_day.strftime("%Y%m%d"))
                if third_bar.shape[0]==0:
                    print("{} second_bar has no data in {}".format(instrument,date))
                    continue
                third_bar = third_bar.iloc[0]
                item = [date,instrument,pred_trend,prev_bar['settle'],
                        first_bar['open'],first_bar['settle'],first_bar['high'],first_bar['low'],
                        third_bar['open'],third_bar['settle'],third_bar['high'],third_bar['low']]
                results.append(item)
        results = pd.DataFrame(np.array(results),columns=PRED_RESULT_COLUMNS) 
        results['date'] = results['date'].astype(int).astype(int)
        results['pred_trend'] = results['pred_trend'].astype(int)
        results['side_flag'] = np.where(results['pred_trend']==1,1,-1)
        results['diff'] = (results['thirdday_open'].astype(float) - results['firstday_open'].astype(float))/results['firstday_open'].astype(float)
        results['diff'] = results['diff'] * results['side_flag'] 
        if date_range is not None:
            results = results[(results['date']>=date_range[0])&(results['date']<=date_range[1])]
        
        return results
        
    def combine_pred_result(self,pred_dir=None):
        """预测数据中整合实际结果"""

        pred_result_file = os.path.join(pred_dir,"pred_result.pkl")
        with open(pred_result_file, "rb") as fin:
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
                pred_trend = item['top_flag']
                prev_bars = self.ds.get_continue_data_by_day(instrument, prev_day.strftime("%Y%m%d"))
                if prev_bars.shape[0]==0:
                    continue
                prev_bar = prev_bars.iloc[0]
                first_bar = self.ds.get_continue_data_by_day(instrument, cur_day.strftime("%Y%m%d")).iloc[0]
                second_bar = self.ds.get_continue_data_by_day(instrument, second_day.strftime("%Y%m%d")).iloc[0]
                item = [date,instrument,pred_trend,prev_bar['settle'],
                        first_bar['open'],first_bar['settle'],first_bar['high'],first_bar['low'],
                        second_bar['open'],second_bar['settle'],second_bar['high'],second_bar['low']]
                results.append(item)
        results = pd.DataFrame(np.array(results),columns=PRED_RESULT_COLUMNS) 
        results['date'] = results['date'].astype(int).astype(int)
        results['pred_trend'] = results['pred_trend'].astype(int)
        results['side_flag'] = np.where(results['pred_trend']==1,1,-1)
        results['diff'] = (results['secondday_close'].astype(float) - results['prev_close'].astype(float))/results['prev_close'].astype(float)
        results['diff'] = results['diff'] * results['side_flag']         
        combine_data_file = os.path.join(self.work_dir,self.combine_pred_filepath)
        results.to_csv(combine_data_file,index=False)         

    def match_val_and_pred(self,val_result_file,pred_result_file,date_range=[20241201,20241231]):
        """匹配验证结果和预测结果是否一致"""

        step2_file = os.path.join(self.work_dir,val_result_file)
        with open(step2_file, "rb") as fin:
            instrument_result_data = pickle.load(fin)  

        with open(pred_result_file, "rb") as fin:
            pred_result_data = pickle.load(fin)  
                                
        date_list = instrument_result_data[(instrument_result_data['date']>=date_range[0])
                                &(instrument_result_data['date']<=date_range[1])].sort_values().unique()
        match_results = {}
        for date in date_list:
            cur_day = datetime.strptime(str(date), "%Y%m%d")
            val_item = instrument_result_data[instrument_result_data['date']==date]
            pred_item = pred_result_data[pred_result_data['date']==date]
            mat_res = np.intersect1d(val_item['instrument'].values,pred_item['instrument'].values)
            match_results[date] = mat_res
        
        match_results
        
    def combine_backtest_result(self):
        """回测数据中整合实际结果"""
        
        trade_dir = self.backtest_dir
        output_file = self.combine_backtest_filepath
        self.combine_trade_result(trade_dir,output_file=output_file)
        
    def combine_trade_result(self,trade_dir,output_file=None):
        """交易数据中整合实际结果"""
        
        file_path = os.path.join(trade_dir,"trade_data.csv")
        lock_file_path = os.path.join(trade_dir,"lock.csv")
        val_path = os.path.join(self.work_dir,self.combine_val_filepath)
        val_result_data = pd.read_csv(val_path) 
        val_result_data = val_result_data.sort_values(by=["date","instrument"])
        trade_data_df = pd.read_csv(file_path,parse_dates=['trade_date'],infer_datetime_format=True)   
        trade_data_df = trade_data_df[trade_data_df["status"]==ORDER_STATUS.FILLED]
        trade_data_df = trade_data_df.sort_values(by=["trade_date","order_book_id"])
        lock_data = pd.read_csv(lock_file_path) 
        
        result_data = pd.DataFrame(columns=BACKTEST_RESULT_COLUMNS)
        # 遍历预测数据，并分别与实际回测数据结果匹配
        for index,row in val_result_data.iterrows():
            item = {}
            date = row['date']
            instrument = row['instrument']
            # 匹配对应的当日主力合约的开仓交易数据
            trade_date_row = trade_data_df[(trade_data_df['position_effect']=='OPEN')
                                      &(trade_data_df['trade_date'].dt.strftime('%Y%m%d')==str(date))]
            trade_row = trade_date_row[(trade_date_row['order_book_id'].str[:-4]==instrument)]
            item['date'] = date
            item['instrument'] = instrument
            item['trade_flag'] = 'trade'
            item['open_price'] = 0
            item['close_price'] = 0
            item['quantity'] = 0
            item['side'] = 'unknown'
            # 如果没有交易则填充空数值
            if trade_row.shape[0]==0:
                # 查看是否锁定，填写标志
                lock_item = self.get_lock_item(lock_data, date, instrument)
                if lock_item.shape[0]==0:
                    item['trade_flag'] = 'ignore'
                else:
                    item['order_book_id'] = lock_item['order_book_id'].values[0]
                    item['trade_flag'] = 'lock'
                item_df = pd.DataFrame.from_dict(item,orient='index').T
                result_data = pd.concat([result_data,item_df])
                continue
            # 向后查找到对应的平仓交易并记录
            order_book_id = trade_row['order_book_id'].values[0]
            item['order_book_id'] = order_book_id
            item['open_price'] = trade_row['price'].values[0]
            item['side'] = trade_row['side'].values[0]
            item['quantity'] = trade_row['quantity'].values[0]
            match_rows = trade_data_df[(trade_data_df['order_book_id']==order_book_id)
                                      &(trade_data_df['position_effect']=='CLOSE')
                                      &(pd.to_numeric(trade_data_df['trade_date'].dt.strftime('%Y%m%d'))>date)]
            # 如果没有交易则设置标志
            if match_rows.shape[0]==0:
                item['trade_flag'] = 'noclose'
                item_df = pd.DataFrame.from_dict(item,orient='index').T
                result_data = pd.concat([result_data,item_df])
                continue
            # 通过排序取得最近的配对交易
            match_row = match_rows.sort_values(by=["trade_date"],ascending=True).iloc[0]
            item['close_price'] = match_row['price']
            item_df = pd.DataFrame.from_dict(item,orient='index').T
            result_data = pd.concat([result_data,item_df])
        
        combine_data_file = os.path.join(self.work_dir,output_file)
        result_data.to_csv(combine_data_file,index=False)      
    
    def analysis_backtest(self,date_range=None):
        """回测分析"""
        
        backtest_combine_file = os.path.join(self.work_dir,self.combine_backtest_filepath)
        backtest_combine_data = pd.read_csv(backtest_combine_file) 
        val_path = os.path.join(self.work_dir,self.combine_val_filepath)
        val_result_data = pd.read_csv(val_path) 
        # 先合并匹配回测数据和验证数据
        combine_result_whole = pd.merge(backtest_combine_data,val_result_data,on=["date","instrument"],how="inner",validate="one_to_many")    
        combine_result_whole['side_flag'] = np.where(combine_result_whole['side']==SIDE.BUY,1,-1)
        # 关注具备交易的数据
        combine_result = combine_result_whole[combine_result_whole['trade_flag']=='trade']
        combine_result_with_lock = combine_result_whole[combine_result_whole['trade_flag'].isin(['trade','lock'])]
        # 计算回测的涨跌幅度(毛利不计算手续费)
        combine_result['gross_profit'] = (combine_result['close_price']-combine_result['open_price'])/combine_result['open_price']*combine_result['side_flag']
        gross_profit = combine_result['gross_profit'].sum()
        backtest_win_rate = np.sum(combine_result['gross_profit']>0)/combine_result.shape[0]
        # 计算对应的验证涨跌幅度
        combine_result['val_diff'] = (combine_result['thirdday_open']-combine_result['firstday_open'])/combine_result['firstday_open']*combine_result['side_flag']
        val_diff = combine_result['val_diff'].sum()
        total_date = val_result_data['date'].unique().shape[0]
        anno_yield = val_diff/8 * 240/total_date
        gross_anno_yield = gross_profit/8 * 240/total_date
        val_win_rate = np.sum(combine_result['val_diff']>0)/combine_result.shape[0]
        # combine_result_with_lock['val_diff'] = (combine_result_with_lock['thirdday_open']-combine_result_with_lock['prev_close'])/combine_result_with_lock['prev_close']*combine_result_with_lock['side_flag']
        # val_diff_with_lock = combine_result_with_lock['val_diff'].sum()        
        print("val_diff:{},val_anno_yield:{},val_win_rate:{},gross_profit:{},gross_anno_yield:{},backtest_win_rate:{}".format(val_diff,anno_yield,val_win_rate,gross_profit,gross_anno_yield,backtest_win_rate))
        exp_filepath = os.path.join(self.work_dir,self.backtest_analysys_filepath)
        combine_result = combine_result[["date","instrument","order_book_id","open_price",
                       "close_price","side_flag","firstday_open","thirdday_open","val_diff","gross_profit"]]
        if date_range is not None:
            combine_result = combine_result[(combine_result['date']>=date_range[0])&(combine_result['date']<=date_range[1])]
        combine_result.to_csv(exp_filepath,index=False)  
          
          
    def analysis_val_data(self,date_range=None):
        
        val_coll_path = os.path.join(self.work_dir," .csv")
        val_result_data = pd.read_csv(val_coll_path)        
        date_num = val_result_data['date'].unique().shape[0]
        diff_total = val_result_data['diff'].sum()
        begin = datetime.strptime("20250101", "%Y%m%d")
        end = datetime.strptime("20251231", "%Y%m%d")
        workdays_num = len(get_workdays(begin,end))
        top_number = 5
        print("汇总收益率:{},交易天数:{},平均收益:{},年化收益率:{}".format(diff_total,date_num,diff_total/top_number,diff_total/top_number/date_num*workdays_num))
                
    def check_step_output(self):
        
        step1_file = os.path.join(self.work_dir,"step1_rs.pkl")
        step2_file = os.path.join(self.work_dir,"step2_rs.pkl")
        with open(step1_file, "rb") as fin:
            indus_result_data = pickle.load(fin)          
        with open(step2_file, "rb") as fin:
            instrument_result_data = pickle.load(fin)  
        print(indus_result_data)    

    def check_pred_output(self):
        
        result_file_path = "/home/qdata/workflow/fur_backtest_flow/task/159/dump_data/pred_result.pkl"
        with open(result_file_path, "rb") as fin:
            result_data = pickle.load(fin)    
        print(result_data)    
               
    def mock_pred_data(self):      
        
        result_file_path = "/home/qdata/workflow/fur_sim_flow_2025/task/162/dump_data/pred_result.pkl"
        with open(result_file_path, "rb") as fin:
            result_data = pickle.load(fin)    
        result_data['date'] = 20250902
        # result_data.loc[result_data['instrument']=='RS','instrument'] = 'PB'
        # result_data.loc[result_data['instrument']=='CJ','instrument'] = 'SP'
        # result_data.loc[result_data['instrument']=='PK','instrument'] = 'JD'
        with open(result_file_path, "wb") as fout:
            pickle.dump(result_data, fout)          
        print(result_data)


    #######################  辅助方法  #############################

    def get_lock_item(self,lock_data,date,instrument):
        
        return lock_data[(lock_data['date']==date)&(lock_data['instrument']==instrument)]        

     
if __name__ == "__main__":
    stats = DataStats(work_dir=RESULT_FILE_PATH,backtest_dir="/home/qdata/workflow/fur_backtest_flow/trader_data/05")  
    # stats.analysis_val_data()
    # stats.check_step_output()
    # stats.check_pred_output()
    # stats.combine_pred_result(pred_dir="/home/qdata/workflow/fur_backtest_flow/task/159/dump_data/")
    # stats.match_val_and_pred(val_result_file=RESULT_FILE_STEP2,pred_result_file="/home/qdata/workflow/fur_backtest_flow/trader_data/03/pred_result.pkl")
    stats.combine_val_result(date_range=[20250301,20250531])
    # stats.mock_pred_data()
    # stats.combine_backtest_result()
    # stats.analysis_backtest(date_range=[20250301,20250531])
    
    
    