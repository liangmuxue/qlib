import backtrader as bt
from dill.tests.test_recursive import Model
import pickle
import pandas as pd
from qlib.strategy.base import BaseStrategy
from darts_pro.data_extension.series_data_utils import get_pred_center_value
from darts import TimeSeries, concatenate
from qlib import data
from gunicorn import instrument
from cus_utils.tensor_viz import TensorViz
from datetime import datetime

class Strategy(bt.Strategy):
    """自定义策略基础类,继承backtrader"""
    
    def __init__(self,model,dataset,pred_df,topk=3):
        """
        Parmas:
            model 训练好的模型
            dataset 训练阶段产生的数据集对象，用于predict
        """
        
        self.model = model
        self.dataset = dataset
        self.pred_df = pred_df

        # 跟踪订单
        self.order = None
        self.pred_data_path = self.model.pred_data_path
        # 策略参数配置
        self.topk = topk
        
        # 保留数据全集，用于后续数据处理
        if self.model.load_dataset_file:
            # 使用之前保存的数据作为当前全集参考数据
            self.df_ref =  self.model.df_ref
        else:
            # 使用上一步骤中dataset对象保存的数据集，作为当前全集参考数据
            self.df_ref = dataset.df_all
        self.df_ref = self.df_ref.reset_index()
        self.has_save_data = False
        # 交易记录
        self.trade_list = []
        self.viz_input = TensorViz(env="data_backtest")
        
    def _get_pickle_path(self,cur_date):
        if self.pred_data_path is None or len(self.pred_data_path)==0:
            return None
        data_path = self.pred_data_path + "/" + str(cur_date) + ".pkl"
        return data_path
    
    def get_data_by_name(self,name):
        for data in self.datas:
            if data._name == name:
                return data
        return None
        
    def log(self, txt, dt=None):
        """记录功能 """
        
        cur_date = self.data.datetime.date(0)
        name = self.data._name
        print("[{} stock_{}] ".format(cur_date,name) + txt)
    
    def notify_order(self, order):
        self.log("notify_order in:{}".format(order.status))
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交和被接收，当前不做处理
            return

        # 检查订单是否成交,注意，如果现金不够的话，订单会被拒接
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED,code:{},price:{}".format(order.data._name,order.executed.price))
                self.update_trade_info_by_order(order)
                self.print_trade_info()
            elif order.issell():
                self.log("SELL EXECUTED,code:{},price:{}".format(order.data._name,order.executed.price))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # 记录没有挂起的订单
        self.order = None

    def print_trade_info(self):
        print('current cash', self.broker.getcash())
        print('total assets', self.broker.getvalue())
        print('current position', self.broker.getposition(self.data).size)
        print('current cost', self.broker.getposition(self.data).price)
    
    def next(self):
        self.log("next in")
        cur_date = self.datas[0].datetime.date(0)
        # 检查是否有挂起的订单，如果有的话，不能再发起一个订单
        if self.order:
            return
        pred_series_list = self.predict_process(cur_date)
        # 只使用预测数据得分高的股票
        pred_series_list = self.dataset.filter_pred_data_by_mape(pred_series_list,result_id=3)
        if self.pred_data_path is not None and len(self.pred_data_path)>0 and not self.model.load_dataset_file:
            if not self.has_save_data:
                # 保存全集，只进行一次
                df_data_path = self.pred_data_path + "/df_all.pkl"
                with open(df_data_path, "wb") as fout:
                    pickle.dump(self.df_ref, fout)     
                self.has_save_data = True             
            data_path = self._get_pickle_path(cur_date)
            with open(data_path, "wb") as fout:
                pickle.dump(pred_series_list, fout)                
        if pred_series_list is None:
            # self.log("pred series none")
            return
        self.process_by_pred_data(pred_series_list)
        
    def process_by_pred_data(self,pred_series_list):
        
        # 根据预测结果，取得排序数据
        df_result = self.order_pred_result(pred_series_list)
        # 存储预测列表
        self.pred_series_list = pred_series_list
        topk = 30
        df_topk = df_result[:topk]
        # 取得所有持仓的股票名
        hold_bond_name = [_p._name for _p in self.broker.positions if self.broker.getposition(_p).size > 0]
        # 买入逻辑执行
        self.exec_buy_logic(df_topk, hold_bond_name,pred_series_list=pred_series_list)
        # 卖出逻辑执行
        self.exec_sell_logic(df_result, hold_bond_name,pred_series_list=pred_series_list)        
            
    def exec_buy_logic(self,df_topk,hold_bond_name,pred_series_list=None,thredhold=8):
        """购买的逻辑"""
        
        self.log("exec_buy_logic in,top price:{}".format(df_topk.iloc[0]["price"]))
        cur_date = self.datas[0].datetime.date(0)
        for index, row in df_topk.iterrows():
            instrument = int(row["instrument"])
            group_code = self.dataset.get_group_code_by_rank(instrument)
            # 如果当天股票不交易，则不处理
            if self.dataset.get_data_by_trade_date(self.df_ref,instrument,cur_date) is None:
                self.log("Has No Trade for:{}".format(instrument))
                continue
            # 取得预测数据与实际数据，进行统一筛选
            pred_and_real = self.get_single_pred_data(pred_series_list,instrument,cur_date)
            # 如果之前的平均日涨幅超过阈值，则不执行买入
            price_range = self.real_price_range(pred_and_real)
            if price_range > 3:
                self.log("Ignore cause price range:{},{}".format(group_code,price_range))
                continue
            # 累计涨幅超过阈值，并且不在持仓列表中，则购买
            if row["price"] > thredhold and instrument not in hold_bond_name:
                title = "buydata:{}_{},{}".format(group_code,cur_date,row["price"])
                self.view_pred_and_val(self.pred_series_list,instrument,title=title)  
                # 如果持仓数超过规定值，则退出
                if self.get_trade_len() > self.topk:
                    break   
                # 取得对应股票代码的data数据
                data = self.get_data_by_name(str(instrument))
                size = 1
                trade_info = {"date":self.datas[0].datetime.date(0),"instrument":instrument,"size":size}
                # 同一股票当天不能购买多次
                if self.get_exist_trade_info(trade_info) is None:
                    self.log("buy data:{}".format(data._name))
                    self.buy(data=data,size=size)     
                    self.add_trade_info(trade_info)   

    def exec_sell_logic(self,df_result,hold_bond_name,pred_series_list=None,thredhold=10):
        """卖出的逻辑"""
        
        self.log("exec_sell_logic in,hold_bond_name:{}".format(hold_bond_name))
        size = 1
        for index, instrument in enumerate(hold_bond_name):
            group_code = self.dataset.get_group_code_by_rank(int(instrument))
            data = self.get_data_by_name(instrument)
            close = data.close[0]
            # 
            pos = self.broker.getposition(data)
            # 计算涨跌幅度
            amplitude = (close - pos.price)/ pos.price * 100
            # 累计涨幅或跌幅超过阈值，则卖出止盈或止损
            if amplitude > thredhold or amplitude < -3: 
                self.sell(data=data,size=size)  
                continue
            # 取得对应股票的预测数据
            result_obj = self.get_result_data(df_result,instrument)
            # 如果此股票后续的预测形式不好（超出卖出阈值），则卖出
            if result_obj["price"]<-3:
                self.sell(data=data,size=size)  
    
    def real_price_range(self,pred_and_real):
        """取得实际平均涨幅"""
        
        data_columns = self.dataset.pred_data_columns()
        pred_len = self.dataset.pred_len
        r = (pred_and_real["price_{}".format(pred_len-1)] - pred_and_real["price_0"]) / pred_and_real["price_0"] / pred_len * 100
        return r.values[0]
            
    def get_result_data(self,df_result,instrument):     
        """根据股票代码，取得计算结果集中对应的预测数据"""      
        
        for index, row in df_result.iterrows():
            if instrument==str(int(row["instrument"])):
                return row
        return None

    def get_trade_len(self):
        """查询未成交交易品种数量"""
        
        cnt = 0
        for item in self.trade_list:
            # 只检查未成交的数量
            if item["status"]==1:
                cnt+=1
                
        # 加上已成交的持仓数，形成全部的交易品种数量
        total_cnt = cnt + self.broker.getposition(self.data).size
        return total_cnt
 
    def update_trade_info_by_order(self,order):
        """根据订单信息，修改交易信息状态"""
        
        cnt = 0
        for item in self.trade_list:
            if item["instrument"]==order.data._name:
                # 修改状态为已成交
                item["status"] = 2 
                
               
    def get_exist_trade_info(self,trade_info):
        """查询是否已有相关交易"""
        
        for item in self.trade_list:
            # 根据日期和股票代码查询，还要满足状态为未成交
            if item["instrument"]==trade_info["instrument"] and item["date"]==trade_info["date"] and item["status"]==1:
                return item
        return None
        
    def add_trade_info(self,trade_info):
        """添加交易信息，可以是成交的以及未成交的"""
        
        trade_info["status"] = 1
        self.trade_list.append(trade_info)
                       
    def predict_process(self,cur_date,outer_df=None):
        """执行预测过程"""
        
        # 根据时间点，取得对应的输入时间序列范围
        total_range,val_range = self.dataset.get_part_time_range(cur_date,ref_df=self.df_ref)
        # 如果不满足预测要求，则返回空
        if total_range is None:
            self.log("pred series none")
            return None
        
        # 从执行器模型中取得已经生成好的模型变量
        my_model = self.model.model
        # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
        series_transformed,val_series_transformed,past_convariates,future_convariates = self.dataset.build_series_data_step_range(total_range,val_range,fill_future=True,outer_df=outer_df)
        my_model.fit(series_transformed,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,verbose=True,epochs=-1)            
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=self.dataset.pred_len, series=val_series_transformed,num_samples=200,
                                            past_covariates=past_convariates,future_covariates=future_convariates)  
        # 归一化反置，恢复到原值
        pred_series_list = self.dataset.reverse_transform_preds(pred_series_list)
        return pred_series_list 
    
    def order_pred_result(self,pred_series_list):
        """对预测结果，按照相关规则进行排序"""
        
        target_column = self.dataset.get_target_column()
        group_column = self.dataset.get_group_rank_column()
        df_result = pd.DataFrame(columns=["instrument", "price"])
        for index,series in enumerate(pred_series_list):
            pred_center_data = get_pred_center_value(series).data
            group_col_val = series.static_covariates[group_column].values[0]
            # 取得区间内涨跌幅度（最后一天的数值减去第一天的数值，并除以第一天数值）
            price_range = (pred_center_data[-1] - pred_center_data[0]) / pred_center_data[0] * 100
            df_result.loc[index] = [group_col_val,price_range]
        # 针对指标进行排序
        df_result.sort_values(by=["price"],ascending=False,inplace=True)   
        return df_result
    
    def get_single_pred_data(self,pred_series_list,group_rank,date):
        """取得单只股票预测及原始数据"""
        
        group_column = self.dataset.get_group_rank_column()
        time_column = self.dataset.get_time_column()
        df_ref = self.df_ref
        for index,series in enumerate(pred_series_list):
            group_rank_in = series.static_covariates[group_column].values[0]
            # 取得给定股票的数据
            if group_rank==group_rank_in:
                # 取得预测数据
                pred_center_data = get_pred_center_value(series).data
                # 拼接预测数据到每个股票
                group_item = self.dataset.get_group_code_by_rank(group_rank)
                time_index_df = df_ref[(df_ref[group_column]==group_rank)&(df_ref["datetime"]>=str(date).replace("-",""))]
                if time_index_df.shape[0]==0:
                    return None
                
                pred_len = self.dataset.pred_len
                data_columns = self.dataset.pred_data_columns()
                time_index = time_index_df.iloc[0][time_column]                 
                # 预测数据
                data_line = [float(group_item),float(str(date).replace("-",""))] + pred_center_data.tolist()
                # 实际数据部分
                label_data = df_ref[(df_ref[group_column]==group_rank)&
                                    (df_ref[time_column]>=time_index)&
                                    (df_ref[time_column]<time_index+5)]["label"].values.tolist()
                # 拼接到一起
                data_line = data_line + label_data 
                # 实际价格部分，往前取相同范围的数据
                pre_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[time_column]>=time_index-5)&(df_ref[time_column]<time_index)]["label_ori"].values.tolist()
                next_price_data = df_ref[(df_ref[group_column]==group_rank)&(df_ref[time_column]>=time_index)&(df_ref[time_column]<time_index+5)]["label_ori"].values.tolist()
                # 实际数据部分，有可能长度不够，补0
                if len(next_price_data)<pred_len:
                    next_price_data = next_price_data + [0.0 for i in range(pred_len-len(next_price_data))]        
                data_line = data_line + pre_price_data + next_price_data       
                target_df = pd.DataFrame([data_line],columns=data_columns)
                return target_df
        return None
        
    def view_pred_and_val(self,pred_list,group_rank_code,title=""):  
        # 根据交易日期和股票，查找出实际数值,后续使用移动平均数据以及实际数值,同时包含预测数据
        df,_ = self.dataset.get_real_data(self.df_ref,pred_list,group_rank_code,extend_begin=10)
        self.view_instrument_data(df, title=title)  
        
    def view_instrument_data(self,df,title=""):
        # 按照预测时间序列，分别回执移动平均价格以及实际价格曲线
        view_data = df[["label_ori","label","pred"]].values
        names = ["price","mean_prices","pred_values"]
        x_range = df["time_idx"].values
        self.viz_input.viz_matrix_var(view_data,win=title,title=title,names=names,x_range=x_range)  
        pass
        
class ResultStrategy(Strategy):
    """使用预存储的数据进行快速回测"""
    
    
    def __init__(self,model,dataset,topk=3,pred_data_path=None):
        """
        Parmas:
            model 训练好的模型
            dataset 训练阶段产生的数据集对象，用于predict
        """
        super().__init__(model,dataset,topk=topk)
        
    def next(self):
        self.log("ResultStrategy next in")
        # self.print_trade_info()
        # 检查是否有挂起的订单，如果有的话，不能再发起一个订单
        if self.order:
            return
        
        cur_date = self.datas[0].datetime.date(0)
        # 直接读取预存的结果文件
        cur_date_str = str(cur_date).replace("-","")
        pred_data_path = self._get_pickle_path(cur_date_str)
        with open(pred_data_path, "rb") as fin:
            pred_series_list = pickle.load(fin)
        if pred_series_list is None:
            self.log("pred series none")
            return
        
        # 只使用预测数据得分高的股票
        pred_series_list = self.dataset.filter_pred_data_by_corr(pred_series_list,result_id=7)
        # self.total_view(pred_series_list)
        self.process_by_pred_data(pred_series_list)
    
    def pred_save(self):
        cur_date = self.datas[0].datetime.date(0)
        pred_series_list = self.predict_process(cur_date,outer_df=self.df_ref)
        data_path = self._get_pickle_path(cur_date)
        with open(data_path, "wb") as fout:
            pickle.dump(pred_series_list, fout)    
                           
    def total_view(self,pred_series_list):
        """对所有跟踪的预测及实际数据进行可视化"""
        
        group_column = self.dataset.get_group_rank_column()
        cur_date = self.datas[0].datetime.date(0)
        for series in pred_series_list:
            group_rank_code = int(series.static_covariates[group_column].values[0])
            group_code = self.dataset.get_group_code_by_rank(group_rank_code)
            title = "pred and result:{}_{}".format(group_code,cur_date)
            self.view_pred_and_val(pred_series_list,group_rank_code,title=title)    
                   
class QlibStrategy(BaseStrategy):
    
    def __init__(self,model,dataset,topk):

        self.model = model
        self.dataset = dataset
        self.order = None
        self.topk = topk   
    
    