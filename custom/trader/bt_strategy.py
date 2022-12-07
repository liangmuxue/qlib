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

class Strategy(bt.Strategy):
    """自定义策略基础类,继承backtrader"""
    
    def __init__(self,model,dataset,topk=3):
        """
        Parmas:
            model 训练好的模型
            dataset 训练阶段产生的数据集对象，用于predict
        """
        
        self.model = model
        self.dataset = dataset

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
        
        self.has_save_data = False
        # 交易记录
        self.trade_list = []
        
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
                self.log("BUY EXECUTED, %.2f" % order.executed.price)
                self.update_trade_info_by_order(order)
                self.print_trade_info()
            elif order.issell():
                self.log("SELL EXECUTED, %.2f" % order.executed.price)

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
        pred_series_list = self.predict_process()
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
        topk = 3
        df_topk = df_result[:topk]
        # 取得所有持仓的股票名
        hold_bond_name = [_p._name for _p in self.broker.positions if self.broker.getposition(_p).size > 0]
        # 买入逻辑执行
        self.exec_buy_logic(df_topk, hold_bond_name)
        # 卖出逻辑执行
        self.exec_sell_logic(df_result, hold_bond_name)        
            
    def exec_buy_logic(self,df_topk,hold_bond_name):
        """购买的逻辑"""
        
        self.log("exec_buy_logic in,top price:{}".format(df_topk.iloc[0]["price"]))
        pos_size = len(hold_bond_name)
        cur_date = self.datas[0].datetime.date(0)
        for index, row in df_topk.iterrows():
            instrument = str(int(row["instrument"]))
            # 如果当天股票不交易，则不处理
            if self.dataset.get_data_by_trade_date(self.df_ref,instrument,cur_date) is None:
                self.log("Has No Trade for:{}".format(instrument))
            # 累计涨幅超过阈值，并且不在持仓列表中，则购买
            if row["price"] > 3 and instrument not in hold_bond_name:
                self.view_pred_buy_data(row,cur_date)
                # 如果持仓数超过规定值，则退出
                if self.get_trade_len() > self.topk:
                    break   
                # 取得对应股票代码的data数据
                data = self.get_data_by_name(instrument)
                size = 1
                trade_info = {"date":self.datas[0].datetime.date(0),"instrument":instrument,"size":size}
                # 同一股票当天不能购买多次
                if self.get_exist_trade_info(trade_info) is None:
                    self.buy(data=data,size=size)     
                    self.add_trade_info(trade_info)   

    def exec_sell_logic(self,df_result,hold_bond_name):
        """卖出的逻辑"""
        
        size = 1
        for index, instrument in enumerate(hold_bond_name):
            data = self.get_data_by_name(instrument)
            close = data.close[0]
            pos = self.broker.getposition(data)
            # 计算涨跌幅度
            amplitude = (close - pos.price)/ pos.price * 100
            # 累计涨幅或跌幅超过阈值，则卖出止盈或止损
            if amplitude > 5 or amplitude < -5: 
                self.sell(data=data,size=size)  
                continue
            # 取得对应股票的预测数据
            result_obj = self.get_result_data(df_result,instrument)
            # 如果此股票后续的预测形式不好（超出卖出阈值），则卖出
            if result_obj["price"]<-2:
                self.sell(data=data,size=size)  
                
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
                       
    def predict_process(self):
        """执行预测过程"""
        
        cur_date = self.datas[0].datetime.date(0)
        # 根据时间点，取得对应的输入时间序列范围
        total_range,val_range = self.dataset.get_part_time_range(cur_date,ref_df=self.dataset.df_all)
        # 如果不满足预测要求，则返回空
        if total_range is None:
            self.log("pred series none")
            return None
        
        # 从执行器模型中取得已经生成好的模型变量
        my_model = self.model.model
        # 每次都需要重新生成时间序列相关数据对象，包括完整时间序列用于fit，以及测试序列，以及相关变量
        series_transformed,val_series_transformed,past_convariates,future_convariates = self.dataset.build_series_data_step_range(total_range,val_range,fill_future=True)
        my_model.fit(series_transformed,val_series=val_series_transformed, past_covariates=past_convariates, future_covariates=future_convariates,
                     val_past_covariates=past_convariates, val_future_covariates=future_convariates,verbose=True,epochs=-1)            
        # 对验证集进行预测，得到预测结果   
        pred_series_list = my_model.predict(n=self.dataset.pred_len, series=val_series_transformed[:93],num_samples=200,
                                            past_covariates=past_convariates[:93],future_covariates=future_convariates[:93])  
        # 归一化反置，恢复到原值
        pred_series_list = self.dataset.reverse_transform_preds(pred_series_list)
        return pred_series_list 
    
    def order_pred_result(self,pred_series_list):
        """对预测结果，按照涨跌幅进行排序"""
        
        target_column = self.dataset.get_target_column()
        group_column = self.dataset.get_group_rank_column()
        df_result = pd.DataFrame(columns=["instrument", "price"])
        for index,series in enumerate(pred_series_list):
            pred_center_data = get_pred_center_value(series).data
            group_col_val = series.static_covariates[group_column].values[0]
            # 由于最后一个数值反应的是这几天的均值，因此使用最后一个数据
            price = pred_center_data[-1]
            df_result.loc[index] = [group_col_val,price]
        df_result.sort_values("price",ascending=False,inplace=True)   
        return df_result
    
    def view_pred_buy_data(self,result_row,trade_date):
        """对于预测买入的数据，进行可视化"""
        
        instrument = result_row["instrument"]
        # 根据交易日期和股票，查找出实际数值,后续使用移动平均数据以及实际数值,同时包含预测数据
        df,time_begin = self.dataset.get_real_data(self.df_ref,self.pred_series_list,instrument)
        viz_input = TensorViz(env="back_test")
        title = "buy data:{}_{}".format(df["instrument"].values[0],trade_date)
        # 按照预测时间序列，分别回执移动平均价格以及实际价格曲线
        view_data = df[["label_ori","label","pred"]].values
        desc = "begin time index:{},pred value:{}".format(time_begin,result_row["price"])
        names = ["values","mean_values","pred_values"]
        # x_range = df["datetime"].dt.strftime('%Y%m%d').astype(int).values
        x_range = df["time_idx"].values
        viz_input.viz_matrix_var(view_data,win=title,title=title,names=names,desc=desc,x_range=x_range)  
    
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
        pred_data_path = self._get_pickle_path(cur_date)
        with open(pred_data_path, "rb") as fin:
            pred_series_list = pickle.load(fin)
        if pred_series_list is None:
            self.log("pred series none")
            return
        self.process_by_pred_data(pred_series_list)
            
                    
class QlibStrategy(BaseStrategy):
    
    def __init__(self,model,dataset,topk):

        self.model = model
        self.dataset = dataset
        self.order = None
        self.topk = topk   
    
    