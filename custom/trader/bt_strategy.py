import backtrader as bt
from dill.tests.test_recursive import Model
import pandas as pd
from qlib.strategy.base import BaseStrategy
from darts_pro.data_extension.series_data_utils import get_pred_center_value

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
        
        # 策略参数配置
        self.topk = topk
                
    def log(self, txt, dt=None):
        """记录功能 """
        
        cur_date = self.datas[0].datetime.date(0)
        print("[{}] ".format(cur_date) + txt)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单提交和成交当前不做处理
            return

        # 检查订单是否成交
        # 注意，如果现金不够的话，订单会被拒接
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, %.2f" % order.executed.price)
            elif order.issell():
                self.log("SELL EXECUTED, %.2f" % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # 记录没有挂起的订单
        self.order = None

    def next(self):
        self.log("next in")
        # 检查是否有挂起的订单，如果有的话，不能再发起一个订单
        if self.order:
            return

        pred_series_list = self.predict_process()
        if pred_series_list is None:
            # self.log("pred series none")
            return
        
        # 根据预测结果，取得排序数据
        df_result = self.order_pred_result(pred_series_list)
        topk = 3
        df_topk = df_result[:topk]
        # 取得所有持仓的股票名
        hold_bond_name = [_p._name for _p in self.broker.positions if self.broker.getposition(_p).size > 0]
        # 买入逻辑执行
        self.exec_buy_logic(df_topk, hold_bond_name)
        # 卖出逻辑执行
        self.exec_sell_logic(df_topk, hold_bond_name)
            
    def exec_buy_logic(self,df_topk,hold_bond_name):
        """购买的逻辑"""
        
        self.log("exec_buy_logic in,top price:{}".format(df_topk.iloc[0]["price"]))
        pos_size = len(hold_bond_name)
        for index, row in df_topk.iterrows():
            # 累计涨幅超过阈值，并且不在持仓列表中，则购买
            if row["price"] > 5 and row["instrument"] not in hold_bond_name:
                pos_size = pos_size + 1
                # 如果持仓数超过规定值，则退出
                if pos_size > self.topk:
                    break   
                self.buy()              
        

    def exec_sell_logic(self,df_result,hold_bond_name):
        """卖出的逻辑"""
        
        pos_size = len(hold_bond_name)
        # for index, instrument in enumerate(hold_bond_name):
        #     item = df_result[df_result["instrument"]==instrument]["price"]
        #     # 累计涨幅超过阈值，并且不在持仓列表中，则购买
        #     if row["price"] > 0.5 and row["instrument"] not in hold_bond_name:
        #         pos_size = pos_size + 1
        #         # 如果持仓数超过规定值，则退出
        #         if pos_size > self.topk:
        #             break   
        #         self.buy()      
                        
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
            # 最后一个标签值减去对一个标签值，得到涨跌幅度
            price = pred_center_data[-1] - pred_center_data[0]
            df_result.loc[index] = [group_col_val,price]
        df_result.sort_values("price",ascending=False,inplace=True)   
        return df_result
            
class QlibStrategy(BaseStrategy):
    
    def __init__(self,model,dataset,topk):

        self.model = model
        self.dataset = dataset
        # 跟踪订单
        self.order = None
        
        # 策略参数配置
        self.topk = topk    
    
    