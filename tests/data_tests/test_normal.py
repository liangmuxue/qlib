from trader.utils.date_util import tradedays
from datetime import datetime
import math
import pickle
import numpy as np
import time

def test_days_dur():
    trade_date = 20230508
    before_date = 20230510
    trade_date = datetime.strptime(str(trade_date),"%Y%m%d")
    before_date = datetime.strptime(str(before_date),"%Y%m%d")
    dur_days = tradedays(trade_date,before_date)
    print(dur_days)

 
def test_crf():
    import torch
    from TorchCRF import CRF
    from torch.utils.data import TensorDataset,DataLoader
    from torch.optim import Adam
    import matplotlib.pyplot as plt
    batch_size = 10
    seq_len = 19
    num_tags = 9
    X = torch.randn(1000,seq_len,num_tags)
    tags = torch.ones([1000,seq_len]).long()
    tensor_data = TensorDataset(X,tags)
    dataloader = DataLoader(tensor_data,shuffle = True,batch_size = batch_size)
    model = CRF(num_tags)
    optimizer = Adam(model.parameters(),lr = 0.05,betas = (0.9,0.99))
    losses = []
    for seq,seq_tag in dataloader:
        loss = (-1)*model(seq,seq_tag)
        loss.backward()
        optimizer.step()
        losses.append(loss.tolist())
    plt.xlabel('number of iter')
    plt.ylabel('loss')
    plt.title('a simple of torchcrf')
    plt.plot(losses)
    plt.legend()
    plt.show()
    print('value',model.decode(X[:4]))

def test_date():
    start_date = 20220801
    start_date = datetime.strptime(str(start_date),'%Y%m%d').date()    
    print(start_date)
    
def test_import():
    from darts_pro.data_extension.custom_nor_model import TFTBatchModel


def debug_data():    
    
    data_path = "custom/data/asis/debug/rate_total.pkl"
    with open(data_path, "rb") as fin:
        rate_total = pickle.load(fin)      
    print(rate_total) 

def test_math():

    def slope(line):
        x1, y1, x2, y2 = line
        if x2 != x1:
            return (y2 - y1) / (x2 - x1)
        else:
            return 0
        
    def slope_to_angle_signed(slope):
        angle_radians = math.atan(slope)
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees if slope >= 0 else 180 + angle_degrees
    
    line = [1, 2, 3, 3]
    s = slope(line)
    angle = slope_to_angle_signed(s)
    angle = -angle if s<0 else angle
    sin_val = math.sin(math.radians(angle))  
    print(f"slope is {s} angle is {angle},sin is:{sin_val}")   
    
    x1, y1, x2, y2 = line
    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    
def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('func f:{},time:{}'.format(f.__name__,(e_time - s_time)))
        return res
    return inner

def test_time_compute():
    
    @get_time
    def tt():
        time.sleep(1)
        
    class Cls():
        @get_time
        def inner_func(self):
            time.sleep(1)
    
    # tt()
    cls = Cls()
    cls.inner_func()

def test_c_head():
    import CppHeaderParser
    cppHeader = CppHeaderParser.CppHeader("custom/trader/emulator/openctp/ThostFtdcUserApiDataType.h")
    for define in cppHeader.deines:
        print(define)

def test_dict_merge():
    a = {"a":1}
    b = {"b":1}
    a.update(b)
    print(a)

if __name__ == "__main__":
    # test_days_dur()

    # test_import()
    # test_date()
    # test_math()
    # test_time_compute()
    # test_c_head()
    test_dict_merge()
    # debug_data()
    