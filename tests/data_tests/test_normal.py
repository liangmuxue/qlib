from trader.utils.date_util import tradedays
from datetime import datetime
import pickle

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
     
if __name__ == "__main__":
    # test_days_dur()

    # test_import()
    # test_date()
    debug_data()
    