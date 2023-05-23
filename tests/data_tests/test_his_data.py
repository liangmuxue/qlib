import tushare as ts
from data_extract.rqalpha.tdx_ds import TdxDataSource
from rqalpha.model.instrument import Instrument
import datetime

def test_efinance():
    import efinance as ef
    stock_code = '600519'
    frequency = 5
    df = ef.stock.get_quote_history(stock_code, klt=frequency)
    print("df shape:{}".format(df.shape))
    
def test_tushare_day():
    ts.set_token("0202c076e8f73e020a2e259c334897db2a546edd30fd4b29973202dc")
    pro = ts.pro_api()
    df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
    print("df shape:{}".format(df.shape))


def test_pytdx():
    from pytdx.hq import TdxHq_API
    api = TdxHq_API()
    with api.connect('119.147.212.81', 7709):
        # data = api.get_security_bars(0,5, '000001', 14448-480, 480)
        data = api.get_security_quotes([(0, '000001'), (1, '600300')])
        print("data size:{}".format(len(data)))

def test_pytdx_code():
    from pytdx.hq import TdxHq_API
    api = TdxHq_API()
    with api.connect('119.147.212.81', 7709):
        data = api.get_security_list(1, 0)
        cnt = api.get_security_count(0)
        print("data size:{}".format(len(data)))

def test_tdx_ds():
    tds = TdxDataSource("/home/liang/.rqalpha/bundle","/home/qdata/stock_data",frequency_sim=False)
    order_book_id = "000702.XSHE"
    instrument = Instrument({"order_book_id":order_book_id,"symbol":"000702","trading_code":"000702"})
    dt = datetime.datetime(2022,6,14,9,45,0)
    snapshot = tds.current_snapshot(instrument, "1m",dt)
    print(snapshot.last)

def batch_rename():
    import os.path
    rootdir = "/home/qdata/stock_data/ak/item/day/institution"
    i=0;
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            newName=filename.split("_")[0] + ".csv"
            os.rename(os.path.join(parent, filename), os.path.join(parent, newName))
            i=i+1    
    
if __name__ == "__main__":
    # test_tushare_day()
    # test_efinance()
    # test_pytdx()
    batch_rename()
    # test_tdx_ds()
    # test_pytdx_code()
    
    