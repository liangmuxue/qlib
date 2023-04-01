import tushare as ts

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
        data = api.get_security_bars(0,5, '000001', 14448-480, 480)
        print("data size:{}".format(len(data)))

def test_pytdx_code():
    from pytdx.hq import TdxHq_API
    api = TdxHq_API()
    with api.connect('119.147.212.81', 7709):
        data = api.get_security_list(1, 0)
        cnt = api.get_security_count(0)
        print("data size:{}".format(len(data)))
           
if __name__ == "__main__":
    # test_tushare_day()
    # test_efinance()
    test_pytdx()
    # test_pytdx_code()