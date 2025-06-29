import pandas as pd
from datetime import date
import datetime as dt
import sys, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
    
def test_pd_ser():
    
    dict_data1 = {
        "Beijing":1000,
        "Shanghai":2000,
        "Shenzhen":500
    }
    
    data1 = pd.Series(dict_data1)
    data1.name = "City Data"
    data1.index.name = "City Name"
    print(data1)
 
def test_pd_df():
    index_arrays = [[1, 1, 2, 2], ['男', '女', '男', '女']]
    columns_arrays = [['2020', '2020', '2021', '2021'],
                      ['上半年', '下半年', '上半年', '下半年',]]
    index = pd.MultiIndex.from_arrays(index_arrays,
                                      names=('班级', '性别'))
    print("index:",index)
    columns = pd.MultiIndex.from_arrays(columns_arrays,
                                        names=('年份', '学期'))
    print("columns:",columns)
    df = pd.DataFrame([(88,99,88,99),(77,88,97,98),
                       (67,89,54,78),(34,67,89,54)],
                      columns=columns, index=index)    
    print("df:",df)    
 
def test_pd_trans():
    np.random.seed(2015)
    
    df = pd.DataFrame({'account': ['foo', 'bar', 'baz'] * 3,
                       'val': np.random.choice([np.nan, 1],size=9)})
    print (df)
    
    df['val1'] = df.groupby('account')['val'].transform('last')
    df['val2'] = df.groupby('account')['val'].transform('nth', -1)
    print (df)       
   
def test_pd_index():
    # Creating index for multi-index dataframe
    tuples = [('A', 'a'), ('A', 'b'), ('B', 'a'), ('B', 'b')]
    index = pd.MultiIndex.from_tuples(tuples)
    # Value corresponding to the index
    data = [2, 4, 6, 8]
    # Creating dataframe using 'data' and 'index'
    df = pd.DataFrame(data = data, index = index, columns = ['value'])
    print(df)
    reset_df = df.reset_index()
    print(reset_df)

def test_pd_timeser():
    data = pd.DataFrame({'ID': ['1/1/2022','7/21/2024','1/1/1931']})
    sr = pd.to_datetime(data['ID'], format= '%m/%d/%Y').squeeze().dt.date
    data['General format'] =  sr.apply(lambda x: (x - date(1900, 1, 1)).days +2 ).to_frame()
    print(data)
    df = pd.DataFrame({'date': ['2008-04-24 01:30:00.000']})
    df['date'] = pd.to_datetime(df['date'])
    df['date'] = df['date'].astype('int64')//1e9
    print(df)
    print(pd.to_datetime(df['date'], unit='s'))

def test_join():
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3'],
                         'key': ['K0', 'K1', 'K0', 'K1']})
    
 
    right = pd.DataFrame({'C': ['C0', 'C1'],
                          'D': ['D0', 'D1']},
                          index=['K0', 'K1'])
    
 
    result = left.join(right, on='key')    
    print("left is",left)
    print("right is",right)
    print("result is",result)
    
def test_merge():
    # left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
    #      'key2': ['K0', 'K1', 'K0', 'K1'],
    #      'A': ['A0', 'A1', 'A2', 'A3'],
    #      'B': ['B0', 'B1', 'B2', 'B3']})
    #
    #
    # right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K1'],
    #       'key2': ['K0', 'K0', 'K0', 'K0'],
    #       'C': ['C0', 'C1', 'C2', 'C3'],
    #       'D': ['D0', 'D1', 'D2', 'D3']})
    #
    #
    # result = pd.merge(left, right, on=['key1'],how="left")  
    # print("left is",left)
    # print("right is",right)
    # print("result is",result)
    
    
    np.random.seed(0)
    left = pd.DataFrame({'key': ['A', 'B', 'C', 'D','G'], 'value': np.random.randn(5)})    
    right = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': np.random.randn(4)})
    
    merged=left.merge(right,on='key',how='left',indicator=True)
    print(left)
    print(merged)

def test_columns():
    idxdata = ["col1","col2"]
    name = "cindex"
    col_idx = pd.Index(idxdata,dtype=object)
    data = pd.DataFrame({'key': ['A', 'B', 'C', 'D','G'], 'value': np.random.randn(5)})  
    data.columns = col_idx
    print(data)
  
def test_group():
    np.random.seed(2015)
    df = pd.DataFrame({'account': ['foo', 'bar', 'baz'] * 3,
                       'val': np.random.choice([0, 1],size=9)})
    print (df)
    df['val1'] = df.groupby('account')['val'].transform('last')
    df['val2'] = df.groupby('account')['val'].transform('nth', -1)
    print (df)

def test_group2():
    np.random.seed(2015)
    df = pd.DataFrame({'instrument': [3, 6, 9] * 3,
                       'val': np.random.choice([0, 1],size=9)})
    print (df)
    df_rank = df.instrument.rank(method='dense',ascending=False)
    dg = df.groupby('instrument')
    print (dg)
    
def test_diff():
    df = pd.DataFrame({"idx":list("11122233444446666"),"timeidx":list("13413579134561589")})
    df['idx'] = df['idx'].astype(int)
    df['timeidx'] = df['timeidx'].astype(int)
    dft = np.sort(df['timeidx'].unique())
    d = dict(enumerate(dft.flatten(), 1))
    inv_map = {v: k for k, v in d.items()}
    # df['timeidx'] = df.groupby(['idx','timeidx'], observed=True).ngroup()
    df['timeidx'] = df['timeidx'].map(inv_map)
    print("df is",df)
    g = df.groupby('idx', observed=True)
    df_index_diff_to_next = -g["timeidx"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
    print("diff",df_index_diff_to_next)
    vc = df_index_diff_to_next.value_counts()
    print("vc",vc)
 
def test_tft():
    type = "sku"
    # type = "stock"
    if type=="sku":
        data = pd.read_pickle("/home/qdata/qlib_data/test/data_ori.pkl")
    else:
        data = pd.read_pickle("/home/qdata/qlib_data/test/data.pkl")
    df_diff_to_next = data["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("df_diff_to_next")
    vcd = df_diff_to_next.value_counts()
    # g = data.groupby(['__group_id__instrument'], observed=True)
    if type=="sku":
        g = data.groupby(['agency','sku'], observed=True)
    else:
        g = data.groupby(['instrument'], observed=True)
    df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")
    vc = df_index_diff_to_next.value_counts()
    # print("vc",vc)
    # print(data["__time_idx__"])
    if type=="sku":
        data.sort_values(by=['agency','sku'])[['agency','sku','__time_idx__']].to_csv("/home/qdata/qlib_data/test/time_idx_ori.csv")
        data["__time_idx__"].to_frame().to_csv("/home/qdata/qlib_data/test/time_idx_ori.csv")
    else:
        data.sort_values(by=['instrument','__time_idx__'],inplace=True)
        # data["__time_idx__"].to_frame().to_csv("/home/qdata/qlib_data/test/time_idxi.csv")
        g = data.groupby('instrument', observed=True)
        df_index_diff_to_next = -g["__time_idx__"].diff(-1).fillna(-1).astype(int).to_frame("time_diff_to_next")    
        print(df_index_diff_to_next.value_counts())
    # print(df_index_diff_to_next)

def test_tft_data_view():
    type = "sku"
    type = "stock"
    if type=="sku":
        data = pd.read_pickle("/home/qdata/qlib_data/test/data_sku.pkl")
        g_sku = data.groupby(['agency','sku'], observed=True)
        # print("sku count",g_sku.size().reset_index(name='counts'))    
    else:
        data = pd.read_pickle("/home/qdata/qlib_data/test/data.pkl")
    data.describe()
    # g_month = data.groupby(['month'], observed=True)
    # print("month count",g_month.size().reset_index(name='counts')["counts"])
  
def dict_to_csv(dict_data,csv_filepath,csv_columns):
    import csv
    with open(csv_filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for key,val in dict_data.items():
            writer.writerow(key,val)

def test_vis():
    import matplotlib.pyplot as plt
    df = pd.DataFrame({"idx":list("111222333444555"),"timeidx":list("123123123123123"),"volume":list("134135791361589")})
    df['volume'] = df['volume'].astype(int)
    df['timeidx'] = df['timeidx'].astype(int)
    print (df)
    df.set_index('timeidx', inplace=True)
    df.groupby(['idx'])['volume'].plot(legend=True)
    plt.show()
    print("show after")

def test_tft_vis():
    import matplotlib.pyplot as plt
    data = pd.read_pickle("/home/qdata/qlib_data/test/data_ori.pkl")
    data.set_index('__time_idx__', inplace=True)
    g = data.groupby(['agency','sku'], observed=True)
    g["volume"].plot(legend=True)
    plt.savefig("/home/qdata/test_img/fig.png")
    # plt.show()
    print("show after")

def test_tft_stock_vis():
    import matplotlib.pyplot as plt
    data = pd.read_pickle("/home/qdata/qlib_data/test/data.pkl")
    pick_list = [43,49,20]
    data = data[data['instrument'].isin(pick_list)]
    data.set_index('datetime', inplace=True)
    g = data.groupby(['instrument'], observed=True)
    g["STD5"].plot(legend=True)
    # plt.savefig("/home/qdata/qlib_data/test/fig_stock.png")
    plt.show()
    print("show after")

def index_inner(df):
    time_uni = np.sort(df['time_idx'].unique())
    time_uni_dict = dict(enumerate(time_uni.flatten(), 1))
    time_uni_dict = {v: k for k, v in time_uni_dict.items()}    
    df["time_idx"]  = df["time_idx"].map(time_uni_dict)  
    return df
    
def test_index():
    data = pd.read_pickle("/home/qdata/qlib_data/test/time_idx.pkl")    
    data = data.groupby("instrument").apply(lambda df: index_inner(df))       
    print("after data",data.describe())
  
def test_data_compare():
    data1 = pd.read_pickle("/home/qdata/qlib_data/test/instrument_test.pkl")    
    data2 = pd.read_pickle("/home/qdata/qlib_data/test/instrument_train.pkl") 
    compare = pd.DataFrame(data1.keys().to_frame().eq(data2.keys().to_frame()))
    print("compare result",compare)
  
def test_viz():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.style.use('tableau-colorblind10') 
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    fig, ax = plt.subplots()
    
    np.random.seed(1)
    ts = pd.Series(np.random.randn(100), index=pd.date_range("1/1/2020", periods=100))
    ts = ts.cumsum()
    # ax.set_title(f"averages")
    # ax.set_xlabel("name")
    # ax.set_ylabel("Prediction")    
    fig.tight_layout()
    fig.legend()   
    ts.plot()
    plt.show()
    print("plot end")

def test_shift():
    data = [0.7186174392700195, -0.4756629467010498, -0.1364290714263916, -0.6151974201202393, -1.0313987731933594, 0.590062141418457, -0.13806819915771484, -2.0405828952789307, -6.1782121658325195, 1.0157227516174316, 1.0055303573608398, 0.07420778274536133, 2.8745293617248535, -0.6084561347961426, 0.9003639221191406, -3.322005271911621, 1.5148401260375977]
    # Creating dataframe using 'data' and 'index'
    df = pd.DataFrame(data = data, columns = ['value'])
    print("df",df)    
    df['mean'] = df['value'].rolling(window=5,min_periods=1).mean() 
    print("df after",df)
                 
def test_build_datetime():        
    begin_date = '2010-10-16'
    end_date = '2011-10-1'     
    df = pd.DataFrame({'forecast_date':pd.date_range(begin_date, end_date)})    
    df['dayofweek'] = df['forecast_date'].dt.dayofweek
    print(df)

def test_scipy_peak():
    x = electrocardiogram()[2000:4000]
    peaks, _ = find_peaks(x, height=0,distance=150)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()

def test_peak_combine():
    file_dir = "/home/qdata/project/qlib/custom/data/asis/view_data"  
    filenames = os.listdir(file_dir)
    for file in filenames:
        whole_file = file_dir + "/" + file
        view_data = np.load(whole_file)[:,1]
        peaks, _ = find_peaks(view_data, height=1.02,distance=5)
        if len(peaks)<2:
            continue
        plt.plot(view_data)
        plt.plot(peaks, view_data[peaks], "x")
        base_axis = np.zeros_like(view_data)
        base_axis = base_axis + 0.95
        plt.plot(base_axis, "--", color="gray")
        plt.show()        

def test_timequery():
    
    query_date = dt.datetime.strptime('20220606 09:10:00', '%Y%m%d %H:%M:%S')
    data = np.expand_dims(np.array([1,query_date]),0)
    df = pd.DataFrame(data,columns=['id','date'])
    df['date'] = df['date'].dt.tz_localize(tz='Asia/Shanghai')
    df['timestamp'] = df['date'].astype(np.int64)//10 ** 9
    test_df = pd.read_csv("/home/liang/test/test_df.csv")
    ret = test_df[(test_df['timestamp']==int(query_date.timestamp()))]
    print(ret)
            
if __name__ == "__main__":
    # test_scipy_peak()
    test_timequery()
    # test_peak_combine()
    # test_pd_index()
    # test_pd_timeser()
    # test_shift()
    # test_join()
    # test_merge()
    # test_columns()
    # test_group2()
    # test_diff()
    # test_vis()
    # test_tft_vis()
    # test_tft_stock_vis()
    # test_tft()
    # test_tft_data_view()
    # test_index()
    # test_data_compare()
    # test_viz()
    # test_build_datetime()
    
    