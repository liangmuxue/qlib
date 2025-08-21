from data_extract.his_data_extractor import MarketType

def transfer_order_book_id(instrument,type=1):
    """股票代码转换为rqalpha格式
        PARAMS:
             type 1 沪市A股 0 深市A股
    """
    
    if isinstance(instrument,int):
        instrument = str(instrument)
    # 对于深市相关编码规则，需要补充到6位数
    if len(instrument)<6:
        instrument = instrument.zfill(6)
    if type==1:
        return "{}.XSHG".format(instrument)
    return "{}.XSHE".format(instrument)

def transfer_futures_order_book_id(instrument,type=1):
    """期货代码转换为rqalpha格式
        PARAMS:
             type 1 沪市A股 0 深市A股
    """
    
    if isinstance(instrument,int):
        instrument = str(instrument)
    # 对于深市相关编码规则，需要补充到6位数
    if len(instrument)<6:
        instrument = instrument.zfill(6)
    if type==1:
        return "{}.XSHG".format(instrument)
    return "{}.XSHE".format(instrument)

def transfer_instrument(order_book_id):
    """股票代码转换为通用格式"""
    
    if order_book_id.endswith(".XSHG"):
        return order_book_id.split(".")[0]
    return order_book_id.split(".")[0]

def judge_market(order_book_id):
    """根据代码取得市场类别,XSHE-深证A股 XSHG 上证A股"""
    
    if order_book_id.endswith("XSHE"):
        return MarketType.SZ.value
    return MarketType.SH.value