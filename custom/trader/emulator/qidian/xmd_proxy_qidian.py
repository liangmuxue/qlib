from trader.emulator.base_xmd_proxy import BaseXmd

from . import xmdapi

class MdSpi(xmdapi.CTORATstpXMdSpi):
    def __init__(self, api):
        xmdapi.CTORATstpXMdSpi.__init__(self)
        self.__api = api

    def OnFrontConnected(self):
        print("OnFrontConnected")
        
        #请求登录，目前未校验登录用户，请求域置空即可
        login_req = xmdapi.CTORATstpReqUserLoginField()
        self.__api.ReqUserLogin(login_req, 1)

    def OnRspUserLogin(self, pRspUserLoginField, pRspInfoField, nRequestID):
        if pRspInfoField.ErrorID == 0:
            print('Login success! [%d]' % nRequestID)

            '''
            订阅行情
            当sub_arr中只有一个"00000000"的合约且ExchangeID填TORA_TSTP_EXD_SSE或TORA_TSTP_EXD_SZSE时，订阅单市场所有合约行情
            当sub_arr中只有一个"00000000"的合约且ExchangeID填TORA_TSTP_EXD_COMM时，订阅全市场所有合约行情
            其它情况，订阅sub_arr集合中的合约行情
            '''
            sub_arr = [b'600004']
            ret = self.__api.SubscribeMarketData(sub_arr, xmdapi.TORA_TSTP_EXD_SSE)
            if ret != 0:
                print('SubscribeMarketData fail, ret[%d]' % ret)
            else:
                print('SubscribeMarketData success, ret[%d]' % ret)



            sub_arr = [b'600004']
            ret = self.__api.UnSubscribeMarketData(sub_arr, xmdapi.TORA_TSTP_EXD_SSE)
            if ret != 0:
                print('UnSubscribeMarketData fail, ret[%d]' % ret)
            else:
                print('SubscribeMarketData success, ret[%d]' % ret)


        else:
            print('Login fail!!! [%d] [%d] [%s]'
                %(nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspSubMarketData(self, pSpecificSecurityField, pRspInfoField):
        if pRspInfoField.ErrorID == 0:
            print('OnRspSubMarketData: OK!')
        else:
            print('OnRspSubMarketData: Error! [%d] [%s]'
                %(pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspUnSubMarketData(self, pSpecificSecurityField, pRspInfoField):
        if pRspInfoField.ErrorID == 0:
            print('OnRspUnSubMarketData: OK!')
        else:
            print('OnRspUnSubMarketData: Error! [%d] [%s]'
                %(pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))

    def OnRtnMarketData(self, pMarketDataField):
        print("SecurityID[%s] SecurityName[%s] LastPrice[%.2f] Volume[%d] Turnover[%.2f] BidPrice1[%.2f] BidVolume1[%d] AskPrice1[%.2f] AskVolume1[%d] UpperLimitPrice[%.2f] LowerLimitPrice[%.2f]"
            % (pMarketDataField.SecurityID, pMarketDataField.SecurityName, pMarketDataField.LastPrice, pMarketDataField.Volume,
               pMarketDataField.Turnover, pMarketDataField.BidPrice1, pMarketDataField.BidVolume1, pMarketDataField.AskPrice1,
               pMarketDataField.AskVolume1, pMarketDataField.UpperLimitPrice, pMarketDataField.LowerLimitPrice))
        
class QidianXmd(BaseXmd):
    """华鑫奇点的行情实现类"""
    
    def __init__(
        self,
        **kwargs,
    ):   
        super().__init__(kwargs)  
        url = "tcp://210.14.72.16:9402"
        self.url = kwargs["xmd_url"]
    
    def init_env(self):
        # 打印接口版本号
        print(xmdapi.CTORATstpXMdApi_GetApiVersion())
        # 创建接口对象
        api = xmdapi.CTORATstpXMdApi_CreateTstpXMdApi()
        # 创建回调对象
        spi = MdSpi(api)
        # 注册回调接口
        api.RegisterSpi(spi)
        # 注册单个行情前置服务地址
        api.RegisterFront(self.url)
        # 注册多个行情前置服务地址，用逗号隔开
        #api.RegisterFront("tcp://10.0.1.101:6402,tcp://10.0.1.101:16402")
        # 注册名字服务器地址，支持多服务地址逗号隔开
        #api.RegisterNameServer('tcp://10.0.1.101:52370')
        #api.RegisterNameServer('tcp://10.0.1.101:52370,tcp://10.0.1.101:62370')
    
        # 启动接口
        api.Init()
          

