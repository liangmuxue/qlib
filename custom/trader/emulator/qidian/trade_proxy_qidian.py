from trader.emulator.base_trade_proxy import BaseTrade
from rqalpha.portfolio import Portfolio
from . import traderapi

from cus_utils.process import IFakeSyncCall
from cus_utils.log_util import AppLogger
logger = AppLogger()

class TraderSpi(traderapi.CTORATstpTraderSpi):
    def __init__(self, api,caller_stub,**kwargs):
        traderapi.CTORATstpTraderSpi.__init__(self)
        self.__api = api
        self.__req_id = 0
        self.__front_id = 0
        self.__session_id = 0
        
        self.user_id = kwargs["user_id"]
        self.password = kwargs["password"]
        self.account_id = kwargs["account_id"]
        # 沪市股东账号
        self.SSE_ShareHolderID = kwargs["SSE_ShareHolderID"]
        # 深市股东账号
        self.SZSE_ShareHolderID = kwargs["SZSE_ShareHolderID"]   
           
        self.caller_stub = caller_stub

    def OnFrontConnected(self) -> "void":
        logger.debug('OnFrontConnected')

        # 获取终端信息
        self.__req_id += 1
        ret = self.__api.ReqGetConnectionInfo(self.__req_id)
        if ret != 0:
            logger.debug('ReqGetConnectionInfo fail, ret[%d]' % ret)
        


    def OnFrontDisconnected(self, nReason: "int") -> "void":
        logger.debug('OnFrontDisconnected: [%d]' % nReason)

    
    def OnRspGetConnectionInfo(self, pConnectionInfoField: "CTORATstpConnectionInfoField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('inner_ip_address[%s]' % pConnectionInfoField.InnerIPAddress)
            logger.debug('inner_port[%d]' % pConnectionInfoField.InnerPort)
            logger.debug('outer_ip_address[%s]' % pConnectionInfoField.OuterIPAddress)
            logger.debug('outer_port[%d]' % pConnectionInfoField.OuterPort)
            logger.debug('mac_address[%s]' % pConnectionInfoField.MacAddress)

            #请求登录
            login_req = traderapi.CTORATstpReqUserLoginField()

            # 支持以用户代码、资金账号和股东账号方式登录
            # （1）以用户代码方式登录
            login_req.LogInAccount = self.user_id
            login_req.LogInAccountType = traderapi.TORA_TSTP_LACT_UserID
            # （2）以资金账号方式登录
            #login_req.DepartmentID = DepartmentID
            #login_req.LogInAccount = AccountID
            #login_req.LogInAccountType = traderapi.TORA_TSTP_LACT_AccountID
            # （3）以上海股东账号方式登录
            # login_req.LogInAccount = SSE_ShareHolderID
            # login_req.LogInAccountType = traderapi.TORA_TSTP_LACT_SHAStock
            # （4）以深圳股东账号方式登录
            #login_req.LogInAccount = SZSE_ShareHolderID
            #login_req.LogInAccountType = traderapi.TORA_TSTP_LACT_SZAStock

            # 支持以密码和指纹(移动设备)方式认证
            # （1）密码认证
            # 密码认证时AuthMode可不填
            #login_req.AuthMode = traderapi.TORA_TSTP_AM_Password
            login_req.Password = self.password
            # （2）指纹认证
            # 非密码认证时AuthMode必填
            #login_req.AuthMode = traderapi.TORA_TSTP_AM_FingerPrint
            #login_req.DeviceID = '03873902'
            #login_req.CertSerial = '9FAC09383D3920CAEFF039'

            # 终端信息采集
            # UserProductInfo填写终端名称
            login_req.UserProductInfo = 'HXRAXYRNRG'
            # 按照监管要求填写终端信息
            login_req.TerminalInfo = 'PC;IIP=123.112.154.118;IPORT=50361;LIP=192.168.118.107;MAC=54EE750B1713FCF8AE5CBD58;HD=TF655AY91GHRVL'
            # 以下内外网IP地址若不填则柜台系统自动采集，若填写则以终端填值为准报送
            #login_req.MacAddress = '5C-87-9C-96-F3-E3'
            #login_req.InnerIPAddress = '10.0.1.102'
            #login_req.OuterIPAddress = '58.246.43.50'

            self.__req_id += 1
            ret = self.__api.ReqUserLogin(login_req, self.__req_id)
            if ret != 0:
                logger.debug('ReqUserLogin fail, ret[%d]' % ret)
            
        else:
            logger.debug('GetConnectionInfo fail, [%d] [%d] [%s]!!!' % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspUserLogin(self, pRspUserLoginField: "CTORATstpRspUserLoginField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('Login success! {},LogInAccount:{}'.format(nRequestID,pRspUserLoginField.LogInAccount))

            self.__front_id = pRspUserLoginField.FrontID
            self.__session_id = pRspUserLoginField.SessionID


    def OnRspUserPasswordUpdate(self, pUserPasswordUpdateField: "CTORATstpUserPasswordUpdateField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('OnRspUserPasswordUpdate: OK! [%d]' % nRequestID)
        else:
            logger.debug('OnRspUserPasswordUpdate: Error! [%d] [%d] [%s]' 
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspOrderInsert(self, pInputOrderField: "CTORATstpInputOrderField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('OnRspOrderInsert: OK! [%d]' % nRequestID)
        else:
            logger.debug('OnRspOrderInsert: Error! [%d] [%d] [%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspOrderAction(self, pInputOrderActionField: "CTORATstpInputOrderActionField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('OnRspOrderAction: OK! [%d]' % nRequestID)
        else:
            logger.debug('OnRspOrderAction: Error! [%d] [%d] [%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspInquiryJZFund(self, pRspInquiryJZFundField: "CTORATstpRspInquiryJZFundField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('OnRspInquiryJZFund: OK! [%d] [%.2f] [%.2f]'
                % (nRequestID, pRspInquiryJZFundField.UsefulMoney, pRspInquiryJZFundField.FetchLimit))
        else:
            logger.debug('OnRspInquiryJZFund: Error! [%d] [%d] [%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspTransferFund(self, pInputTransferFundField: "CTORATstpInputTransferFundField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int") -> "void":
        if pRspInfoField.ErrorID == 0:
            logger.debug('OnRspTransferFund: OK! [%d]' % nRequestID)
        else:
            logger.debug('OnRspTransferFund: Error! [%d] [%d] [%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRtnOrder(self, pOrderField: "CTORATstpOrderField") -> "void":
        logger.debug('OnRtnOrder: InvestorID[%s] SecurityID[%s] OrderRef[%d] OrderLocalID[%s] LimitPrice[%.2f] VolumeTotalOriginal[%d] OrderSysID[%s] OrderStatus[%s]'
            % (pOrderField.InvestorID, pOrderField.SecurityID, pOrderField.OrderRef, pOrderField.OrderLocalID, 
            pOrderField.LimitPrice, pOrderField.VolumeTotalOriginal, pOrderField.OrderSysID, pOrderField.OrderStatus))


    def OnRtnTrade(self, pTradeField: "CTORATstpTradeField") -> "void":
        logger.debug('OnRtnTrade: TradeID[%s] InvestorID[%s] SecurityID[%s] OrderRef[%d] OrderLocalID[%s] Price[%.2f] Volume[%d]'
            % (pTradeField.TradeID, pTradeField.InvestorID, pTradeField.SecurityID,
               pTradeField.OrderRef, pTradeField.OrderLocalID, pTradeField.Price, pTradeField.Volume))


    def OnRtnMarketStatus(self, pMarketStatusField: "CTORATstpMarketStatusField") -> "void":
        logger.debug('OnRtnMarketStatus: MarketID[%s] MarketStatus[%s]'
            % (pMarketStatusField.MarketID, pMarketStatusField.MarketStatus))


    def OnRspQrySecurity(self, pSecurityField: "CTORATstpSecurityField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQrySecurity[%d]: SecurityID[%s] SecurityName[%s] MarketID[%s] OrderUnit[%s] OpenDate[%s] UpperLimitPrice[%.2f] LowerLimitPrice[%.2f]'
                % (nRequestID, pSecurityField.SecurityID, pSecurityField.SecurityName, pSecurityField.MarketID,
                pSecurityField.OrderUnit, pSecurityField.OpenDate, pSecurityField.UpperLimitPrice, pSecurityField.LowerLimitPrice))
        else:
            logger.debug('查询合约结束[%d] ErrorID[%d] ErrorMsg[%s]'
            % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspQryInvestor(self, pInvestorField: "CTORATstpInvestorField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQryInvestor[%d]: InvestorID[%s] InvestorName[%s] Operways[%s]'
                %(nRequestID, pInvestorField.InvestorID, pInvestorField.InvestorName, 
                pInvestorField.Operways))
        else:
            logger.debug('查询投资者结束[%d] ErrorID[%d] ErrorMsg[%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspQryShareholderAccount(self, pShareholderAccountField: "CTORATstpShareholderAccountField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQryShareholderAccount[%d]: InvestorID[%s] ExchangeID[%s] ShareholderID[%s]'
                %(nRequestID, pShareholderAccountField.InvestorID, pShareholderAccountField.ExchangeID, pShareholderAccountField.ShareholderID))
        else:
            logger.debug('查询股东账户结束[%d] ErrorID[%d] ErrorMsg[%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspQryTradingAccount(self, pTradingAccountField: "CTORATstpTradingAccountField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQryTradingAccount[%d]: DepartmentID[%s] InvestorID[%s] AccountID[%s] CurrencyID[%s] UsefulMoney[%.2f] FetchLimit[%.2f]'
                % (nRequestID, pTradingAccountField.DepartmentID, pTradingAccountField.InvestorID, pTradingAccountField.AccountID, pTradingAccountField.CurrencyID,
                pTradingAccountField.UsefulMoney, pTradingAccountField.FetchLimit))
        else:
            logger.debug('查询资金账号结束[%d] ErrorID[%d] ErrorMsg[%s]'
                % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))


    def OnRspQryOrder(self, pOrderField: "CTORATstpOrderField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQryOrder[%d]: SecurityID[%s] OrderLocalID[%s] OrderRef[%d] OrderSysID[%s] VolumeTraded[%d] OrderStatus[%s] OrderSubmitStatus[%s], StatusMsg[%s]'
                % (nRequestID, pOrderField.SecurityID, pOrderField.OrderLocalID, pOrderField.OrderRef, pOrderField.OrderSysID, 
                pOrderField.VolumeTraded, pOrderField.OrderStatus, pOrderField.OrderSubmitStatus, pOrderField.StatusMsg))
        else:
            logger.debug('查询报单结束[%d] ErrorID[%d] ErrorMsg[%s]'
                 % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))

    def OnRspQryPosition(self, pPositionField: "CTORATstpPositionField", pRspInfoField: "CTORATstpRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if bIsLast != 1:
            logger.debug('OnRspQryPosition[%d]: InvestorID[%s] SecurityID[%s] HistoryPos[%d] TodayBSPos[%d] TodayBSPosFrozen[%d] TodayPRPos[%d] AvailablePosition[%d]'
                % (nRequestID, pPositionField.InvestorID, pPositionField.SecurityID, pPositionField.HistoryPos, pPositionField.TodayBSPosFrozen,
                pPositionField.TodayBSPos, pPositionField.TodayPRPos,pPositionField.AvailablePosition))
        else:
            logger.debug('查询持仓结束[%d] ErrorID[%d] ErrorMsg[%s]'
                 % (nRequestID, pRspInfoField.ErrorID, pRspInfoField.ErrorMsg))

        self.caller_stub.onFakeSyncCall('get_portfolio', (pPositionField))
     
class QidianTrade(BaseTrade):
    """奇点的仿真交易类"""
    
    def __init__(
        self,
        context,
        **kwargs,
    ):   
        super().__init__(context,**kwargs)  
        
        self.token = kwargs["token"]
        self.end_point = kwargs["end_point"]
        self.account_id = kwargs["account_id"]
        self.account_alias = kwargs["account_alias"]
        
        # 初始化环境以及登录
        self.init_env()
        
    def init_env(self):
        
        api = traderapi.CTORATstpTraderApi.CreateTstpTraderApi('./flow', False)
        self.api = api
        # 创建回调对象,注入自身用于回调
        spi = TraderSpi(api,self)
    
        # 注册回调接口
        api.RegisterSpi(spi)
    
        # 注册单个交易前置服务地址
        api.RegisterFront(self.end_point)
        # 注册多个交易前置服务地址，用逗号隔开
        #api.RegisterFront('tcp://10.0.1.101:6500,tcp://10.0.1.101:26500')
        # 注册名字服务器地址，支持多服务地址逗号隔开
        #api.RegisterNameServer('tcp://10.0.1.101:52370')
        #api.RegisterNameServer('tcp://10.0.1.101:52370,tcp://10.0.1.101:62370')
    
        #订阅私有流
        api.SubscribePrivateTopic(traderapi.TORA_TERT_QUICK)
        #订阅公有流
        api.SubscribePublicTopic(traderapi.TORA_TERT_QUICK)
    
        # 启动接口
        api.Init()
        
    @IFakeSyncCall.FAKE_SYNCALL
    def get_positions(self):  
        portfolio = self.get_portfolio()
        return portfolio.get_positions()
        
    @IFakeSyncCall.FAKE_SYNCALL
    def get_portfolio(self):
        """取得当前快照信息,借用rqalpha相关对象"""        
        
        portfolio = Portfolio()
        pPositionField = yield (self.api.OnRspQryPosition, ())
        logger.debug("pPositionField:{}".format(pPositionField))
    
    