from trader.emulator.base_trade_proxy import BaseTrade
from rqalpha.portfolio import Portfolio

import copy
import threading

from cus_utils.process import IFakeSyncCall
from cus_utils.log_util import AppLogger
logger = AppLogger()

import trader.emulator.ctpapi.thosttraderapi as tdapi 

from rqalpha.const import SIDE,ORDER_STATUS as RQ_ORDER_STATUS
from rqalpha.apis import Environment
from rqalpha.core.events import EVENT, Event
from trader.rqalpha.model.trade import Trade

emaphore = threading.Semaphore(0)

class TdImpl(tdapi.CThostFtdcTraderSpi):
    def __init__(self, host, broker, user, password, appid, authcode,listenner=None):
        super().__init__()

        self.broker = broker
        self.user = user
        self.password = password
        self.appid = appid
        self.authcode = authcode

        self.TradingDay = ""
        self.FrontID = 0
        self.SessionID = 0
        self.OrderRef = 0
        
        self.listenner = listenner
        
        self.api: tdapi.CThostFtdcTraderApi = tdapi.CThostFtdcTraderApi.CreateFtdcTraderApi()
        self.api.RegisterSpi(self)
        self.api.RegisterFront(host)
        self.api.SubscribePrivateTopic(tdapi.THOST_TERT_QUICK)
        self.api.SubscribePublicTopic(tdapi.THOST_TERT_QUICK)
        
        ######### 状态相关 #########
        # 初始化状态 0 未初始化 1 初始化成功 2 初始化失败
        self.init_status = 0
        # 登录状态,0 未登录 1 登录化成功 2 登录失败
        self.login_status = 0

    def Run(self):
        self.api.Init()

    def OnFrontConnected(self):
        logger.info("OnFrontConnected")

        req = tdapi.CThostFtdcReqAuthenticateField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.AppID = self.appid
        req.AuthCode = self.authcode
        self.api.ReqAuthenticate(req, 0)

    def OnFrontDisconnected(self, nReason: int):
        logger.info(f"OnFrontDisconnected.[nReason={nReason}]")

    def OnRspAuthenticate(
            self,
            pRspAuthenticateField: tdapi.CThostFtdcRspAuthenticateField,
            pRspInfo: tdapi.CThostFtdcRspInfoField,
            nRequestID: int,
            bIsLast: bool,
    ):
        """客户端认证响应"""
        
        if pRspInfo and pRspInfo.ErrorID != 0:
            self.init_status = 2
            logger.warning("Authenticate Failed")
            return
        
        logger.info("Authenticate Suceess")
        self.init_status = 1
            
        # 登录
        req = tdapi.CThostFtdcReqUserLoginField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.Password = self.password
        req.UserProductInfo = "demo"
        self.api.ReqUserLogin(req, 0)

    def OnRspUserLogin(
            self,
            pRspUserLogin: tdapi.CThostFtdcRspUserLoginField,
            pRspInfo: tdapi.CThostFtdcRspInfoField,
            nRequestID: int,
            bIsLast: bool,
    ):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            self.login_status = 2
            logger.warning("UserLogin Failed")
            return
        
        self.login_status = 1
        logger.info(f"Login succeed. TradingDay: {pRspUserLogin.TradingDay}, MaxOrderRef: {pRspUserLogin.MaxOrderRef}, SystemName: {pRspUserLogin.SystemName}")
        self.TradingDay = pRspUserLogin.TradingDay
        self.FrontID = pRspUserLogin.FrontID
        self.SessionID = pRspUserLogin.SessionID
        self.OrderRef = 1

    def OnRspOrderInsert(self, pInputOrder, pRspInfo, nRequestID, bIsLast):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            logger.warning(f"OnRspOrderInsert failed: {pRspInfo.ErrorMsg}")
            self.listenner.on_order_failed(pInputOrder)
            return 

        if pInputOrder is not None:
            # 回调入口类，处理后续业务逻辑
            self.listenner.on_order_confirm(pInputOrder)

    def OnRspOrderAction(self, pInputOrderAction: "CThostFtdcInputOrderActionField", pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspOrderAction failed: {pRspInfo.ErrorMsg}")

        if pInputOrderAction is not None:
            print(f"OnRspOrderAction:"
                  f"UserID={pInputOrderAction.UserID} "
                  f"ActionFlag={pInputOrderAction.ActionFlag} "
                  f"OrderActionRef={pInputOrderAction.OrderActionRef} "
                  f"BrokerID={pInputOrderAction.BrokerID} "
                  f"InvestorID={pInputOrderAction.InvestorID} "
                  f"ExchangeID={pInputOrderAction.ExchangeID} "
                  f"InstrumentID={pInputOrderAction.InstrumentID} "
                  f"FrontID={pInputOrderAction.FrontID} "
                  f"SessionID={pInputOrderAction.SessionID} "
                  f"OrderRef={pInputOrderAction.OrderRef} "
                  f"OrderSysID={pInputOrderAction.OrderSysID} "
                  f"InvestUnitID={pInputOrderAction.InvestUnitID} "
                  f"IPAddress={pInputOrderAction.IPAddress} "
                  f"MacAddress={pInputOrderAction.MacAddress} "
                  )

    def OnErrRtnOrderInsert(self, pInputOrder: "CThostFtdcInputOrderField", pRspInfo: "CThostFtdcRspInfoField") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnErrRtnOrderInsert failed: {pRspInfo.ErrorMsg}")

        if pInputOrder is not None:
            print(f"OnErrRtnOrderInsert:"
                  f"UserID={pInputOrder.UserID} "
                  f"BrokerID={pInputOrder.BrokerID} "
                  f"InvestorID={pInputOrder.InvestorID} "
                  f"ExchangeID={pInputOrder.ExchangeID} "
                  f"InstrumentID={pInputOrder.InstrumentID} "
                  f"Direction={pInputOrder.Direction} "
                  f"CombOffsetFlag={pInputOrder.CombOffsetFlag} "
                  f"CombHedgeFlag={pInputOrder.CombHedgeFlag} "
                  f"OrderPriceType={pInputOrder.OrderPriceType} "
                  f"LimitPrice={pInputOrder.LimitPrice} "
                  f"VolumeTotalOriginal={pInputOrder.VolumeTotalOriginal} "
                  f"OrderRef={pInputOrder.OrderRef} "
                  f"TimeCondition={pInputOrder.TimeCondition} "
                  f"GTDDate={pInputOrder.GTDDate} "
                  f"VolumeCondition={pInputOrder.VolumeCondition} "
                  f"MinVolume={pInputOrder.MinVolume} "
                  f"RequestID={pInputOrder.RequestID} "
                  f"InvestUnitID={pInputOrder.InvestUnitID} "
                  f"CurrencyID={pInputOrder.CurrencyID} "
                  f"AccountID={pInputOrder.AccountID} "
                  f"ClientID={pInputOrder.ClientID} "
                  f"IPAddress={pInputOrder.IPAddress} "
                  f"MacAddress={pInputOrder.MacAddress} "
                  )

    def OnErrRtnOrderAction(self, pOrderAction: "CThostFtdcOrderActionField", pRspInfo: "CThostFtdcRspInfoField") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnErrRtnOrderAction failed: {pRspInfo.ErrorMsg}")

        if pOrderAction is not None:
            print(f"OnErrRtnOrderAction:"
                  f"UserID={pOrderAction.UserID} "
                  f"ActionFlag={pOrderAction.ActionFlag} "
                  f"OrderActionRef={pOrderAction.OrderActionRef} "
                  f"BrokerID={pOrderAction.BrokerID} "
                  f"InvestorID={pOrderAction.InvestorID} "
                  f"ExchangeID={pOrderAction.ExchangeID} "
                  f"InstrumentID={pOrderAction.InstrumentID} "
                  f"FrontID={pOrderAction.FrontID} "
                  f"SessionID={pOrderAction.SessionID} "
                  f"OrderRef={pOrderAction.OrderRef} "
                  f"OrderSysID={pOrderAction.OrderSysID} "
                  f"InvestUnitID={pOrderAction.InvestUnitID} "
                  f"IPAddress={pOrderAction.IPAddress} "
                  f"MacAddress={pOrderAction.MacAddress} "
                  )

    def OnRspQryInstrument(self, pInstrument: tdapi.CThostFtdcInstrumentField, pRspInfo: "CThostFtdcRspInfoField",
                           nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryInstrument failed: {pRspInfo.ErrorMsg}")
            return

        if pInstrument is not None:
            print(
                f"OnRspQryInstrument:"
                f"InstrumentID={pInstrument.InstrumentID} "
                f"InstrumentName={pInstrument.InstrumentName} "
                f"ExchangeID={pInstrument.ExchangeID} "
                f"ProductClass={pInstrument.ProductClass} "
                f"ProductID={pInstrument.ProductID} "
                f"VolumeMultiple={pInstrument.VolumeMultiple} "
                f"PositionType={pInstrument.PositionType} "
                f"PositionDateType={pInstrument.PositionDateType} "
                f"PriceTick={pInstrument.PriceTick} "
                f"ExpireDate={pInstrument.ExpireDate} "
                f"UnderlyingInstrID={pInstrument.UnderlyingInstrID} "
                f"StrikePrice={pInstrument.StrikePrice} "
                f"OptionsType={pInstrument.OptionsType} "
                f"MinLimitOrderVolume={pInstrument.MinLimitOrderVolume} "
                f"MaxLimitOrderVolume={pInstrument.MaxLimitOrderVolume} "
            )
        if bIsLast == True:
           semaphore.release()

    def OnRspQryExchange(self, pExchange: "CThostFtdcExchangeField", pRspInfo: "CThostFtdcRspInfoField",
                         nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryExchange failed: {pRspInfo.ErrorMsg}")
            return
        print(f"OnRspQryExchange:"
              f"ExchangeID={pExchange.ExchangeID} "
              f"ExchangeName={pExchange.ExchangeName} "
              )
        if bIsLast == True:
            semaphore.release()

    def OnRspQryProduct(self, pProduct: "CThostFtdcProductField", pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int",
                        bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryProduct failed: {pRspInfo.ErrorMsg}")
            return
        print(f"OnRspQryProduct:{pProduct.ProductID} "
              f"ProductName={pProduct.ProductName} "
              f"ExchangeID={pProduct.ExchangeID} "
              f"ProductClass={pProduct.ProductClass} "
              f"VolumeMultiple={pProduct.VolumeMultiple} "
              f"PriceTick={pProduct.PriceTick} "
              f"PositionType={pProduct.PositionType} "
              f"PositionDateType={pProduct.PositionDateType} "
              f"TradeCurrencyID={pProduct.TradeCurrencyID} "
              f"UnderlyingMultiple={pProduct.UnderlyingMultiple} "
              )
        if bIsLast == True:
            semaphore.release()

    def OnRtnOrder(self, pOrder):
        self.listenner.on_order_rtn(pOrder)
        
    def OnRtnTrade(self, pTrade):
        print(f"OnRtnTrade:"
              f"BrokerID={pTrade.BrokerID} "
              f"InvestorID={pTrade.InvestorID} "
              f"ExchangeID={pTrade.ExchangeID} "
              f"InstrumentID={pTrade.InstrumentID} "
              f"Direction={pTrade.Direction} "
              f"OffsetFlag={pTrade.OffsetFlag} "
              f"HedgeFlag={pTrade.HedgeFlag} "
              f"Price={pTrade.Price}  "
              f"Volume={pTrade.Volume} "
              f"OrderSysID={pTrade.OrderSysID} "
              f"OrderRef={pTrade.OrderRef} "
              f'TradeID={pTrade.TradeID} '
              f'TradeDate={pTrade.TradeDate} '
              f'TradeTime={pTrade.TradeTime} '
              f'ClientID={pTrade.ClientID} '
              f'TradingDay={pTrade.TradingDay} '
              f'OrderLocalID={pTrade.OrderLocalID} '
              f'BrokerOrderSeq={pTrade.BrokerOrderSeq} '
              f'InvestUnitID={pTrade.InvestUnitID} '
              f'ParticipantID={pTrade.ParticipantID} '
              )


    def OnRspQryOrder(self, pOrder: tdapi.CThostFtdcOrderField, pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int",
                      bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryOrder failed: {pRspInfo.ErrorMsg}")
            return

        if pOrder is not None:
            print(f"OnRspQryOrder:"
                  f"UserID={pOrder.UserID} "
                  f"BrokerID={pOrder.BrokerID} "
                  f"InvestorID={pOrder.InvestorID} "
                  f"ExchangeID={pOrder.ExchangeID} "
                  f"InstrumentID={pOrder.InstrumentID} "
                  f"Direction={pOrder.Direction} "
                  f"CombOffsetFlag={pOrder.CombOffsetFlag} "
                  f"CombHedgeFlag={pOrder.CombHedgeFlag} "
                  f"OrderPriceType={pOrder.OrderPriceType} "
                  f"LimitPrice={pOrder.LimitPrice} "
                  f"VolumeTotalOriginal={pOrder.VolumeTotalOriginal} "
                  f"FrontID={pOrder.FrontID} "
                  f"SessionID={pOrder.SessionID} "
                  f"OrderRef={pOrder.OrderRef} "
                  f"TimeCondition={pOrder.TimeCondition} "
                  f"GTDDate={pOrder.GTDDate} "
                  f"VolumeCondition={pOrder.VolumeCondition} "
                  f"MinVolume={pOrder.MinVolume} "
                  f"RequestID={pOrder.RequestID} "
                  f"InvestUnitID={pOrder.InvestUnitID} "
                  f"CurrencyID={pOrder.CurrencyID} "
                  f"AccountID={pOrder.AccountID} "
                  f"ClientID={pOrder.ClientID} "
                  f"IPAddress={pOrder.IPAddress} "
                  f"MacAddress={pOrder.MacAddress} "
                  f"OrderSysID={pOrder.OrderSysID} "
                  f"OrderStatus={pOrder.OrderStatus} "
                  f"StatusMsg={pOrder.StatusMsg} "
                  f"VolumeTotal={pOrder.VolumeTotal} "
                  f"VolumeTraded={pOrder.VolumeTraded} "
                  f"OrderSubmitStatus={pOrder.OrderSubmitStatus} "
                  f"TradingDay={pOrder.TradingDay} "
                  f"InsertDate={pOrder.InsertDate} "
                  f"InsertTime={pOrder.InsertTime} "
                  f"UpdateTime={pOrder.UpdateTime} "
                  f"CancelTime={pOrder.CancelTime} "
                  f"UserProductInfo={pOrder.UserProductInfo} "
                  f"ActiveUserID={pOrder.ActiveUserID} "
                  f"BrokerOrderSeq={pOrder.BrokerOrderSeq} "
                  f"TraderID={pOrder.TraderID} "
                  f"ClientID={pOrder.ClientID} "
                  f"ParticipantID={pOrder.ParticipantID} "
                  f"OrderLocalID={pOrder.OrderLocalID} "
                  )

        if bIsLast == True:
            semaphore.release()

    def OnRspQryTrade(self, pTrade: tdapi.CThostFtdcTradeField, pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int",
                      bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryTrade failed: {pRspInfo.ErrorMsg}")
            return

        if pTrade is not None:
            print(f"OnRspQryTrade:"
                  f"BrokerID={pTrade.BrokerID} "
                  f"InvestorID={pTrade.InvestorID} "
                  f"ExchangeID={pTrade.ExchangeID} "
                  f"InstrumentID={pTrade.InstrumentID} "
                  f"Direction={pTrade.Direction} "
                  f"OffsetFlag={pTrade.OffsetFlag} "
                  f"HedgeFlag={pTrade.HedgeFlag} "
                  f"Price={pTrade.Price}  "
                  f"Volume={pTrade.Volume} "
                  f"OrderSysID={pTrade.OrderSysID} "
                  f"OrderRef={pTrade.OrderRef} "
                  f'TradeID={pTrade.TradeID} '
                  f'TradeDate={pTrade.TradeDate} '
                  f'TradeTime={pTrade.TradeTime} '
                  f'ClientID={pTrade.ClientID} '
                  f'TradingDay={pTrade.TradingDay} '
                  f'OrderLocalID={pTrade.OrderLocalID} '
                  f'BrokerOrderSeq={pTrade.BrokerOrderSeq} '
                  f'InvestUnitID={pTrade.InvestUnitID} '
                  f'ParticipantID={pTrade.ParticipantID} '
                  )

        if bIsLast == True:
            semaphore.release()

    def OnRspQryTradingAccount(self, pTradingAccount: tdapi.CThostFtdcTradingAccountField,
                               pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryTradingAccount failed: {pRspInfo.ErrorMsg}")
            return

        if pTradingAccount is not None:
            print(f"OnRspQryTradingAccount: "
                  f"PreBalance={pTradingAccount.PreBalance} "
                  f"PreMargin={pTradingAccount.PreMargin} "
                  f"FrozenMargin={pTradingAccount.FrozenMargin} "
                  f"CurrMargin={pTradingAccount.CurrMargin} "
                  f"Commission={pTradingAccount.Commission} "
                  f"FrozenCommission={pTradingAccount.FrozenCommission} "
                  f"Available={pTradingAccount.Available} "
                  f"Balance={pTradingAccount.Balance} "
                  f"CloseProfit={pTradingAccount.CloseProfit} "
                  f"CurrencyID={pTradingAccount.CurrencyID} "
                  )

        if bIsLast == True:
            semaphore.release()

    def OnRspQryInvestor(self, pInvestor: "CThostFtdcInvestorField", pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryDepthMarketData failed: {pRspInfo.ErrorMsg}")
            return

        if pInvestor is not None:
            print(f"OnRspQryInvestor: "
                  f"InvestorID={pInvestor.InvestorID} "
                  f"InvestorName={pInvestor.InvestorName} "
                  f"Telephone={pInvestor.Telephone} "
                  )

        if bIsLast == True:
            semaphore.release()

    def OnRspQryDepthMarketData(self, pDepthMarketData: tdapi.CThostFtdcDepthMarketDataField,
                                pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f"OnRspQryDepthMarketData failed: {pRspInfo.ErrorMsg}")
            return

        if pDepthMarketData is not None:
            print(f"OnRspQryDepthMarketData: "
                  f"InstrumentID={pDepthMarketData.InstrumentID}"
                  f"LastPrice={pDepthMarketData.LastPrice} "
                  f"Volume={pDepthMarketData.Volume} "
                  f"OpenPrice={pDepthMarketData.OpenPrice} "
                  f"HighestPrice={pDepthMarketData.HighestPrice} "
                  f"LowestPrice={pDepthMarketData.LowestPrice} "
                  f"ClosePrice={pDepthMarketData.ClosePrice} "
                  f"OpenInterest={pDepthMarketData.OpenInterest} "
                  f"UpperLimitPrice={pDepthMarketData.UpperLimitPrice} "
                  f"LowerLimitPrice={pDepthMarketData.LowerLimitPrice} "
                  f"SettlementPrice={pDepthMarketData.SettlementPrice} "
                  f"PreSettlementPrice={pDepthMarketData.PreSettlementPrice} "
                  f"PreClosePrice={pDepthMarketData.PreClosePrice} "
                  f"PreOpenInterest={pDepthMarketData.PreOpenInterest} "
                  f"BidPrice1={pDepthMarketData.BidPrice1} "
                  f"BidVolume1={pDepthMarketData.BidVolume1} "
                  f"AskPrice1={pDepthMarketData.AskPrice1} "
                  f"AskVolume1={pDepthMarketData.AskVolume1} "
                  f"UpdateTime={pDepthMarketData.UpdateTime} "
                  f"UpdateMillisec={pDepthMarketData.UpdateMillisec} "
                  f"ActionDay={pDepthMarketData.ActionDay} "
                  f"TradingDay={pDepthMarketData.TradingDay} "
                  )

        if bIsLast == True:
            semaphore.release()

    def OnRspQryInstrumentCommissionRate(self, pInstrumentCommissionRate: tdapi.CThostFtdcInstrumentCommissionRateField,
                                         pRspInfo: tdapi.CThostFtdcRspInfoField, nRequestID: int, bIsLast: bool):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f'OnRspQryInstrumentCommissionRate failed: {pRspInfo.ErrorMsg}')
            exit(-1)
        if pInstrumentCommissionRate is not None:
            print(f"OnRspQryInstrumentCommissionRate:"
                  f"ExchangeID={pInstrumentCommissionRate.ExchangeID} "
                  f"InstrumentID={pInstrumentCommissionRate.InstrumentID} "
                  f"InvestorRange={pInstrumentCommissionRate.InvestorRange} "
                  f"InvestorID={pInstrumentCommissionRate.InvestorID} "
                  f"OpenRatioByMoney={pInstrumentCommissionRate.OpenRatioByMoney} "
                  f"OpenRatioByVolume={pInstrumentCommissionRate.OpenRatioByVolume} "
                  f"CloseRatioByMoney={pInstrumentCommissionRate.CloseRatioByMoney} "
                  f"CloseRatioByVolume={pInstrumentCommissionRate.CloseRatioByVolume} "
                  f"CloseTodayRatioByMoney={pInstrumentCommissionRate.CloseTodayRatioByMoney} "
                  f"CloseTodayRatioByVolume={pInstrumentCommissionRate.CloseTodayRatioByVolume} "
                  )
        if bIsLast == True:
            semaphore.release()

    def OnRspQryInstrumentMarginRate(self, pInstrumentMarginRate: tdapi.CThostFtdcInstrumentMarginRateField,
                                     pRspInfo: tdapi.CThostFtdcRspInfoField, nRequestID: int, bIsLast: bool):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f'OnRspQryInstrumentMarginRate failed: {pRspInfo.ErrorMsg}')
            exit(-1)
        if pInstrumentMarginRate is not None:
            print(f"OnRspQryInstrumentMarginRate:"
                  f"ExchangeID={pInstrumentMarginRate.ExchangeID} "
                  f"InstrumentID={pInstrumentMarginRate.InstrumentID} "
                  f"InvestorRange={pInstrumentMarginRate.InvestorRange} "
                  f"InvestorID={pInstrumentMarginRate.InvestorID} "
                  f"HedgeFlag={pInstrumentMarginRate.HedgeFlag} "
                  f"LongMarginRatioByMoney={pInstrumentMarginRate.LongMarginRatioByMoney} "
                  f"LongMarginRatioByVolume={pInstrumentMarginRate.LongMarginRatioByVolume} "
                  f"ShortMarginRatioByMoney={pInstrumentMarginRate.ShortMarginRatioByMoney} "
                  f"ShortMarginRatioByVolume={pInstrumentMarginRate.ShortMarginRatioByVolume} "
                  f"IsRelative={pInstrumentMarginRate.IsRelative} "
                  )
        if bIsLast == True:
            semaphore.release()

    def OnRspQryInstrumentOrderCommRate(self, pInstrumentOrderCommRate: tdapi.CThostFtdcInstrumentOrderCommRateField,
                                        pRspInfo: tdapi.CThostFtdcRspInfoField, nRequestID: int, bIsLast: bool):
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f'OnRspQryInstrumentOrderCommRate failed: {pRspInfo.ErrorMsg}')
            exit(-1)
        if pInstrumentOrderCommRate is not None:
            print(f"OnRspQryInstrumentOrderCommRate:"
                  f"ExchangeID={pInstrumentOrderCommRate.ExchangeID} "
                  f"InstrumentID={pInstrumentOrderCommRate.InstrumentID} "
                  f"InvestorRange={pInstrumentOrderCommRate.InvestorRange} "
                  f"InvestorID={pInstrumentOrderCommRate.InvestorID} "
                  f"HedgeFlag={pInstrumentOrderCommRate.HedgeFlag} "
                  f"OrderCommByVolume={pInstrumentOrderCommRate.OrderCommByVolume} "
                  f"OrderActionCommByVolume={pInstrumentOrderCommRate.OrderActionCommByVolume} "
                  )
        if bIsLast == True:
            semaphore.release()

    def OnRspQryTradingCode(self, pTradingCode: "CThostFtdcTradingCodeField", pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f'OnRspQryTradingCode failed: {pRspInfo.ErrorMsg}')
            exit(-1)
        if pTradingCode is not None:
            print(f"OnRspQryTradingCode:"
                  f"BrokerID={pTradingCode.BrokerID} "
                  f"InvestorID={pTradingCode.InvestorID} "
                  f"ExchangeID={pTradingCode.ExchangeID} "
                  f"ClientID={pTradingCode.ClientID} "
                  )
        if bIsLast == True:
            semaphore.release()

    def OnRspQrySettlementInfo(self, pSettlementInfo: "CThostFtdcSettlementInfoField", pRspInfo: "CThostFtdcRspInfoField", nRequestID: "int", bIsLast: "bool") -> "void":
        if pRspInfo is not None and pRspInfo.ErrorID != 0:
            print(f'OnRspQrySettlementInfo failed: {pRspInfo.ErrorMsg}')
            exit(-1)
        if pSettlementInfo is not None:
            # print(f"OnRspQrySettlementInfo:TradingDay={pSettlementInfo.TradingDay},InvestorID={pSettlementInfo.InvestorID},CurrencyID={pSettlementInfo.CurrencyID},Content={pSettlementInfo.Content}")
            print(pSettlementInfo.Content.decode('gbk'))
        if bIsLast == True:
            semaphore.release()

    def OnRtnInstrumentStatus(self, pInstrumentStatus: "CThostFtdcInstrumentStatusField") -> "void":
        print(f"OnRtnInstrumentStatus:"
              f"ExchangeID={pInstrumentStatus.ExchangeID} "
              f"InstrumentID={pInstrumentStatus.InstrumentID} "
              f"InstrumentStatus={pInstrumentStatus.InstrumentStatus}  "
              f"TradingSegmentSN={pInstrumentStatus.TradingSegmentSN} "
              f'EnterTime={pInstrumentStatus.EnterTime} '
              f"EnterReason={pInstrumentStatus.EnterReason} "
              )

    def QryInstrument(self, exchangeid, productid, instrumentid):
        req = tdapi.CThostFtdcQryInstrumentField()
        req.ExchangeID = exchangeid
        req.ProductID = productid
        req.InstrumentID = instrumentid
        self.api.ReqQryInstrument(req, 0)

    def QryExchange(self):
        req = tdapi.CThostFtdcQryExchangeField()
        self.api.ReqQryExchange(req, 0)

    def QryProduct(self, ExchangeID, ProductID):
        req = tdapi.CThostFtdcQryProductField()
        req.ExchangeID = ExchangeID
        req.ProductID = ProductID
        self.api.ReqQryProduct(req, 0)

    def QryPrice(self, ExchangeID, InstrumentID):
        req = tdapi.CThostFtdcQryDepthMarketDataField()
        req.ExchangeID = ExchangeID
        req.InstrumentID = InstrumentID
        self.api.ReqQryDepthMarketData(req, 0)

    def QryInvestor(self):
        req = tdapi.CThostFtdcQryInvestorField()
        req.BrokerID = self.broker
        self.api.ReqQryInvestor(req, 0)

    def QryAccount(self):
        req = tdapi.CThostFtdcQryTradingAccountField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        self.api.ReqQryTradingAccount(req, 0)

    def QryPosition(self, InstrumentID):
        req = tdapi.CThostFtdcQryInvestorPositionField()
        req.InvestorID = self.user
        req.BrokerID = self.broker
        req.InstrumentID = InstrumentID
        self.api.ReqQryInvestorPosition(req, 0)

    def QryPositionDetail(self, InstrumentID):
        req = tdapi.CThostFtdcQryInvestorPositionDetailField()
        req.InvestorID = self.user
        req.BrokerID = self.broker
        req.InstrumentID = InstrumentID
        self.api.ReqQryInvestorPositionDetail(req, 0)

    def QryOrder(self, InstrumentID):
        req = tdapi.CThostFtdcQryOrderField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.InstrumentID = InstrumentID
        self.api.ReqQryOrder(req, 0)

    def QryTrade(self, InstrumentID):
        req = tdapi.CThostFtdcQryTradeField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.InstrumentID = InstrumentID
        self.api.ReqQryTrade(req, 0)

    def QryCommissionRate(self, InstrumentID):
        req = tdapi.CThostFtdcQryInstrumentCommissionRateField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.InstrumentID = InstrumentID
        self.api.ReqQryInstrumentCommissionRate(req, 0)

    def QryMarginRate(self, ExchangeID, InstrumentID):
        req = tdapi.CThostFtdcQryInstrumentMarginRateField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.ExchangeID = ExchangeID
        req.InstrumentID = InstrumentID
        self.api.ReqQryInstrumentMarginRate(req, 0)

    def QryOrderCommRate(self, InstrumentID):
        req = tdapi.CThostFtdcQryInstrumentOrderCommRateField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.InstrumentID = InstrumentID
        self.api.ReqQryInstrumentOrderCommRate(req, 0)

    def QryTradingCode(self):
        req = tdapi.CThostFtdcQryTradingCodeField()
        self.api.ReqQryTradingCode(req, 0)

    def QrySettlementInfo(self, TradingDay):
        req = tdapi.CThostFtdcQrySettlementInfoField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        req.TradingDay = TradingDay
        self.api.ReqQrySettlementInfo(req, 0)

    def ConfirmSettlementInfo(self):
        req = tdapi.CThostFtdcSettlementInfoConfirmField()
        req.BrokerID = self.broker
        req.InvestorID = self.user
        self.api.ReqSettlementInfoConfirm(req, 0)

    def OrderInsert(self, ExchangeID, InstrumentID, Direction, Offset, PriceType, Price, Volume, TimeCondition, VolumeCondition, MinVolume,OrderRef=0):
        req = tdapi.CThostFtdcInputOrderField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.InvestorID = self.user
        req.ExchangeID = ExchangeID
        req.InstrumentID = InstrumentID
        req.Direction = Direction
        req.CombOffsetFlag = Offset
        req.CombHedgeFlag = tdapi.THOST_FTDC_HF_Speculation
        req.OrderPriceType = PriceType
        # 唯一编号，用于串接上下文
        req.OrderRef = OrderRef
        if Price != "":
            req.LimitPrice = float(Price)
        if Volume != "":
            req.VolumeTotalOriginal = int(Volume)
        req.TimeCondition = TimeCondition
        req.VolumeCondition = VolumeCondition
        if MinVolume != "":
            req.MinVolume = int(MinVolume)
        req.OrderRef = str(self.OrderRef)
        self.OrderRef = self.OrderRef + 1
        req.ForceCloseReason = tdapi.THOST_FTDC_FCC_NotForceClose
        req.ContingentCondition = tdapi.THOST_FTDC_CC_Immediately
        req.OrderLocalID = OrderRef
        self.api.ReqOrderInsert(req, 0)

    def OrderCancel(self, ExchangeID, InstrumentID, OrderSysID, FrontID, SessionID, OrderRef):
        req = tdapi.CThostFtdcInputOrderActionField()
        req.BrokerID = self.broker
        req.UserID = self.user
        req.InvestorID = self.user
        req.ExchangeID = ExchangeID
        req.InstrumentID = InstrumentID
        req.OrderSysID = OrderSysID
        if FrontID != "":
            req.FrontID = int(FrontID)
        if SessionID != "":
            req.SessionID = int(SessionID)
        req.OrderRef = OrderRef
        req.ActionFlag = tdapi.THOST_FTDC_AF_Delete
        self.api.ReqOrderAction(req, 0)
        
class QidianFuturesTrade(BaseTrade):
    """奇点的仿真交易类"""
    
    def __init__(
        self,
        context,
        account_alias=None,
    ):   
        super().__init__(context)  
        
        self.host = account_alias["host"]
        self.broker = account_alias["broker"]
        self.user = account_alias["user"]
        self.password = account_alias["password"]
        self.appid = account_alias["appid"]
        self.authcode = account_alias["authcode"]
        
        self.order_ref_id = 0
        self.order_queue = []
        # 初始化环境以及登录
        self.init_env()
        
    def init_env(self):
        
        host = self.host
        broker = self.broker
        user = self.user
        password = self.password
        appid = self.appid
        authcode = self.authcode
        
        tdImpl = TdImpl(host, broker, user, password, appid, authcode,listenner=self)
        self.api = tdImpl
        tdImpl.Run()
    
    def build_order_ref_id(self):
        
        self.order_ref_id += 1
        return self.order_ref_id
    
    ########################交易数据获取#################################  
        
    def find_cache_order(self,order_ref_id):
        """查找已生成订单"""
        
        # 匹配已存储订单
        for order in self.order_queue:
            if order.ref_id==order_ref_id:
                return order
        return None
    
    ########################交易请求#################################  
    
    def submit_order(self,order_in):
        
        # 如果没有登录到仿真系统，则不处理
        if self.api.login_status!=1:
            logger.warning("Have Not Login!")
            return 
        
        ExchangeID = order_in.exchange_id
        InstrumentID = order_in.order_book_id
        Direction = order_in.long_short
        if order_in.side=='BUY':
            Offset = 0
        else:
            Offset = 1
        # 默认交易类型为限价
        PriceType = "2"
        Price = order_in.price
        Volume = order_in.quantity
        # 挂单当日有效
        TimeCondition = "3"
        # 可以任意数量成单
        VolumeCondition = "1"
        MinVolume = "1"

        # 生成用于串接上下文的唯一编号
        OrderRef = self.build_order_ref_id()
        # 放入待处理队列
        order = copy.copy(order_in)
        order.ref_id = OrderRef
        order.res_result = {}
        self.order_queue.append(order)
                
        self.api.OrderInsert(ExchangeID, InstrumentID, Direction, Offset, PriceType, Price, Volume, TimeCondition,
                       VolumeCondition, MinVolume,OrderRef=OrderRef)   
        

                
    ######################## 回调事件调用 #################################  
    
    def on_order_rtn(self,pOrder):
        
        order = self.find_cache_order(pOrder.OrderLocalID)
        if order is None:
            logger.warning("order not in cache:{}".format(pOrder.OrderLocalID))
            return
        
        print(f"OnRtnOrder:"
                      f"UserID={pOrder.UserID} "
                      f"BrokerID={pOrder.BrokerID} "
                      f"InvestorID={pOrder.InvestorID} "
                      f"ExchangeID={pOrder.ExchangeID} "
                      f"InstrumentID={pOrder.InstrumentID} "
                      f"Direction={pOrder.Direction} "
                      f"CombOffsetFlag={pOrder.CombOffsetFlag} "
                      f"CombHedgeFlag={pOrder.CombHedgeFlag} "
                      f"OrderPriceType={pOrder.OrderPriceType} "
                      f"LimitPrice={pOrder.LimitPrice} "
                      f"VolumeTotalOriginal={pOrder.VolumeTotalOriginal} "
                      f"FrontID={pOrder.FrontID} "
                      f"SessionID={pOrder.SessionID} "
                      f"OrderRef={pOrder.OrderRef} "
                      f"TimeCondition={pOrder.TimeCondition} "
                      f"GTDDate={pOrder.GTDDate} "
                      f"VolumeCondition={pOrder.VolumeCondition} "
                      f"MinVolume={pOrder.MinVolume} "
                      f"RequestID={pOrder.RequestID} "
                      f"InvestUnitID={pOrder.InvestUnitID} "
                      f"CurrencyID={pOrder.CurrencyID} "
                      f"AccountID={pOrder.AccountID} "
                      f"ClientID={pOrder.ClientID} "
                      f"IPAddress={pOrder.IPAddress} "
                      f"MacAddress={pOrder.MacAddress} "
                      f"OrderSysID={pOrder.OrderSysID} "
                      f"OrderStatus={pOrder.OrderStatus} "
                      f"StatusMsg={pOrder.StatusMsg} "
                      f"VolumeTotal={pOrder.VolumeTotal} "
                      f"VolumeTraded={pOrder.VolumeTraded} "
                      f"OrderSubmitStatus={pOrder.OrderSubmitStatus} "
                      f"TradingDay={pOrder.TradingDay} "
                      f"InsertDate={pOrder.InsertDate} "
                      f"InsertTime={pOrder.InsertTime} "
                      f"UpdateTime={pOrder.UpdateTime} "
                      f"CancelTime={pOrder.CancelTime} "
                      f"UserProductInfo={pOrder.UserProductInfo} "
                      f"ActiveUserID={pOrder.ActiveUserID} "
                      f"BrokerOrderSeq={pOrder.BrokerOrderSeq} "
                      f"TraderID={pOrder.TraderID} "
                      f"ClientID={pOrder.ClientID} "
                      f"ParticipantID={pOrder.ParticipantID} "
                      f"OrderLocalID={pOrder.OrderLocalID} "
                      )        
        
    def on_order_confirm(self,pInputOrder):
        """成单事件"""
        
        logger.info(f"OnRspOrderInsert:"
                          f"UserID={pInputOrder.UserID} "
                          f"BrokerID={pInputOrder.BrokerID} "
                          f"InvestorID={pInputOrder.InvestorID} "
                          f"ExchangeID={pInputOrder.ExchangeID} "
                          f"InstrumentID={pInputOrder.InstrumentID} "
                          f"Direction={pInputOrder.Direction} "
                          f"CombOffsetFlag={pInputOrder.CombOffsetFlag} "
                          f"CombHedgeFlag={pInputOrder.CombHedgeFlag} "
                          f"OrderPriceType={pInputOrder.OrderPriceType} "
                          f"LimitPrice={pInputOrder.LimitPrice} "
                          f"VolumeTotalOriginal={pInputOrder.VolumeTotalOriginal} "
                          f"OrderRef={pInputOrder.OrderRef} "
                          f"TimeCondition={pInputOrder.TimeCondition} "
                          f"GTDDate={pInputOrder.GTDDate} "
                          f"VolumeCondition={pInputOrder.VolumeCondition} "
                          f"MinVolume={pInputOrder.MinVolume} "
                          f"RequestID={pInputOrder.RequestID} "
                          f"InvestUnitID={pInputOrder.InvestUnitID} "
                          f"CurrencyID={pInputOrder.CurrencyID} "
                          f"AccountID={pInputOrder.AccountID} "
                          f"ClientID={pInputOrder.ClientID} "
                          f"IPAddress={pInputOrder.IPAddress} "
                          f"MacAddress={pInputOrder.MacAddress} "
                          )    
        
        # 衔接RQ的后续处理，记录成单信息    
        env = Environment.get_instance()
        cur_price = pInputOrder.LimitPrice
        # 通过引用订单编号找到上下文对应的订单
        order = self.find_cache_order(pInputOrder.OrderRef)
        try:     
            trade = Trade.__from_create__(
                order_id=order.order_id,
                price=cur_price,
                amount=order.quantity,
                side=order.side,
                position_effect=order.position_effect,
                order_book_id=order.order_book_id,
                # 冻结价格取当前成交价格
                frozen_price=cur_price,
                # 当日可平仓位取0
                close_today_amount=0
            )
        except RuntimeError as e:
            logger.error("trade create err,order.order_book_id:{},price:{},amount:{},error:{}".format(order.order_book_id,cur_price,order.quantity,e))
            return
        order.fill(trade)       
        # 手续费
        total_turnover = order.quantity * order.price
        trade._commission = total_turnover * 0.03/100
        # 印花税
        trade._tax = total_turnover * 0.01/100      
        account = env.get_account(order.order_book_id)  
        # 修改撮合系统的订单状态
        self.update_order_status(order.order_book_id,RQ_ORDER_STATUS.FILLED)                       
        # 发送成单事件          
        env.event_bus.publish_event(Event(EVENT.TRADE, account=account, trade=trade, order=order))  
                  
    def on_order_failed(self,order):
        """订单失败事件"""        
        
        logger.warning("on_order_failed in:{}".format(order))
    