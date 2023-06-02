
import asyncio
from pyppeteer import launch
from pyquery import PyQuery as pq 

import requests

async def main():
    browser = await launch()
    page = await browser.newPage()
    await page.goto('http://quotes.toscrape.com/js/')
    doc = pq(await page.content())
    print('Quotes:', doc('.quote').length)
    await browser.close()


def test_Pyppeteer():
    asyncio.get_event_loop().run_until_complete(main())

def test_east_fund_query():
    session = requests.Session()
    url = "https://jywg.18.cn/Com/queryAssetAndPositionV1?validatekey=c1bb5b40-d40c-4c6d-a634-ae549b5a8580"
    headers = {
        "User-Agent" : "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.1.6) ",
        "Accept" : "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language" : "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,da;q=0.5",
        "Connection" : "keep-alive"
    }    
    cookie = "st_si=65591614796329; Hm_lvt_d7c7037093938390bc160fc28becc542=1685425226; Yybdm=1105; Uid=E81BJcVMRT24SeddAK+xLQ==; Khmc=lmx; eastmoney_txzq_zjzh=MTEwNTAwMDEzNDg5fA==; st_asi=delete; mobileimei=73f967fb-d161-4149-9def-43173398ba58; Uuid=11a5598aeaa84e52bea5b787e239f1b0; st_pvi=43809534314854; st_sp=2023-05-29 14:53:29; st_inirUrl=https://emt.18.cn/uniAcct-signup; st_sn=11; st_psi=20230530164103498-11923323340385-2037952858; Hm_lpvt_d7c7037093938390bc160fc28becc542=1685436064"
    cookie_dict = {i.split("=")[0]:i.split("=")[-1] for i in cookie.split("; ")}
    params = {
        "moneyType": "RMB",
        # "validatekey": "c1bb5b40-d40c-4c6d-a634-ae549b5a8580",
    }
    
    r = session.post(url, headers=headers,cookies=cookie_dict,params=params)    
    data_text = r.text
    print(data_text)

def test_east_order_submit():
    session = requests.Session()
    url = "https://jywg.18.cn/Trade/SubmitTradeV2?validatekey=c1bb5b40-d40c-4c6d-a634-ae549b5a8580"
    headers = {
        "User-Agent" : "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.1.6) ",
        "Accept" : "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language" : "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,da;q=0.5",
        "Connection" : "keep-alive"
    }    
    cookie = "st_si=65591614796329; Hm_lvt_d7c7037093938390bc160fc28becc542=1685425226; Yybdm=1105; Uid=E81BJcVMRT24SeddAK+xLQ==; Khmc=lmx; eastmoney_txzq_zjzh=MTEwNTAwMDEzNDg5fA==; st_asi=delete; mobileimei=73f967fb-d161-4149-9def-43173398ba58; Uuid=11a5598aeaa84e52bea5b787e239f1b0; st_pvi=43809534314854; st_sp=2023-05-29 14:53:29; st_inirUrl=https://emt.18.cn/uniAcct-signup; st_sn=11; st_psi=20230530164103498-11923323340385-2037952858; Hm_lpvt_d7c7037093938390bc160fc28becc542=1685436064"
    cookie_dict = {i.split("=")[0]:i.split("=")[-1] for i in cookie.split("; ")}
    params = {
        "stockCode": "000012",
        "price": "6.16",
        "amount": "100",
        "tradeType": "B",
        "zqmc": "南玻A",
        "market": "SA",                
    }
    
    r = session.post(url, headers=headers,cookies=cookie_dict,params=params)    
    data_text = r.text
    print(data_text)
  
def regist():
    session = requests.Session()
    url = "https://emt.18.cn/api/uniAcct/personRegister"
    headers = {
        "User-Agent" : "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9.1.6) ",
        "Accept" : "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language" : "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,da;q=0.5",
        "Connection" : "keep-alive"
    }    
    cookie = "st_si=65591614796329; Hm_lvt_d7c7037093938390bc160fc28becc542=1685343210; st_asi=delete; st_pvi=43809534314854; st_sp=2023-05-29 14:53:29; st_inirUrl=https://emt.18.cn/uniAcct-signup; st_sn=14; st_psi=20230531094435944-1134234315725-0483187242; XJSESSIONID=01c557d6-708d-4ad2-bbe7-93424939c60a; Hm_lpvt_d7c7037093938390bc160fc28becc542=1685497476"
    cookie_dict = {i.split("=")[0]:i.split("=")[-1] for i in cookie.split("; ")}
    params = {
        "email": "PQloKPQQqWdeQ6f2AxAmXtXa2bI003V5VtgwkQYGfy2Ar9XhkU8hgdt21PHku6Qjz1um7so5+Ac5ZuQ+HJ6c92mjwpef0PSe7OomGuosINjXyz9/WpwUSFOUrsgg2QHrL7dXmm69DXZPpiJ52Wp//iYQhzrugK7ZyWBNFmZHjPQ3k1F7gcFeK8y1/kfUoUh2IPS4YHYWzol7JwDHyBUCnV+uAiT75Cdyqy3LgINLeTYSWXTLEp6n43VckLsqJPIQq6QVg2cfrJ4Al14Stlwqbqr1soi8a4RfKVwTb2AlPgWaJ9j/cJE1l/kj/9qSalMpf6vmHBSyOqNkNnYY3U082w==",
        "funcId": "uniAcct/personRegister",
        "fundId": "110500013489",
        "password": "rikQ0nYPhTJQJwixdEfocx1PDJ7vE4a1+/RFAG/jaJAEQkSf6gXSkMMavAc+2XXwSLd3CxNWLKj4I1+mg5j5CegHzYnYRAWrLheVtYXMbboz6ZIfvl6vgVinlPWuHChsSOSQ0+X71YWkjrMjJB0Ga1jPUscvgQSBj+Wrk21ZkR95yDJD06CkSuOUI4LAQVMgrHS0Fs2WPVkrf8c53nLaUFOG8ThB5Qw1kGbH5spBqfQfZxmZzPad1Z0R3RPPdQQbCqs1d71Pv3oa8rkaRSt/tWxUW5+h7kc2dnTCh0F75jlUgeO589l4bWNj2nPalrevekAXqPVwA41IPoLeIrbsIg==",
        "phone": "oyDoErQ6x7EJT7CSiI0l8eCrl1BnXcgiKNdGcuPcx0uih7AKcAahSNLV9wi3id8dP6rR+75TsLpjykNpB00GUAQ1GsPKt/CKa07CMk3saTt1/DN2KsEPSmu3hwrB8rYnU2aSZ1WXYfxk9xa9rW9y3PC9P12558E0TP3c7E4a8rIrZphRj4oXyPfTnjXC+sQO/+oaRZHUy1BKSsmjgoRhaIdLRpMB3zxdSr4u33KgocZdFaEggO32+dmWW18JnVC7roldn1Nwue3rAUIlDGq3/zIZhJE6nm3rplKqZUHQra9QfhjuM5SlV6RTfrmg1qkrBbzCFodqoaOxtf6natTE+w==",
        "pwdConfirm": "rikQ0nYPhTJQJwixdEfocx1PDJ7vE4a1+/RFAG/jaJAEQkSf6gXSkMMavAc+2XXwSLd3CxNWLKj4I1+mg5j5CegHzYnYRAWrLheVtYXMbboz6ZIfvl6vgVinlPWuHChsSOSQ0+X71YWkjrMjJB0Ga1jPUscvgQSBj+Wrk21ZkR95yDJD06CkSuOUI4LAQVMgrHS0Fs2WPVkrf8c53nLaUFOG8ThB5Qw1kGbH5spBqfQfZxmZzPad1Z0R3RPPdQQbCqs1d71Pv3oa8rkaRSt/tWxUW5+h7kc2dnTCh0F75jlUgeO589l4bWNj2nPalrevekAXqPVwA41IPoLeIrbsIg==",
        "requestId": "c04994d8-6eb1-4fb5-98d2-e46b6d26c7cf",
        "userCode": "admin",
        "userName": "梁慕学"
    }
    
    r = session.post(url, headers=headers,cookies=cookie_dict,params=params)    
    data_text = r.text
    print(data_text)    
  
if __name__ == "__main__": 
    # test_Pyppeteer()
    # test_east_fund_query()
    # test_east_order_submit()
    regist()
    
    