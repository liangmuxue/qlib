from trader.utils.date_util import tradedays
from datetime import datetime
def test_days_dur():
    trade_date = 20230508
    before_date = 20230510
    trade_date = datetime.strptime(str(trade_date),"%Y%m%d")
    before_date = datetime.strptime(str(before_date),"%Y%m%d")
    dur_days = tradedays(trade_date,before_date)
    print(dur_days)

        
if __name__ == "__main__":
    test_days_dur()