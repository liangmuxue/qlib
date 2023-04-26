from cus_utils.process import IFakeSyncCall

from rqalpha.const import SIDE

class BaseTrade(IFakeSyncCall):
    
    def __init__(
        self,
        context,
        **kwargs,
    ):   
        super(BaseTrade, self).__init__()
        
        self.context = context
        self.kwargs = kwargs
        
        
    def get_portfolio(self):
        pass
    
    def submit_order(self):
        pass