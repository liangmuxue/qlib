class IFakeSyncCall(object):
    def __init__(self):
        super(IFakeSyncCall, self).__init__()
        self.generators = {}
  
    @staticmethod
    def FAKE_SYNCALL():
        def fwrap(method):
            def fakeSyncCall(instance, *args, **kwargs):
                instance.generators[method.__name__] = method(instance, *args, **kwargs)
                func, args = instance.generators[method.__name__].__next__()
                func(*args)
            return fakeSyncCall
        return fwrap
  
    def onFakeSyncCall(self, identify, result):
        try:
            func, args  = self.generators[identify].send(result)
            func(*args)
        except StopIteration:
            self.generators.pop(identify)