from darts.logging import get_logger, raise_if, raise_if_not, raise_log

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
            
def create_from_cls_and_kwargs(cls, kws):
    logger = get_logger(__name__)
    try:
        return cls(**kws)
    except (TypeError, ValueError) as e:
        raise_log(
            ValueError(
                "Error when building the optimizer or learning rate scheduler;"
                "please check the provided class and arguments"
                "\nclass: {}"
                "\narguments (kwargs): {}"
                "\nerror:\n{}".format(cls, kws, e)
            ),
            logger,
        )            