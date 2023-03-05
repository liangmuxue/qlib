from rqalpha.interface import AbstractMod

from .tdx_ds import TdxDataSource

class ExtDataMod(AbstractMod):
    """用于加载自定义数据源"""
    
    def __init__(self):
        pass

    def start_up(self, env, mod_config):
        env.set_data_source(TdxDataSource(env.config.base.data_bundle_path))

    def tear_down(self, code, exception=None):
        return
    
def load_mod():
    return ExtDataMod()   