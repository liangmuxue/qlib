import unittest

import sys, os
from pathlib import Path

import time
import yaml
import numpy as np
import pandas as pd

import qlib
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.log import TimeInspector
from qlib.config import C

def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)

def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))
        
class TestHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "freq": "day",
                "config": self.get_feature_config(),
                "swap_level": False,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = ["Ref($close, -1)","Ref($open, 1)", "Ref($close, 1)", "Ref($volume, 1)", "$open", "$close", "$volume"]
        names = ["close-1","open+1", "close+1", "volume+1", "open", "close", "volume"]
        # fields = [ "$close", "$volume"]
        # names = ["clsose_1", "volume_1"]
        return fields, names


class TestHandlerStorage(TestAutoData):

    market = "test_one"

    start_time = "2008-05-21"
    end_time = "2008-05-28"
    train_end_time = "2008-05-21"
    test_start_time = "2008-05-28"

    data_handler_kwargs = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": market,
    }

    def test_handler_storage(self):
        uri_folder="mlruns"
        config_path = "custom/config/workflow_config_tft.yaml"
        
        with open(config_path) as fp:
            config = yaml.safe_load(fp)
    
        # config the `sys` section
        sys_config(config, config_path)
    
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)      
          
        # init data handler
        data_handler = TestHandler(**self.data_handler_kwargs)

        # init data handler with hasing storage
        data_handler_hs = TestHandler(**self.data_handler_kwargs, infer_processors=["HashStockFormat"])

        fetch_start_time = "2008-05-21"
        fetch_end_time = "2008-05-28"
        instruments = D.instruments(market=self.market)
        instruments = D.list_instruments(
            instruments=instruments, start_time=fetch_start_time, end_time=fetch_end_time, as_list=True
        )

        fetch_stock = instruments[0]
        data = data_handler.fetch(selector=(fetch_stock, slice(fetch_start_time, fetch_end_time)), level=None)
        pd.set_option('display.max_columns', None)
        print(data)


if __name__ == "__main__":
    unittest.main()
