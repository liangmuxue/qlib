#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
import warnings
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Union, List, Optional

from qlib.utils.exceptions import LoadObjectError
from qlib.contrib.evaluate import risk_analysis, indicator_analysis

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.backtest import backtest as normal_backtest
from qlib.log import get_module_logger
from qlib.utils import flatten_dict, class_casting
from qlib.utils.time import Freq
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return, calc_long_short_prec


logger = get_module_logger("workflow", logging.INFO)


class DataImportRecorder:


    artifact_path = None
    depend_cls = None  # the depend class of the record; the record will depend on the results generated by `depend_cls`

    def __init__(self, recorder):
        self._recorder = recorder
        
    @classmethod
    def get_path(cls, path=None):
        names = []
        if cls.artifact_path is not None:
            names.append(cls.artifact_path)

        if path is not None:
            names.append(path)

        return "/".join(names)

    def save(self, **kwargs):

        art_path = self.get_path()
        if art_path == "":
            art_path = None
        self.recorder.save_objects(artifact_path=art_path, **kwargs)


    @property
    def recorder(self):
        if self._recorder is None:
            raise ValueError("This RecordTemp did not set recorder yet.")
        return self._recorder

    def generate(self, **kwargs):
        """生成预测记录，用于后续回测"""
        
        # 返回值包括预测列表，以及原数据列表，列表中对象为TimeSeries类别
        pred_list,val_list = self.model.predict(self.dataset)
        # 预测数据进行加工，生成复合数据进行保存
        # pred_label_df_list = self.dataset.align_pred_and_label(pred_list,val_list)
        # self.save(**{"pred_label.pkl": pred_label_df_list})
        
        pred_save_path = "pred.pkl"
        label_save_path = "label.pkl"
        # self.save(**{pred_save_path: pred_list})
        # self.save(**{label_save_path: val_list})
        print("pkl save ok")

    def load(self, name: str, parents: bool = True):
        try:
            return self.recorder.load_object(self.get_path(name))
        except LoadObjectError:
            if parents:
                if self.depend_cls is not None:
                    with class_casting(self, self.depend_cls):
                        return self.load(name, parents=True)

    def list(self):
        return []

    def check(self, include_self: bool = False, parents: bool = True):
        if include_self:

            # Some mlflow backend will not list the directly recursively.
            # So we force to the directly
            artifacts = {}

            def _get_arts(dirn):
                if dirn not in artifacts:
                    artifacts[dirn] = self.recorder.list_artifacts(dirn)
                return artifacts[dirn]

            for item in self.list():
                ps = self.get_path(item).split("/")
                dirn, fn = "/".join(ps[:-1]), ps[-1]
                if self.get_path(item) not in _get_arts(dirn):
                    raise FileNotFoundError
        if parents:
            if self.depend_cls is not None:
                with class_casting(self, self.depend_cls):
                    self.check(include_self=True)



