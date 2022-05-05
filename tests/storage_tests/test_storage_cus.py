# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from pathlib import Path
from collections.abc import Iterable

import numpy as np
from qlib.tests import TestAutoData

from qlib.data.storage.file_storage import (
    FileCalendarStorage as CalendarStorage,
    FileInstrumentStorage as InstrumentStorage,
    FileFeatureStorage as FeatureStorage,
)

_file_name = Path(__file__).name.split(".")[0]
DATA_DIR = Path(__file__).parent.joinpath(f"{_file_name}_data")
QLIB_DIR = DATA_DIR.joinpath("qlib")
QLIB_DIR.mkdir(exist_ok=True, parents=True)

import unittest

class TestStorage(TestAutoData):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.provider_uri = "/home/qdata/qlib_data/custom_cn_data"  # target_dir
        self.provider_uri_1day = "/home/qdata/qlib_data/custom_cn_data"  # target_dir
        self.provider_uri_1min = "~/.qlib/qlib_data/cn_data_1min"

    def test_feature_storage(self):
        """
        Calendar:
            pd.date_range(start="2005-01-01", stop="2005-03-01", freq="1D")

        Instrument:
            {
                "SH600000": [(2005-01-01, 2005-01-31), (2005-02-15, 2005-03-01)],
                "SH600001": [(2005-01-01, 2005-03-01)],
                "SH600002": [(2005-01-01, 2005-02-14)],
                "SH600003": [(2005-02-01, 2005-03-01)],
            }

        Feature:
            Stock data(close):
                            2005-01-01  ...   2005-02-01   ...   2005-02-14  2005-02-15  ...  2005-03-01
                SH600000     1          ...      3         ...      4           5               6
                SH600001     1          ...      4         ...      5           6               7
                SH600002     1          ...      5         ...      6           nan             nan
                SH600003     nan        ...      1         ...      2           3               4

            FeatureStorage(SH600000, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 6)
                ]

                ====> [(0, 1), ..., (59, 6)]


            FeatureStorage(SH600002, close):

                [
                    (calendar.index("2005-01-01"), 1),
                    ...,
                    (calendar.index("2005-02-14"), 6)
                ]

                ===> [(0, 1), ..., (44, 6)]

            FeatureStorage(SH600003, close):

                [
                    (calendar.index("2005-02-01"), 1),
                    ...,
                    (calendar.index("2005-03-01"), 4)
                ]

                ===> [(31, 1), ..., (59, 4)]

        """

        feature = FeatureStorage(instrument="600520", field="close", freq="day", start="2008-01-02", stop="2008-01-08",provider_uri=self.provider_uri)


        print(f"feature[0: 1]: \n{feature[0: 1]}")

            
            
if __name__ == "__main__":
    # tsj = TestStorage()
    # tsj.test_feature_storage()     
    unittest.main()      
