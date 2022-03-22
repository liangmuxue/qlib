# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import requests
import datetime
from tqdm import tqdm
from pathlib import Path
from loguru import logger
import pandas as pd
import akshare as ak

class StockDataExtractor:
    """证券数据采集"""
    
    DATASET_VERSION = "v2"
    REMOTE_URL = "http://fintech.msra.cn/stock_data/downloads"
    QLIB_DATA_NAME = "{dataset_name}_{region}_{interval}_{qlib_version}.zip"

    def __init__(self, delete_zip_file=False):
        """

        Parameters
        ----------
        delete_zip_file : bool, optional
            Whether to delete the zip file, value from True or False, by default False
        """
        self.delete_zip_file = delete_zip_file
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.save_path = root_path + "/../data/macro" 
        self.macro_data_dict = ["qyspjg"]

    def normalize_dataset_version(self, dataset_version: str = None):
        if dataset_version is None:
            dataset_version = self.DATASET_VERSION
        return dataset_version

    def merge_remote_url(self, file_name: str, dataset_version: str = None):
        return f"{self.REMOTE_URL}/{self.normalize_dataset_version(dataset_version)}/{file_name}"

    def download_data(self, file_type: str):
        # 使用akshare获取宏观数据，并保存
        if file_type=="qyspjg":
            macro_china_qyspjg_df = ak.macro_china_qyspjg()
            save_path = self.save_path + "/qyspjg.pickle"
            macro_china_qyspjg_df.to_pickle(save_path)
        
    def load_data(self,file_type):
        if file_type=="qyspjg":
            path = self.save_path + "/qyspjg.pickle"
            macro_china_qyspjg_df = pd.read_pickle(path)
            return macro_china_qyspjg_df
        
if __name__ == "__main__":    
    extractor = StockDataExtractor()       
    extractor.download_data(file_type="qyspjg")
        
