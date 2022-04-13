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
            # 添加月份编号
            macro_china_qyspjg_df["month"] = macro_china_qyspjg_df['月份'].astype(str).str.slice(0,7)
            # 修改字段
            macro_china_qyspjg_df.rename(columns={'总指数-指数值':'qyspjg_total', '总指数-同比增长':'qyspjg_yoy', '总指数-环比增长':'qyspjg_mom'}, inplace = True)
            macro_china_qyspjg_df = macro_china_qyspjg_df[["month","qyspjg_total","qyspjg_yoy","qyspjg_mom"]]
            return macro_china_qyspjg_df
        
if __name__ == "__main__":    
    extractor = StockDataExtractor()       
    extractor.download_data(file_type="qyspjg")
        
