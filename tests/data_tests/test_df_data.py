import pandas as pd
from datetime import date
import sys, os
import numpy as np
import cv2
import pickle

def test_instrument_data():
    
    instrument = 603997
    df_data_path = "/home/qdata/workflow/wf_backtest_flow/task/20/dump_data/df_all.pkl"
    with open(df_data_path, "rb") as fin:
        df_ref = pickle.load(fin)          
        df_ref_item = df_ref[df_ref["instrument"]==instrument]
        print("df_ref_item shape:{}".format(df_ref_item.shape))
        print("df_ref_item time idx:{}".format(df_ref_item.time_idx))

if __name__ == "__main__":
    test_instrument_data()

    
    