import pandas as pd
import pickle

def mean_data_compare():
    df_data_path = "custom/data/darts/dump_data" + "/df_all.pkl"
    with open(df_data_path, "rb") as fin:
        df_ref = pickle.load(fin)    
    
    # df_ref.set_index(df_ref["time_idx"],inplace=True)  
    
    df_data =  df_ref[(df_ref["time_idx"]>=2551)&
                      (df_ref["time_idx"]<2568)&(df_ref["instrument"]==600031)]  
    # print("values:{}".format(df_data["label_ori"].values.tolist()))
    # print("mean value:{}".format(df_data["label"]))
    df_data["mean_label"] = df_data["label_ori"].shift(5).rolling(window=5,min_periods=1).mean()
    print("cur mean value:{}".format(df_data[["label_ori","label","mean_label"]]))

   
if __name__ == "__main__":
    mean_data_compare()