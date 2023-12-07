from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
from cus_utils.data_filter import DataFilter
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def test_scaler():        
    source_data = np.load("custom/data/aug/tests/scaler.npy")
    ori_shape = source_data.shape
    source_data = np.expand_dims(source_data.reshape(-1),axis=-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    target = scaler.fit_transform(source_data)  
    target = target.reshape(ori_shape)
    print(target)

def build_air_numpy_data():        
    data_filter = DataFilter()
    series = AirPassengersDataset().load()
    df = series.pd_dataframe()
    df.reset_index(inplace=True)
    df["Year"] = df.Month.dt.strftime('%Y').astype(int)
    df["Month"] = df.Month.dt.strftime('%m').astype(int)
    months = df["Year"] * 12 + df["Month"]
    df["time_idx"] = months - min(months) + 1
    df = df[["time_idx","Year","Month","#Passengers"]]
    wave_data = data_filter.filter_wave_data(df, target_column="#Passengers", group_column=None,forecast_horizon=1,wave_period=16,
                                                 wave_threhold_type="more",wave_threhold=0,over_time=1) 
    np.save("custom/data/aug/tests/air.npy",wave_data)   

def test_tsaug():
    import tsaug
    from tsaug.visualization import plot
    X = np.array([[[1,2],[2,3],[3,4]]])
    Y = np.array([[1,2,3]])
    X_aug, Y_aug = tsaug.AddNoise(scale=0.01).augment(X, Y)
    X_aug, Y_aug = tsaug.Quantize(n_levels=10).augment(X, Y)
    X_aug, Y_aug = tsaug.Pool(size=2).augment(X, Y)
    plot(X_aug, Y_aug);    
   
if __name__ == "__main__":
    # test_scaler()
    test_tsaug()
    # build_air_numpy_data()

    
    