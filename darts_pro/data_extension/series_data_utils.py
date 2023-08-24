import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
import xarray as xr

import joblib
from sklearn.metrics import accuracy_score
import sklearn.neighbors
from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax, \
    TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, \
    KNeighborsTimeSeries
    
from tft.class_define import CLASS_LAST_VALUES,get_simple_class,get_complex_class
from cus_utils.tensor_viz import TensorViz
from cus_utils.log_util import AppLogger

logger = AppLogger()

def get_pred_center_value(series):
    """取得区间范围的中位数数值"""
    
    comp = series.data_array().sel(component="label")
    central_series = comp.quantile(q=0.5, dim="sample")
    return central_series

def get_np_center_value(np_data):
    """取得区间范围的中位数数值"""
    
    m = np.mean(np_data,axis=1)
    return m

def build_serices_with_ndarray(np_array):
    """根据numpy数组数据，生成时间序列对象"""
    
    times = [i for i in range(np_array.shape[0])]
    sample = [i for i in range(np_array.shape[1])]
    arr = xr.DataArray(np_array, coords=[times, sample], dims=['time_index', 'component', 'sample'],name="label")

    series = TimeSeries.from_xarray(arr)   
    return series 


class StatDataAssis():
    def __init__(self,filepath="custom/data/asis"):
        # self.df_all = df_all
        self.df_target_train = None
        self.df_target_valid = None
        self.filepath = filepath
        self.target_data_train = None
        self.target_data_valid = None
        self.target_class_data_train = None
        self.target_class_data_valid = None
        self.output_data_train = None
        self.output_data_valid = None
                
    def build_filter_target_data(self,data_batch,batch_idx,type="train"):
        
        # df_all = self.df_all
        if type=="train":
            df_target_all = self.df_target_train
            target_data = self.target_data_train
            target_class_data = self.target_class_data_train 
        else:
            df_target_all = self.df_target_valid
            target_data = self.target_data_valid
            target_class_data = self.target_class_data_valid
                        
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scalers,target_class,target,target_info) = data_batch
        target = [scaler.inverse_transform(target[index,:,:].cpu().numpy()) for index,scaler in enumerate(scalers)]
        price_target = [ts["price_array"][-5:] for ts in target_info]
        target = np.stack(target)
        price_target = np.expand_dims(np.stack(price_target),axis=-1)
        target = np.concatenate((target,price_target),axis=-1)
        target_class = target_class.cpu().numpy()
        # for ts in target_info:
            # df_target = df_all[(df_all["time_idx"]>=ts["start"])&(df_all["time_idx"]<ts["end"])&
            #                         (df_all["instrument_rank"]==ts["item_rank_code"])]          
            # if self.df_target_all is None:
            #     df_target_all = df_target
            # else:
            #     df_target_all = pd.concat([df_target_all,df_target])
        if target_data is None:
            target_data = target
            target_class_data = target_class
        else:
            target_data = np.concatenate((target_data,target),axis=0)
            target_class_data = np.concatenate((target_class_data,target_class),axis=0)

        if type=="train":
            self.target_data_train = target_data
            self.target_class_data_train = target_class_data
        else:
            self.target_data_valid = target_data
            self.target_class_data_valid = target_class_data

    def build_output_data(self,output,batch_idx,type="train"):
        
        # df_all = self.df_all
        if type=="train":
            target_data = self.output_data_train
        else:
            target_data = self.output_data_valid
        
        # output = [output_item[:,:,0,0].detach().cpu().numpy() for output_item in output]
        # output = np.stack(output,axis=-1)
        if target_data is None:
            target_data = output
        else:
            target_data = np.concatenate((target_data,output),axis=0)

        if type=="train":
            self.output_data_train = target_data
        else:
            self.output_data_valid = target_data
                                
    def finish_build_target(self,type="train"):    
        
        if type=="train":
            target_data = self.target_data_train
            target_class_data = self.target_class_data_train             
        else:
            target_data = self.target_data_valid
            target_class_data = self.target_class_data_valid            
        
        data_filepath = "{}/data_{}.npy".format(self.filepath,type)
        class_filepath = "{}/class_{}.npy".format(self.filepath,type)
        np.save(data_filepath,target_data)
        np.save(class_filepath,target_class_data)
        
    def finish_build_output(self,type="train"):   
        if type=="train":
            output_data = self.output_data_train   
        else:
            output_data = self.output_data_valid 
        
        output_filepath = "{}/output_{}.npy".format(self.filepath,type)
        np.save(output_filepath,output_data)
                    
    def view_data(self):
        viz = TensorViz(env="stat_knn_data")
        
        X = np.load("custom/data/asis/data_train.npy")[:,:,:2]
        y = np.load("custom/data/asis/class_train.npy")[:,0,0]   
        import_index = np.where(y==3)[0] 
        for i in range(10):  
            idx = import_index[i]
            view_data = X[idx]
            label = y[idx].item()
            # view_data = view_data.transpose(1,0)
            win = "ana_data_win_{}".format(i)
            names = ["label","obv_tar"]
            title = "class_{}".format(label)
            viz.viz_matrix_var(view_data,win=win,title=title,names=names)   

    def analysis_data(self):
        X = np.load("custom/data/asis/data_train.npy")
        y = np.load("custom/data/asis/class_train.npy")[:,0,0]
        # price_array = X[:,:,-1]
        # raise_range = (price_array[:,-1] - price_array[:,0])/price_array[:,0]*100
        # y_real = [get_simple_class(rr) for rr in raise_range]
        train_end = 1000
        X = X[:train_end,:,:]
        y = y[:train_end]
        # y_test = y[train_end:test_start]
        # predicted_labels,X_test,y_test = self.predict_data(X,y)
        predicted_labels,X_test,y_test = self.fit_target_data(X,y)
        # self.fit_pred_data_batch(X,y)
        predicted_labels_imp_index = np.where(predicted_labels==3)[0]
        acc_cnt = np.sum(y_test[predicted_labels_imp_index]==3)
        acc_rate = acc_cnt/predicted_labels_imp_index.shape[0]
        total_imp_acc = np.sum(y_test==3)
        logger.debug("Correct classification rate:{}".format(accuracy_score(y_test, predicted_labels)))    
        logger.debug("important Correct classification cnt:{}, rate:{},total_cnt:{}".format(acc_cnt,acc_rate,total_imp_acc))  
        viz = TensorViz(env="stat_pred_data")
        names = ["label","obv","price"]
        for i in range(10):  
            rand_index = np.random.randint(0,predicted_labels_imp_index.shape[0]-1)
            idx = predicted_labels_imp_index[rand_index]
            view_data = np.concatenate((X_test[idx,:,:2],X_test[idx,:,-1:]),axis=-1)
            label = y_test[idx]
            pred = int(predicted_labels[idx])
            # view_data = view_data.transpose(1,0)
            win = "ana_pred_win_{}".format(i)
            title = "class_{},pred_{}".format(label,pred)
            viz.viz_matrix_var(view_data,win=win,title=title,names=names)          

    def analysis_output_data(self):
        X = np.load("custom/data/asis/data_train.npy")
        y = np.load("custom/data/asis/class_train.npy")[:,0,0]
        output = np.load("custom/data/asis/output_train.npy")
        
        # price_array = X[:,:,-1]
        # raise_range = (price_array[:,-1] - price_array[:,0])/price_array[:,0]*100
        # y_real = [get_simple_class(rr) for rr in raise_range]
        train_end = 300
        X = X[:train_end,:,:]
        y = y[:train_end]
        output = output[:train_end,:,:2]
        # output = X[:,:,:2]
        # y_test = y[train_end:test_start]
        # predicted_labels,X_test,y_test = self.predict_data(X,y)
        predicted_labels = self.fit_pred_data(output)
        # self.fit_pred_data_batch(X,y)
        predicted_labels_imp_index = np.where(predicted_labels==3)[0]
        acc_cnt = np.sum(y[predicted_labels_imp_index]==3)
        acc_rate = acc_cnt/predicted_labels_imp_index.shape[0]
        total_imp_acc = np.sum(y==3)
        logger.debug("Correct classification rate:{}".format(accuracy_score(y, predicted_labels)))    
        logger.debug("important Correct classification cnt:{}, rate:{},total_cnt:{}".format(acc_cnt,acc_rate,total_imp_acc))  
        viz = TensorViz(env="stat_pred_data")
        names = ["label","obv","price","pred_label","pred_obv"]
        for i in range(10):  
            rand_index = np.random.randint(0,predicted_labels_imp_index.shape[0]-1)
            idx = predicted_labels_imp_index[rand_index]
            output_item = output[idx]
            view_data = np.concatenate((X[idx,:,:2],X[idx,:,-1:]),axis=-1)
            view_data = np.concatenate((view_data,output_item[:,:2]),axis=-1)
            label = y[idx]
            pred = int(predicted_labels[idx])
            # view_data = view_data.transpose(1,0)
            win = "ana_pred_win_{}".format(i)
            title = "class_{},pred_{}".format(label,pred)
            viz.viz_matrix_var(view_data,win=win,title=title,names=names) 
                
    def fit_target_data(self,X,y,save_weight=True):
        scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))
        X_scaled = scaler.fit_transform(X[:,:,:2])       
        train_end = X_scaled.shape[0]//3*2
        X_train = X_scaled[:train_end,:,:]
        X_test = X_scaled[train_end:,:,:]
        X_test_ori = X[train_end:,:,:]
        y_train = y[:train_end]
        y_test = y[train_end:]
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=4, metric="softdtw")
        knn_clf.fit(X_train, y_train)
        predicted_labels = knn_clf.predict(X_test)    
        if save_weight:
            model_path = "{}/knn_clf.model".format(self.filepath)
            joblib.dump(knn_clf, model_path)        
        return predicted_labels,X_test_ori,y_test

    def fit_pred_data(self,output):
        model_path = "{}/knn_clf.model".format(self.filepath)
        knn_clf = joblib.load(model_path)
        logger.debug("begin pred")
        predicted_labels = knn_clf.predict(output)    
        return predicted_labels
    
    def fit_pred_data_batch(self,X,y):
        scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))
        X_scaled = scaler.fit_transform(X[:,:,:2])      
        knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=4, metric="softdtw")
        batch_size = 1000
        epoch_num = X_scaled.shape[0]//batch_size - 1
        logger.debug("begin fit")
        for epoch in range(epoch_num):
            start = batch_size*epoch
            end = batch_size*epoch + batch_size
            X_train = X_scaled[start:end,:,:]
            y_train = y[start:end]
            knn_clf.partial_fit(X_train, y_train)
            logger.debug("fit epoch:{}".format(epoch))
        
        model_path = "{}/knn_clf.model".format(self.filepath)
        joblib.dump(knn_clf, model_path)
        logger.debug("fit_pred_data_batch ok")
        
    def predict_data(self,X,y):
        predicted_labels = np.ones(X.shape[0])
        data = X[:,:,0]
        raise_range = (data[:,-1] - data[:,0])/data[:,0]*100
        for i in range(4):
            if i==0:
                condition = (raise_range<=-3)
            if i==1:
                condition = (raise_range>-3)&(raise_range<=0)
            if i==2:
                condition = (raise_range>0)&(raise_range<=3)  
            if i==3:
                condition = (raise_range>3)                             
            index = index = np.where(condition)[0]
            predicted_labels[index] = i
        return predicted_labels,X,y

    
                          
if __name__ == "__main__":       
    data_assis = StatDataAssis()
    # data_assis.view_data()
    # data_assis.analysis_data()
    data_assis.analysis_output_data()
        