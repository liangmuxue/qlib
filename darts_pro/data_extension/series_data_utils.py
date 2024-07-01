import numpy as np
import pandas as pd
from darts.timeseries import TimeSeries
import xarray as xr
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score
import scipy.spatial as spt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tft.class_define import CLASS_SIMPLE_VALUE_MAX,CLASS_SIMPLE_VALUES,get_complex_class
from cus_utils.tensor_viz import TensorViz
from cus_utils.common_compute import np_qcut,pairwise_distances,batch_cov,normalization_axis
from cus_utils.log_util import AppLogger
from losses.mtl_loss import UncertaintyLoss
from projects.kmeans_pytorch import kmeans, kmeans_predict
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import DBSCAN,KMeans,SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

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

    def data_corr_analysis(self,ds_data,analysis_columns = ["PRICE_SCOPE","MASCOPE5","OBV5","RSI5","MACD"]):
        # df_expirement = df_all[["label","CLOSE"]]
        # df_expirement = df_expirement.iloc[:3000]
        # df_corr = df_expirement.corr(method="spearman")
        # print(df_corr)
        # sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
        fit_names = ds_data.fit_names
        np_target = ds_data.np_data
        # np_target = normalization_axis(np_target,axis=1)         
        # df_target = df_target.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        size = np_target.shape[0]
        df_combine = None
        fit_last_values = np.zeros((size,len(analysis_columns)-1))
        for i in range(size):
            target_item = np_target[i]
            scaler = MinMaxScaler()
            scaler.fit(target_item[:25])             
            past_target = scaler.transform(target_item[:25])   
            future_target = scaler.transform(target_item[25:])     
            # tar_data = np.concatenate([past_target,future_target],axis=0)    
            df_item = pd.DataFrame(future_target,columns=ds_data.fit_names)
            tar_data = df_item[analysis_columns]        
            fit_last_values[i] = tar_data.values[-1,1:]     
            df_corr = tar_data.corr(method="spearman").iloc[[0]]
            if df_combine is None:
                df_combine = df_corr
            else:
                df_combine = pd.concat([df_combine,df_corr])        
        print("corr value:{}".format(df_combine.mean()))
        
        # plt.savefig('./custom/data/asis/seaborn_heatmap_corr_result.png')
        
        # 衡量序列末尾数值，与价格涨幅的关系
        price_target = np_target.squeeze()
        price_range = (price_target[:,-1,0] - price_target[:,-5,0])
        for i in range(len(analysis_columns)-1):
            col = analysis_columns[i+1]
            concat = np.stack([price_range,fit_last_values[:,i]])
            corr = np.corrcoef(concat)[0,1]
            print("{}:{}".format(col,corr))
        
    def output_corr_analysis(self,ds_data,analysis_columns=None,fit_names=None,target_col=None,diff_columns=None):
        size = len(ds_data)
        df_combine = None
        df_combine_fits = None
        df_combine_diff = None
        viz = TensorViz(env="data_analysis")
        viz_fail = TensorViz(env="data_analysis_neg")
        viz_normal = TensorViz(env="data_analysis_normal")
        col_index = analysis_columns.index(target_col[0])
        index = 0
        for i in range(size):
            target_item,target_class = ds_data.__getitem__(i)
            # 合并目标数据与输出数据，以进行相关性比较
            combine_data = np.concatenate((target_item[:,col_index:col_index+1],ds_data.output_data[i]),axis=-1)
            tar_data = pd.DataFrame(combine_data,columns=analysis_columns)
            df_corr = tar_data.corr(method="spearman").iloc[[0]]
            if df_combine is None:
                df_combine = df_corr
            else:
                df_combine = pd.concat([df_combine,df_corr])
            
            # 直接在目标数据间比较
            df_item = pd.DataFrame(target_item,columns=fit_names)
            df_corr_fit = df_item.corr(method="spearman").iloc[[0]]            
            if df_combine_fits is None:
                df_combine_fits = df_corr_fit
            else:
                df_combine_fits = pd.concat([df_combine_fits,df_corr_fit])  

            # 比较差分数据
            df_diff_data = tar_data[diff_columns].diff()
            df_diff_data = pd.concat([df_item[target_col],df_diff_data],axis=1)
            df_corr_diff = df_diff_data.corr(method="spearman").iloc[[0]]
            if df_combine_diff is None:
                df_combine_diff = df_corr_diff
            else:
                df_combine_diff = pd.concat([df_combine_diff,df_corr_diff]) 
                                        
            # 合并显示预测和实际数据
            target_value = df_item[target_col].values
            # 查看涨幅或跌幅达标的数据
            tar_data_rm = tar_data[tar_data.columns[1:]]
            df_analysis = pd.concat([tar_data_rm,df_item],axis=1)            
            if target_class==3:
                viz_real = viz
            elif target_class==0:
                viz_real = viz_fail   
            else:
                viz_real = viz_normal   
            index += 1
            self.show_ana_data(df_analysis,index=index,viz=viz_real)                       
        print("corr output value:{}".format(df_combine.mean()))
        print("corr diff value:{}".format(df_combine_diff.mean()))
        print("corr target value:{}".format(df_combine_fits.mean()))       

    def output_target_viz(self,ds_data,fit_names):
        size = len(ds_data)
        viz = TensorViz(env="data_analysis_valid")
        viz_fail = TensorViz(env="data_analysis_valid_neg")
        viz_normal = TensorViz(env="data_analysis_valid_normal")
        index = 0
        tar_import = []
        tar_normal = []
        tar_neg = []
        pad_data = [0 for i in range(25)]
        pad_data = np.array([pad_data,pad_data,pad_data]).transpose(1,0)
        names = ["MACD","RANKMA5","QTLU","MACD_output","RANKMA5_output","QTLU_output"]
        names = ["RANKMA5","RANKMA5_output","QTLU","QTLU_output"]
        for i in range(200):
            target_item,target_class,output_inverse,target_info = ds_data.__getitem__(i)
            output_total = np.concatenate((pad_data,output_inverse),axis=0)  
            view_data = np.concatenate((target_item,output_total),axis=1)      
            view_data = np.stack((view_data[:,1],view_data[:,4],view_data[:,2],view_data[:,5]),axis=1)      
            target_title = "item_{} ,price class:{}".format(index,target_class.item())
            win = "index_{}".format(index)                   
            # 查看涨幅或跌幅达标的数据
            if target_class==3:
                tar_import.append(target_item)
                viz.viz_matrix_var(view_data,win=win,title=target_title,names=names)                 
            elif target_class==0:
                tar_neg.append(target_item)
                viz_fail.viz_matrix_var(view_data,win=win,title=target_title,names=names)
            else:
                tar_normal.append(target_item)
            index += 1

        # df_normal = pd.DataFrame(np.array(tar_normal),columns=fit_names)
        # self.show_ana_data(df_normal,index=index,viz=viz_normal) 
        # df_neg = pd.DataFrame(np.array(tar_neg),columns=fit_names)
        # self.show_ana_data(df_neg,index=index,viz=viz_fail)   
                
    def show_ana_data(self,df_data,index=0,viz=None):
        # if index>18:
        #     return
        title = "tar_view_{}".format(index)
        win = "win_analysis_{}".format(index)
        view_data = df_data.values
        names = df_data.columns
        viz.viz_matrix_var(view_data,win=win,title=title,names=names)    

    def batch_data_ana(self,ds):
        fit_names = ["MACD","RANKMA5","QTLU"]
        
        # 取得所有完整周期的目标数据
        whole_target_inverse,price_array,target_class = ds.build_origin_target_data()
        target_data = whole_target_inverse[:,25:,:]
        price_target_array = price_array[:,25:]
        p_target_class = target_class 
        import_price_index = np.where(p_target_class==CLASS_SIMPLE_VALUE_MAX)[0]
        print("import_price_index cnt:",import_price_index.shape[0])
        print("total cnt:",len(price_array))
        # 对每个独立指标，查看上涨占比
        for i in range(3):
            if i==0:
                measure_bool = ((target_data[:,-1,i] - target_data[:,0,i])/np.abs(target_data[:,0,i])*100)>0
            elif i==1:
                measure_bool = ((target_data[:,-1,i] - target_data[:,0,i])/np.abs(target_data[:,0,i])*100)>5
            else:
                measure_bool = ((target_data[:,-1,i] - target_data[:,0,i])/np.abs(target_data[:,0,i])*100)<0
            measure_index = np.where(measure_bool)[0]
            match_index = np.intersect1d(measure_index,import_price_index)
            print("{},match cnt:{}".format(fit_names[i],match_index.shape[0]))
            print("{}，measure cnt:{}".format(fit_names[i],measure_index.shape[0]))
            
    def clustering_output(self,ds_data):
        """对输出按照目标类别进行聚类"""
        
        loss_unity = UncertaintyLoss()
        viz = TensorViz(env="data_analysis")
        index = 0
        (past_target,past_covariates, historic_future_covariates,future_covariates,static_covariates,scaler_tuple,target_class,target,target_info) = ds_data.target_data
        target_class = target_class[:,0,0]
        output_data = ds_data.output_data
        device_str = 'cuda:0'
        device = torch.device(device_str)
        num_clusters = len(CLASS_SIMPLE_VALUES.keys())
        # 按照不同的指标分别聚类
        for i in range(output_data.shape[-1]):
            output = output_data[:,:,i]
            # pair_dis = pairwise_distances(output,metric=loss_unity.ccc_distance_torch,n_jobs=6)
            db = DBSCAN(eps=0.6, metric=loss_unity.ccc_distance_torch, min_samples=50,n_jobs=4).fit(output)  
            print("cluster_centers_{}:{},{}".format(i,db.core_sample_indices_,db.core_sample_indices_.shape))     
            # total_acc_cnt = np.sum(db.labels_==target_class)   
            # total_acc = total_acc_cnt/target_class.shape[0]
            # import_index = np.where(db.labels_==CLASS_SIMPLE_VALUE_MAX)[0]
            # import_acc_cnt = np.sum(target_class[import_index]==CLASS_SIMPLE_VALUE_MAX)
            # import_acc = import_acc_cnt/import_index.shape[0]
            # print("acc_{},total_acc:{},import_acc:{}".format(i,total_acc,import_acc)) 

    def analysis_compare_output(self,ds_data):
        """使用度量的方式，综合分析输出值与目标值"""
        
        viz = TensorViz(env="data_analysis")
        index = 0
        (past_target,target_class,target,pca_target,price_target,target_info) = ds_data.target_data
        target_class = target_class[:,0,0]
        output_data = ds_data.output_data
        (x_bar, z_pca,lattend) = output_data
        device_str = 'cuda:0'
        device_str = 'cpu'
        device = torch.device(device_str)
        loss_unity = UncertaintyLoss(device=device)
        num_clusters = len(CLASS_SIMPLE_VALUES.keys())
        import_index = np.where(target_class==3)[0]
        neg_index = np.where(target_class==0)[0]
        combine_index = np.where((target_class==3)|(target_class==0)|(target_class==1)|(target_class==2))[0]
        # combine_index = np.where((target_class==3)|(target_class==0))[0]
        labels = target_class[combine_index]
        output_pair_dis_arr = []
        # 按照不同的指标分别聚类
        for i in range(x_bar.shape[-1]):
            # if i!=2:
            #     continue
            output_item = z_pca[...,i]
            output = output_item[combine_index]
            target_single = target[...,i]
            pca_target_single = pca_target[...,i]
            target_single = target_single[combine_index]
            
            # 可视化，使用二维坐标在图形展示        
            # self.matrix_results_viz(coords=coords,labels=labels,noise_index=noise_index,name="output_pn_{}".format(i))
            self.matrix_results_viz(coords=output_item,labels=labels,noise_index=None,name="output_s_{}".format(i))
            # 对目标值分布的可视化
            self.matrix_results_viz(pca_target_single,labels=labels,name="target_pn_{}".format(i))  
              
            # target_db = DBSCAN(eps=0.2, metric='precomputed',min_samples=4,n_jobs=2).fit(target_pair_dis)  
            # self.dbscan_results(target_db,target_pair_dis,name="target")            
            # import_acc = import_acc_cnt/import_index.shape[0]
            # print("acc_{},total_acc:{},import_acc:{}".format(i,total_acc,import_acc)) 

    def analysis_compare_output_clu(self,ds_data):
        """使用度量的方式，综合分析输出值与目标值"""
        
        viz = TensorViz(env="data_analysis")
        index = 0
        (past_target,scaler,target_class,future_target,future_price_target,slope_target,target_info) = ds_data.target_data
        
        price_data = ds_data.price_data
        target = future_target[:,:,:-2,:]
        pca_target = future_target[:,:,-2:,:]
        target_class = target_class[...,0,0]
        output_data = ds_data.output_data
        x_bar_total = output_data[0].transpose(1,2,3,0)
        pred_total = output_data[1].transpose(1,2,3,0)
                
        device_str = 'cuda:0'
        device_str = 'cpu'
        device = torch.device(device_str)
        loss_unity = UncertaintyLoss(device=device)
        num_clusters = len(CLASS_SIMPLE_VALUES.keys())
        
        import_index = np.where(target_class==3)[0]
        neg_index = np.where(target_class==0)[0]
        
        # Viz target distribution
        sample_index = 16
        # self.target_cov_viz(target,target_class,sample_index=sample_index)
        # self.output_cov_viz(pred_total,target_class,sample_index=sample_index)
        # self.out_tar_cov_viz(target,pred_total)
        self.view_target_gmm(target)
        
        return
        sample_index = 30
        loss = ds_data.loss_batch_data[:,sample_index]
        price_data_sample = price_data[sample_index]
        slope_target_sample = slope_target[sample_index]
        x_bar_sample = x_bar_total[sample_index]
        future_price_sample = future_price_target[sample_index]
        pred_sample = pred_total[sample_index]
        pca_target_sample = pca_target[sample_index]
        target_class_sample = target_class[sample_index]
        target_sample = target[sample_index]
        combine_index = np.where((target_class_sample==3)|(target_class_sample==0)|(target_class_sample==1)|(target_class_sample==2))[0]
        # combine_index = np.where((target_class==3)|(target_class==0))[0]
        labels_sample = target_class_sample[combine_index]    
 
        future_price_target_flat = future_price_target.reshape(-1,future_price_target.shape[2])
        
        mds = MDS(n_components=2, dissimilarity='precomputed',random_state=1)        
        mds = MDS(n_components=2, random_state=1)            
        # target_pair_dis = pairwise_distances(torch.Tensor(slope_target_sample).to(device),distance_func=loss_unity.mse_dis,
        #                             make_symmetric=True).cpu().numpy()
        coords = mds.fit_transform(slope_target_sample)                               
        self.matrix_results_viz(coords,labels=labels_sample,name="target_combine")   
                               
        for i in range(3):
            # if i!=2:
            #     continue
            x_bar = x_bar_sample[...,i]
            pred = pred_sample[...,i]
            target_single = target_sample[...,i]
            pca_target_single = pca_target_sample[...,i]
            loss_item = loss[i]
            # print("loss is:",loss_item)
            # 生成配对距离矩阵
            # output_pair_dis = pairwise_distances(torch.Tensor(pred).to(device),distance_func=loss_unity.mse_dis,
            #                             make_symmetric=True).cpu().numpy()         

            # 生成二维坐标数据
            # coords = mds.fit_transform(output_pair_dis)    
            coords = mds.fit_transform(pred)        
            
            # 可视化，使用二维坐标在图形展示        
            self.matrix_results_viz(coords=coords,labels=labels_sample,noise_index=None,name="output_s_{}".format(i))

            
        # 合并指标聚类分析
        # output = output_data[combine_index,1:]        
        # target_ana = target[combine_index]
        # output_pair_dis = pairwise_distances(torch.Tensor(output).to(device),distance_func=loss_unity.ccc_distance_torch,
        #                             make_symmetric=True,reduction="max").cpu().numpy()         
        # self.matrix_results_viz(output_pair_dis,labels=labels,name="output_combine")  
        # target_pair_dis = pairwise_distances(torch.Tensor(target_ana).to(device),distance_func=loss_unity.ccc_distance_torch,
        #                             make_symmetric=True).cpu().numpy()
        # self.matrix_results_viz(target_pair_dis,labels=labels,name="target_combine")   
               
    
    def target_cov_viz(self,target_data,target_class,sample_index=0):
        
        sp_len = 10
        sample_index = 3
        sample_target = target_data[sample_index]
        sample_target_class = target_class[sample_index]
        split_size = target_data.shape[0]//sp_len
        split_index_arr = np.array_split(np.arange(target_data.shape[1]), sp_len)
        
        for i in range(target_data.shape[-1]):
            cov_matrix = batch_cov(torch.Tensor(target_data[...,i])).numpy()   
            cov_target = cov_matrix[:,:,0]        
            sample_cov_target = cov_target[sample_index]
            coords = None
            for j in range(sp_len):
                axis_y = np.array([(j + 1) for _ in range(split_index_arr[j].shape[0])])
                axis_x = sample_cov_target[split_index_arr[j]]
                item = np.array([(a,b) for a, b in zip(axis_x,axis_y)])
                if coords is None:
                    coords = item
                else:
                    coords = np.concatenate((coords,item),axis=0)
            self.matrix_results_viz(coords=coords,labels=sample_target_class,noise_index=None,name="target_cov_{}".format(i))

    def output_cov_viz(self,output_data,target_class,sample_index=0):
        
        sp_len = 10
        sample_target = output_data[sample_index,:,0,:]
        sample_target_class = target_class[sample_index]
        split_index_arr = np.array_split(np.arange(output_data.shape[1]), sp_len)
        
        for i in range(output_data.shape[-1]):
            sample_cov_target = sample_target[:,i]
            coords = None
            for j in range(sp_len):
                axis_y = np.array([(j + 1) for _ in range(split_index_arr[j].shape[0])])
                axis_x = sample_cov_target[split_index_arr[j]]
                item = np.array([(a,b) for a, b in zip(axis_x,axis_y)])
                if coords is None:
                    coords = item
                else:
                    coords = np.concatenate((coords,item),axis=0)
            self.matrix_results_viz(coords=coords,labels=sample_target_class,noise_index=None,name="output_{}".format(i))

    def out_tar_cov_viz(self,target_data,output_data,target_class=None,sample_index=0):
        
        sp_len = 1
        sample_index = 12
        sample_output = output_data[sample_index,:,0,:]
        split_index_arr = np.array_split(np.arange(target_data.shape[1]), sp_len)
        pred_total = output_data[:,:,0,:]
                           
        viz_util = [TensorViz(env="out_tar_cov_{}".format(i)) for i in range(target_data.shape[-1])]
        names = ["target","output"]
        scale_size = [1,50,20]
        
        for i in range(target_data.shape[-1]):
            viz_util = TensorViz(env="out_tar_cov_{}".format(i))
            
            pred_t = F.softmax(torch.Tensor(pred_total[:,:,i]), dim=1).numpy() 
            pred_t_sample = pred_t[sample_index,:]
            cov_matrix = batch_cov(torch.Tensor(target_data[...,i]))
            cov_target = cov_matrix[:,:,0]       
            cov_target_t = F.softmax(cov_target, dim=1).numpy()
            cov_target_t = cov_target_t[sample_index]
            for j in range(sp_len):
                sp_output = pred_t_sample[split_index_arr[j]]
                sp_target = cov_target_t[split_index_arr[j]]
                target_title = "ot_cov_{}".format(j)
                win = "win_{}".format(j)
                view_data = np.stack([sp_target*scale_size[i],sp_output]).transpose(1,0)
                viz_util.viz_matrix_var(view_data,win=win,title=target_title,names=names)                   
                
    def view_target_gmm(self,target):     
        
        X = target[1,...,0]
        gmm = GaussianMixture(n_components=4)
        gmm.fit(X)
        clusters = gmm.predict(X)
        probs = gmm.predict_proba(X)
        print(probs[:100].round(2))    
        
                           
    def filter_noise_data(self,data,noise_index,eps=0.1):
        tree = spt.cKDTree(data=data)  
        results = []
        for n_index in noise_index:
            point = data[n_index]
            distances, indexs = tree.query(point, k=2)
            if distances[1]>eps:
                results.append(n_index)
        return np.array(results)
                        
    def matrix_results_viz(self,coords=None,labels=None,noise_index=None,name="target"):
        plt.figure(name,figsize=(12,9))
        if labels is None:
            plt.scatter(coords[:,0],coords[:,1],marker='o')
        else:
            p_index = np.where(labels==3)[0]
            p2_index = np.where(labels==2)[0]
            n_index = np.where(labels==0)[0]
            n2_index = np.where(labels==1)[0]
            plt.scatter(coords[p_index,0],coords[p_index,1], marker='o',color="r", s=50)
            plt.scatter(coords[p2_index,0],coords[p2_index,1], marker='o',color="b", s=50)
            plt.scatter(coords[n_index,0],coords[n_index,1], marker='x',color="k", s=50)
            plt.scatter(coords[n2_index,0],coords[n2_index,1], marker='x',color="y", s=50)
            # 显示离群点
            if noise_index is not None and noise_index.shape[0]>0:
                plt.scatter(coords[noise_index,0],coords[noise_index,1], marker='p',color="m", s=80)
        # plt.show()      
        plt.savefig('./custom/data/results/{}_matrix_result.png'.format(name))  
        return coords
    
    
    def draw_distance_elbaw(self,distance_data):
        """做肘形图，通过拐点确定dbscan的聚类半径(eps)参数"""
        
        plt.figure(figsize=(10,5))
        distances = np.sort(distance_data, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.show()      
        print("el")  
        
    
    def dbscan_results(self,db,data,name="output"):
        labels = db.labels_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True        
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)        
        
        # plt.figure(name)
        #
        # # Black removed and is used for noise instead.
        # unique_labels = set(labels)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #
        #     class_member_mask = (labels == k)
        #     xy = data[class_member_mask]
        #     if k == -1:
        #         noise_index = np.where(xy)[0]
        #         noise_index = self.filter_noise_data(data, noise_index)
        #         xy = xy[noise_index]
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=6)
        #     if k == -1:
        #         xy_rtn = xy[:, :2]           
        
        # plt.title('Estimated number of clusters: %d' % n_clusters_)    
        # plt.savefig('./custom/data/results/{}_cluster_result.png'.format(name))
        
        return labels

    def spec_results(self,pred,data,name="output"):
        labels = pred.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)        
        
        plt.figure(name)
        
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
        
            class_member_mask = (labels == k)
            xy = data[class_member_mask]
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
            if k == -1:
                xy_rtn = xy[:, :2]           
        
        plt.title('Estimated number of clusters: %d' % n_clusters_)    
        # plt.savefig('./custom/data/results/{}_cluster_result.png'.format(name))
        
    def est_thredhold(self,target_ds):
        """预估预测阈值"""
        
        # self.viz_est = TensorViz(env="dataview_est_thredhold")
        viz_target = TensorViz(env="data_target")
        target_data = target_ds.target_data
        (past_target,target_class,future_target,pca_target,price_target,target_info) = target_data
        target_class = target_class[:,0,0]
        bins_cnt = 10
        total_size = pca_target.shape[0] 
        names = ["tar_{}".format(i) for i in range(4)]
        for j in range(pca_target.shape[-1]):
            plt.figure(figsize=(24,8))
            bins_data = []
            # 分箱模式，拆分为多个区间
            pca_target_item = pca_target[:,0,j]
            data_cont = np.histogram(pca_target_item, bins=bins_cnt)            
            for i in range(len(CLASS_SIMPLE_VALUES)):
                plt.subplot(1, len(CLASS_SIMPLE_VALUES), i+1)
                title = "est_type{}".format(i)
                plt.title(title)
                idx = np.where(target_class==i)[0]
                pca_target_v = pca_target_item[idx]
                # 根据总体分箱范围，对特定类别进行再次区域拆分统计
                data_item = np.histogram(pca_target_v, bins=data_cont[1])  
                # 计算数量占比
                data_item_per = np.round(data_item[0]/total_size,5)
                bins_data.append(data_item_per)
                # 做柱状图，衡量重叠部分阈值
                plt.hist(pca_target_v, data_cont[1], alpha=0.5, label=title)
            bins_data = np.stack(bins_data).transpose(1,0)
            win = "win_{}".format(j)
            target_title = "target_{}".format(j)
            viz_target.viz_matrix_var(bins_data,win=win,title=target_title,names=names,x_range=data_cont[1])   
            save_path = "custom/data/results/est/estth_{}.png".format(j)
            plt.savefig(save_path)
            plt.clf()
            # plt.close()        
                    
if __name__ == "__main__":       
    data_assis = StatDataAssis()
    # data_assis.view_data()
    # data_assis.analysis_data()
    data_assis.analysis_output_data()
        