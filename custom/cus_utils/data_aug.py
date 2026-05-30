import os
import yaml
import numpy as np
import pandas as pd
import random
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from cus_utils.db_accessor import DbAccessor
from pickle import TRUE
from cus_utils.common_compute import linear_map, normalization_axis,check_iqr_outler
from darts_pro.mam.futures_transformer_module import build_mul_scale_arr
from tft.class_define import get_simple_class
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)
                
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def box_plot_outliers(data_ser, box_scale):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    val_low = data_ser.quantile(0.25) - iqr
    val_up = data_ser.quantile(0.75) + iqr
    rule_low = (data_ser < val_low)
    rule_up = (data_ser > val_up)
    return (rule_low, rule_up), (val_low, val_up)
    
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认box_plot(scale=3)进行清洗
    param data: 接收pandas数据格式
    param col_name: pandas列名
    param scale: 尺度
    """
    
    data_n = data.copy()
    data_serier = data_n[col_name]
    rule, value = box_plot_outliers(data_serier, box_scale=scale)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]
    print("Delete number is:{}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is:{}".format(data_n.shape[0]))
    index_low = np.arange(data_serier.shape[0])[rule[0]]
    outliers = data_serier.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_serier.shape[0])[rule[1]]
    outliers = data_serier.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    # sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n      

def aug_data_view(file_path):
    data = np.load(file_path,allow_pickle=True)
    print("shape",data.shape)

def aug_pd_data_view(file_path):
    df = pd.read_pickle(file_path)
    print("df",df)
    
def aug_data_to_pd(file_path,tar_file_path,columns):
    data = np.load(file_path,allow_pickle=True)
    data = np.reshape(data,(-1,len(columns)))
    pd_data = pd.DataFrame(data,columns=columns)
    pd_data.to_pickle(tar_file_path)
    print("save ok")
        
def aug_data_process(file_path,train_path=None,test_path=None,sp_rate=0.7):
    """再次加工数据"""
    
    data = np.load(file_path,allow_pickle=True)
    # 累加时间部分的后五项数据，形成15+1=16的长度 
    data = np.concatenate((data[:,0:15,:],np.expand_dims(data[:,15:,:].sum(axis=1),axis=1)),axis=1)
    train_len = data.shape[0] * sp_rate
    # 切分并保存
    train_data = data[:train_len,:,:]   
    test_data = data[train_len:,:,:]   
    np.save(train_path,train_data)
    np.save(test_path,test_data)

def compare_dataset_consistence():
    """比较验证数据与推理数据的一致性"""
    
    val_file_path = "custom/data/results/data_compare_val_20250313.pkl"
    pred_file_path = "custom/data/results/data_compare_predresult_20250313.pkl"
    with open(val_file_path, "rb") as fin:
        result_data_val = pickle.load(fin)           
    with open(pred_file_path, "rb") as fin:
        result_data_pred = pickle.load(fin) 

    names = ["target_info","past_target_total", "past_covariate_total", "historic_future_covariates_total","future_covariates_total","static_covariate_total"
               ,"past_future_round_targets","index_round_targets"]
    eps = 1e-3
    for i in range(1,len(result_data_val)):
        val_item = result_data_val[i]
        pred_item = result_data_pred[i]
        diff = np.abs(val_item - pred_item)
        compare_rs = np.where(diff>eps)
        # if names[i]=="past_future_round_targets":
        #     print("ggg")
        if np.sum(compare_rs)>1:
            print("{} difference:{}".format(names[i],compare_rs))

def compare_clean_data_and_continus_data(match_date=None):
    """比较用于训练的数据和主连数据是否一致及完备"""
    
    dbaccessor = DbAccessor({})
    ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
    ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
    compare_results = []
    for row in ins_data.values:
        symbol = row[0] 
        sql = "select c.date,v.exchange_id from dominant_continues_data_cross c left join trading_variety v on " \
            "c.code=v.code where c.code='{}' order by date desc limit 1".format(symbol)
        result = dbaccessor.do_query(sql)
        if len(result)==0:
            date = 0
            exchange_id = 0
        else:
            date = int(result[0][0].strftime("%Y%m%d"))
            exchange_id = int(result[0][1])
        compare_results.append([symbol,date,exchange_id])
    # print(compare_results)
    compare_results = pd.DataFrame(np.array(compare_results),columns=['code','date','exchange_id'])
    compare_results['date'] = compare_results['date'].astype(int)
    compare_results['exchange_id'] = compare_results['exchange_id'].astype(int)
    lack_data = compare_results[compare_results['date']<match_date]
    print("lack data:{}".format(lack_data))
    print("min date:{}".format(lack_data['date'].min()))
    return lack_data

def compare_clean_data_and_1min_cross_data(match_date=None):
    """比较用于训练的数据和1分钟主力合约交错数据是否一致及完备"""
    
    dbaccessor = DbAccessor({})
    from data_extract.akshare_futures_extractor import AkFuturesExtractor
    extractor = AkFuturesExtractor(savepath="/home/qdata/futures_data")   
    
    ins_file_path = "/home/qdata/qlib_data/futures_data/instruments/clean_data.txt"
    ins_data = pd.read_table(ins_file_path,sep='\t',header=None)
    compare_results = []
    for row in ins_data.values:
        symbol = row[0] 
        main_contract_name = extractor.get_main_contract_name(symbol, str(match_date), use_1min_data=True)
        item_save_path = os.path.join(extractor.get_1min_save_path(),"{}.csv".format(main_contract_name))
        # 检查分钟数据（交错模式）的合约文件是否存在
        if not os.path.exists(item_save_path):
            file_exists_flag = 0
            main_code = symbol
            date = '2005-01-01 00:00:00'
        else:
            file_exists_flag = 1
            sql = "select datetime from dominant_real_data_1min_cross where code='{}' order by datetime desc limit 1".format(main_contract_name)
            result = dbaccessor.do_query(sql)
            if len(result)==0:
                date = '2005-01-01 00:00:00'
                main_code = symbol
            else:
                date = result[0][0]
                main_code = main_contract_name
        compare_results.append([symbol,main_code,date,file_exists_flag])
    # print(compare_results)
    compare_results = pd.DataFrame(np.array(compare_results),columns=['code','main_code','date','file_exists_flag'])
    compare_results['file_exists_flag'] = compare_results['file_exists_flag'].astype(int)
    compare_results['date'] = compare_results['date'].astype('datetime64[ns]')
    match_date_date = datetime.strptime(str(match_date), "%Y%m%d")
    lack_data = compare_results[compare_results['date']<match_date_date]
    print(compare_results)
    print("lack data:{}".format(lack_data))
    # print("min date:{}".format(lack_data['date'].min()))
    return lack_data

def get_high_corr_pairs(corr_matrix, threshold=0.85):
    correlated_pairs = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlated_pairs.add((col1, col2, corr_matrix.iloc[i, j]))
    return list(correlated_pairs)

def compare_distribution_ks(train_feat, val_feat, feat_name):
    # KS 检验
    ks_stat, p_value = stats.ks_2samp(train_feat, val_feat)
    return {
        "feature": feat_name,
        "ks_stat": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "is_different": p_value < 0.05  # True=分布不一致
    }

def kl_divergence(p, q):
    p = np.histogram(p, bins=50, density=True)[0] + 1e-8
    q = np.histogram(q, bins=50, density=True)[0] + 1e-8
    return np.sum(p * np.log(p / q))
    
#######################  For Training #############################

class CollResAna():
    
    def __init__(self,file_path,yaml_file):
        self.file_path = file_path
        self.yaml_file = yaml_file

    def prepare_data(self):
        """验证结果数据分析"""
        
        import qlib
        from qlib.utils import  init_instance_by_config
        from qlib.workflow import R
        
        result_file_path = os.path.join(self.file_path,"coll_record.csv") 
        col_data_types = {"top_index":int,"instrument":str,"yield_rate":float,"result":int,"trend_value":int,"date":int}   
        self.coll_result_data = pd.read_csv(result_file_path,dtype=col_data_types)  
        # 使用验证数据集的数据协助分析
        yaml_file = self.yaml_file 
        with open(yaml_file) as fp:
            config = yaml.safe_load(fp)    
        experiment_name = "workflow"
        qlib_init_config = config["qlib_init"]
        qlib.init(provider_uri=qlib_init_config["provider_uri"], region=qlib_init_config["region"])  
        with R.start(experiment_name=experiment_name, recorder_name=None):              
            dataset = init_instance_by_config(config["task"]["dataset"])
            train_data,val_data = dataset.build_series_data()
            train_series_transformed,past_convariates_train,future_convariates_train = train_data
            val_series_transformed,past_convariates_val,future_convariates_val = val_data
            process_model = init_instance_by_config(config["task"]["model"])
            process_model.init_env(dataset)
            self.output_chunk_length = process_model.optargs["forecast_horizon"]
            emb_size = dataset.get_emb_size()
            model = process_model._build_model(dataset,emb_size=emb_size,use_model_name=False,mode=1) 
            model.set_outer_params({'pred_weights':process_model.optargs["pred_weights"],'mode':process_model.type,'candidate_inverse':process_model.optargs['candidate_inverse']}) 
            model.mode = "predict"
            _,_,train_loader,val_loader= \
            model.fit(train_series_transformed, past_covariates=past_convariates_train, future_covariates=future_convariates_train,
                    val_series=val_series_transformed,val_past_covariates=past_convariates_val,val_future_covariates=future_convariates_val,
                     max_samples_per_ts=None,trainer=None,epochs=0,verbose=True,num_loader_workers=0,seperate_mode=False)  
            self.val_dataset = val_loader.dataset
            self.train_dataset = train_loader.dataset
        self.tft_dataset = dataset
    
    def extre_data_invest(self):
        """排查异常数据"""
        
        futures_dataset = self.train_dataset
        # futures_dataset = self.val_dataset
        sw_ins_mappings = futures_dataset.sw_ins_mappings
        scale_arr = build_mul_scale_arr(sw_ins_mappings,mode=6,dataset=self.tft_dataset)
        outler_data = []
        outler_scale_data = []
        scale_diff_in_outlier = None
        for i in range(len(futures_dataset)):
            past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                index_round_targets,long_diff_seq_targets,target_info_total = futures_dataset[i]
            future_start_datetime = int(futures_dataset.date_list[i])   
            ins_diff = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info_total)])
            ins_diff_norm = np.array([t['open_diff_norm'] if t is not None else 0 for t in np.array(target_info_total)])
            scale_diff = []
            scale_diff_all = []
            for k,scale_item in scale_arr.iterrows():
                ins = scale_item['instruments']
                ins_diff_inner = np.array([t['open_diff'] if t is not None else 0 for t in np.array(target_info_total)[ins]])
                # ins_diff_norm_inner = np.array([t['open_diff_norm'] if t is not None else 0 for t in np.array(target_info_total)[ins]])
                ins_diff_inner = ins_diff_inner[ins_diff_inner!=0]
                if ins_diff_inner.shape[0]>0:
                    mean_data = ins_diff_inner.mean()
                    scale_diff.append(mean_data)
                    scale_diff_all.append([future_start_datetime,k,mean_data])
                else:
                    scale_diff_all.append([future_start_datetime,k,0])
            scale_diff = np.array(scale_diff)
            scale_diff_all = np.array(scale_diff_all)
            outler = check_iqr_outler(ins_diff)
            outler_scale = check_iqr_outler(scale_diff,10)
            
            if outler.shape[0]>0:
                outler_data.append([future_start_datetime,outler.index.values,np.round(outler.values[:,0],2)])
            if outler_scale.shape[0]>0:
                if scale_diff_in_outlier is None:
                    scale_diff_in_outlier = scale_diff_all
                else:
                    scale_diff_in_outlier = np.concatenate([scale_diff_in_outlier,scale_diff_all])
                outler_scale_data.append([future_start_datetime,outler_scale.index.values,np.round(outler_scale.values[:,0],2)])        
        outler_data = pd.DataFrame(np.array(outler_data),columns=['date','index','value'])
        outler_scale_data = pd.DataFrame(np.array(outler_scale_data),columns=['date','index','value'])
        scale_diff_in_outlier = pd.DataFrame(np.array(scale_diff_in_outlier),columns=['date','index','value'])
        outler_scale_data
                          
    def build_match_results(self):
        """分别找出比较准的日期和不太准的日期"""
        
        yield_results = self.coll_result_data.groupby(by='date').apply(lambda x: x['diff_range'].sum()).reset_index()
        yield_results = pd.DataFrame(yield_results.values,columns=['date','yield'])
        
        match_results = self.coll_result_data.groupby(by='date').apply(
            lambda x: pd.Series([(x['target_class']>=2).sum(), x['diff_range'].sum()], index=['match_cnt','yield'])
        ).reset_index()
        # match_results =  pd.DataFrame(match_results.values,columns=['date','match_cnt','yield'])
        match_dates = match_results[match_results['match_cnt']>=3].index.values
        no_match_dates = match_results[match_results.values<3].index.values
        return  match_results, match_dates,  no_match_dates         

    def comprisive_stat(self):
        self.prepare_data()
        # self.extre_data_invest()
        self.fea_rel_stat()
        # self.relative_stat()
        # self.normal_stat()
        # self.scale_info_stat()
        # self.trend_info_stat()
        # self.price_range_stat()
        # self.target_corr_stat()
        # self.ins_index_stat()

    def target_corr_stat(self):
        """查看价格涨跌幅与辅助指标的协同关系"""
 
        futures_dataset = self.train_dataset
        futures_dataset = self.val_dataset
        main_index = futures_dataset.main_index
        instrument_index = futures_dataset.instrument_index
        output_chunk_length = self.output_chunk_length
        cut_len = futures_dataset.cut_len
        self.cut_len = cut_len
        diff_data_total = []
        trend_data_total = []
        target_index = 0
        att_target_index = 1
        
        for i in range(len(futures_dataset)):
            past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
            future_start_datetime = int(futures_dataset.date_list[i])    
            price_diff = long_diff_seq_targets[0]    
            # 取cut_len相关目标值做比较
            target_len = -output_chunk_length+cut_len-1
            round_target = past_future_round_targets[main_index,target_len,att_target_index]
            open_diff = np.array([item['open_diff_arr'] for item in target_info_total])
            diff_range_main = open_diff[main_index][:-self.output_chunk_length]
            # 优化目标值映射到价格涨跌幅数组数据空间
            target_series = past_target_total[main_index,:,att_target_index]
            round_target_series_mapped = linear_map(target_series, diff_range_main.min(), diff_range_main.max()) 
            # 取得映射后实际价格目标对应的下标，查看映射的数据和实际价格数据的相关性
            round_target_item = round_target_series_mapped[target_len]
            trend_data_total.append([future_start_datetime,price_diff,round_target,round_target_item])  
            
            # 品种间的目标值和价格涨跌幅度的一致性
            round_target_ins = past_future_round_targets[instrument_index,target_len,target_index]
            price_diff_ins = price_targets[instrument_index]
            price_diff_ins = self.compute_diff_range_class(None, target_info_arr=np.array(target_info_total)[instrument_index],jump_mode=False)[0]            
            diff_data_arr = np.stack([round_target_ins,price_diff_ins]).transpose(1,0)
            diff_data_arr = pd.DataFrame(diff_data_arr,columns=['target_round_ins','price_diff_ins'])
            corr_data = diff_data_arr[['target_round_ins','price_diff_ins']].corr().values
            top_num = 3
            top_round = price_diff_ins[np.argsort(price_diff_ins)[:top_num]]
            top_round_inverse = price_diff_ins[np.argsort(price_diff_ins)[-top_num:]]                     
            top_round_price = np.concatenate([top_round,top_round_inverse])
            top_round = round_target_ins[np.argsort(round_target_ins)[:top_num]]
            top_round_inverse = round_target_ins[np.argsort(round_target_ins)[-top_num:]]                     
            top_round_att = np.concatenate([top_round,top_round_inverse])   
            diff_top_data_arr = np.stack([top_round_att,top_round_price]).transpose(1,0)
            diff_top_data_arr = pd.DataFrame(diff_top_data_arr,columns=['top_target_round_ins','top_price_diff_ins'])
            top_corr_data = diff_top_data_arr[['top_target_round_ins','top_price_diff_ins']].corr().values                     
            diff_data_total.append([future_start_datetime,corr_data[0,1],top_corr_data[0,1]])
             
        # 总体趋势数据一致性
        trend_data_total = pd.DataFrame(np.array(trend_data_total),columns=['date','trend_price_diff','trend_round_target','trend_target_map'])
        trend_data_total['date'] = trend_data_total['date'].astype(int)
        corr_data = trend_data_total[['trend_price_diff','trend_round_target','trend_target_map']].corr().values
        pd.set_option('expand_frame_repr', False)
        print("trend_data_total:\n ",trend_data_total)
        print("trend_data corr:\n {}".format(corr_data))
        # 品种间的目标值和价格涨跌幅度的一致性
        diff_data_total = pd.DataFrame(np.array(diff_data_total),columns=['date','round_price_corr','top_round_price_corr'])
        print("round_price corr:\n {}".format(diff_data_total))
        print("round_price mean:{}".format(diff_data_total['round_price_corr'].mean()))
    
    def fea_rel_stat(self):
        """特征间相关性排查"""
        
        dataset = self.tft_dataset
        columns = list(set(dataset.get_past_columns()))
        
        corr_matrix = dataset.df_all[columns].corr()
        high_corr = get_high_corr_pairs(corr_matrix, threshold=0.85)
        for pair in high_corr:
            print(f"{pair[0]} <-> {pair[1]} | corr: {pair[2]:.2f}")        
        
        X = dataset.df_all[columns]
        
        # VIF排查
        X = dataset.df_all[columns]
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values("VIF", ascending=False)
        print(vif_data)
        
        # KS检验
        X_train = self.train_dataset.df_data[columns]
        X_val = self.val_dataset.df_data[columns]
        dist_diff = []
        for col in X_train.columns:
            res = compare_distribution_ks(X_train[col], X_val[col], col)
            dist_diff.append(res)
        
        dist_df = pd.DataFrame(dist_diff)
        dist_df = dist_df.sort_values("is_different", ascending=False)
        print("=== 训练/验证集分布差异（KS检验）===")
        print(dist_df)        
        
        print("=== 训练/验证集分布差异（KL检验）===")
        for col in columns:
            kl = kl_divergence(X_train[col], X_val[col])
            print("{}:{}".format(col,kl))
        
        
    def ins_index_stat(self):
        """查看品种与指数的协同关系"""
 
        futures_dataset = self.train_dataset
        futures_dataset = self.val_dataset
        main_index = futures_dataset.main_index
        instrument_index = futures_dataset.instrument_index
        output_chunk_length = self.output_chunk_length
        cut_len = futures_dataset.cut_len
        diff_data_total = []
        trend_data_total = []
        target_index = 0
        att_target_index = 2
        target_len = -output_chunk_length+cut_len-1
        for i in range(len(futures_dataset)):
            past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
            future_start_datetime = int(futures_dataset.date_list[i])    
            # 价格涨跌幅指标
            price_target_total = np.concatenate([past_target_total[...,0],future_target_total[...,0]],-1)
            price_main = price_target_total[main_index]
            price_main_mean = np.mean(price_target_total[instrument_index],0)
            # 辅助指标
            att_target_total = np.concatenate([past_target_total[...,att_target_index],future_target_total[...,att_target_index]],-1)
            att_target_main = att_target_total[main_index]
            att_target_main_mean = np.mean(att_target_total[instrument_index],0)
            att_main_mean_nor = normalization_axis(att_target_main_mean)
            price_att_arr = np.stack([price_main,price_main_mean,att_target_main,att_target_main_mean,att_main_mean_nor]).transpose(1,0)
            price_att_arr = pd.DataFrame(price_att_arr,columns=['price_index','price_mean','att_index','att_mean','att_mean_nor'])
            att_main_mean_nor_future = normalization_axis(att_target_main_mean[-output_chunk_length:])
            price_future_nor = normalization_axis(price_main_mean[-output_chunk_length:])
            price_att_arr_future = price_att_arr.iloc[-output_chunk_length:]
            price_att_arr_future['att_mean_nor'] = att_main_mean_nor_future
            price_att_arr_future['price_mean_nor'] = price_future_nor
            # 添加目标点位的价格和辅助指标
            price_cut_tar = price_att_arr['price_mean'].iloc[target_len]
            price_cut_nor = price_att_arr_future['price_mean_nor'].iloc[target_len]
            att_cut_tar = att_main_mean_nor_future[target_len]
            # 查看整体指数和品种平均的一致性
            # TODO
            # 查看价格指数与辅助指标指数的一致性
            corr_data = price_att_arr_future[['price_mean','att_mean']].corr().values
            diff_data_total.append([future_start_datetime,corr_data[0,1],price_cut_tar,price_cut_nor,att_cut_tar])            

        # 价格指数与辅助指标指数的一致性
        diff_data_total = pd.DataFrame(np.array(diff_data_total),columns=['date','price_att_corr','price_cut_tar','price_cut_nor','att_cut_tar'])
        print("price_att_corr:\n {}".format(diff_data_total))
        print("price_att_corr mean:{}".format(diff_data_total['price_att_corr'].mean()))

    def check_train_val_corr(self):
        """检查训练集和测试集的数据一致性"""
 
        futures_dataset = self.train_dataset
        futures_dataset = self.val_dataset
        main_index = futures_dataset.main_index
        instrument_index = futures_dataset.instrument_index
        output_chunk_length = self.output_chunk_length
        cut_len = futures_dataset.cut_len
        diff_data_total = []
        trend_data_total = []
        target_index = 0
        att_target_index = 2
        target_len = -output_chunk_length+cut_len-1
        for i in range(len(futures_dataset)):
            past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
            future_start_datetime = int(futures_dataset.date_list[i])    
            # 价格涨跌幅指标
            price_target_total = np.concatenate([past_target_total[...,0],future_target_total[...,0]],-1)
            price_main = price_target_total[main_index]
            price_main_mean = np.mean(price_target_total[instrument_index],0)
            # 辅助指标
            att_target_total = np.concatenate([past_target_total[...,att_target_index],future_target_total[...,att_target_index]],-1)
            att_target_main = att_target_total[main_index]
            att_target_main_mean = np.mean(att_target_total[instrument_index],0)
            att_main_mean_nor = normalization_axis(att_target_main_mean)
            price_att_arr = np.stack([price_main,price_main_mean,att_target_main,att_target_main_mean,att_main_mean_nor]).transpose(1,0)
            price_att_arr = pd.DataFrame(price_att_arr,columns=['price_index','price_mean','att_index','att_mean','att_mean_nor'])
            att_main_mean_nor_future = normalization_axis(att_target_main_mean[-output_chunk_length:])
            price_future_nor = normalization_axis(price_main_mean[-output_chunk_length:])
            price_att_arr_future = price_att_arr.iloc[-output_chunk_length:]
            price_att_arr_future['att_mean_nor'] = att_main_mean_nor_future
            price_att_arr_future['price_mean_nor'] = price_future_nor
            # 添加目标点位的价格和辅助指标
            price_cut_tar = price_att_arr['price_mean'].iloc[target_len]
            price_cut_nor = price_att_arr_future['price_mean_nor'].iloc[target_len]
            att_cut_tar = att_main_mean_nor_future[target_len]
            # 查看整体指数和品种平均的一致性
            # TODO
            # 查看价格指数与辅助指标指数的一致性
            corr_data = price_att_arr_future[['price_mean','att_mean']].corr().values
            diff_data_total.append([future_start_datetime,corr_data[0,1],price_cut_tar,price_cut_nor,att_cut_tar])            

        # 价格指数与辅助指标指数的一致性
        diff_data_total = pd.DataFrame(np.array(diff_data_total),columns=['date','price_att_corr','price_cut_tar','price_cut_nor','att_cut_tar'])
        print("price_att_corr:\n {}".format(diff_data_total))
        print("price_att_corr mean:{}".format(diff_data_total['price_att_corr'].mean()))
                        
    def price_range_stat(self):
        """统计价格涨跌幅度以及指标的分布情况"""
        
        futures_dataset = self.train_dataset
        futures_dataset = self.val_dataset
        main_index = futures_dataset.main_index
        output_chunk_length = self.output_chunk_length
        data_total = []
        for i in range(len(futures_dataset)):
            past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
            future_start_datetime = int(futures_dataset.date_list[i])    
            price_diff = long_diff_seq_targets[0]    
            target = past_future_round_targets[main_index,-1,0]
            data_total.append([future_start_datetime,price_diff,target])   
        data_total = pd.DataFrame(np.array(data_total),columns=['date','price_diff','target'])
        data_total['date'] = data_total['date'].astype(int)
        data_total['price_diff'].describe()
    
    def relative_stat(self):   
        """目标数据与价格数据相关性检验"""
        
        # 遍历并取得指定数据
        futures_dataset = self.val_dataset
        output_chunk_length = self.output_chunk_length
        match_results, match_dates, no_match_dates = self.build_match_results()
        for index,dates in enumerate([match_dates,no_match_dates]):
            target_data = []
            for i in range(len(futures_dataset)):
                past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                    covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                    index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
                future_start_datetime = int(futures_dataset.date_list[i])
                # 计算预测目标与价格目标的相关性
                if future_start_datetime in dates:
                    for idx,target_info in enumerate(target_info_total):
                        diff_range = self._compute_diff_range(target_info, output_chunk_length)
                        future_round_target = past_future_round_targets[idx,-1:,-1] 
                        target_data.append([future_start_datetime,target_info['instrument'],diff_range,future_round_target])
            
            target_data = pd.DataFrame(target_data,columns=['date','instrument','open_diff','round_target'])
            target_data = target_data[~target_data['instrument'].str.startswith("ZS")]
            target_data['date'] = target_data['date'].astype(int)
            target_data = target_data.sort_values(by='date')
            corr_info = target_data.groupby(by='date')[['open_diff','round_target']].corr().reset_index()
            corr_combine = corr_info.iloc[::2][['date','round_target']]
            title = "correct" if index==0 else "fail"
            print("{} eva corr:{}".format(title,corr_combine))
    
    def _compute_diff_range(self,target_info,output_chunk_length):
        open_array = target_info['open_array']
        diff_range = (open_array[-1] - open_array[-output_chunk_length])/open_array[-output_chunk_length]*100
        return diff_range
                
    def normal_stat(self):   
        """统计基础信息"""
        
        futures_dataset = self.val_dataset
        output_chunk_length = self.output_chunk_length
        match_results, match_dates, no_match_dates = self.build_match_results()
        
        for index,dates in enumerate([match_dates,no_match_dates]):
            target_data = []
            for i in range(len(futures_dataset)):
                past_target_total, past_covariate_total, historic_future_covariates_total,future_covariates_total,static_covariate_total, \
                    covariate_future_total,future_target_total,target_class_total,price_targets,past_future_round_targets,\
                    index_round_targets,long_diff_seq_targets,target_info_total  = futures_dataset[i]
                future_start_datetime = int(futures_dataset.date_list[i])
                if future_start_datetime in dates:
                    for target_info in target_info_total:
                        diff_range = self._compute_diff_range(target_info, output_chunk_length)
                        target_data.append([future_start_datetime,target_info['instrument'],diff_range])
                    
            target_data = pd.DataFrame(target_data,columns=['date','instrument','open_diff'])
            target_data = target_data[~target_data['instrument'].str.startswith("ZS")]
            target_data['date'] = target_data['date'].astype(int)
            normal_info = target_data.groupby(by='date')['open_diff'].agg(['mean', 'std']).reset_index()
            title = "correct" if index==0 else "fail"
            print("{} eva info: \n {}".format(title,normal_info))
            # print("{} eva mean:{},std:{}".format(title,normal_info['mean'].describe(),normal_info['std'].describe()))
        self.coll_result_data['date'].isin(match_dates)
    
    def scale_info_stat(self):   
        """根据预测评估数据，统计业务属性信息"""
        
        # 引入基础信息包括行业分类、创建年份等
        result_data = self.coll_result_data.merge(self.tft_dataset.base_info, on='instrument', how='left')
        result_data['industry'] = result_data['industry'].astype(int)
        # 关注失败记录
        fail_result = result_data[result_data['diff_range']<0]
        suc_result = result_data[result_data['diff_range']>=0]
        fail_result

    def trend_info_stat(self):   
        
        save_path = os.path.join(self.file_path,"trend_result.csv")
        df = self.coll_result_data.drop_duplicates(subset=['date', 'pred_trend_value'])
        df[['date', 'pred_trend_value', 'pred_trend_flag','real_trend_values','real_trend_ref_values', 
                             'real_trend_flag','past_ind', 'ind_data', 'trend_eva_diff',
                            'trend_match_flag']].to_csv(save_path, index=False)   
            
    def compute_diff_range_class(self,target_info,target_info_arr=None,is_main=False,jump_mode=False):
        """根据实际涨跌数据计算类别"""
        
        target_len = -self.output_chunk_length+self.cut_len-1
        total_len = -self.output_chunk_length
        if jump_mode:
            target_len = target_len + 1
            total_len = total_len + 1
        # 对于整体指标，不能使用开盘和收盘价格直接计算，使用原数据（所有品种收盘价差的均值,之前的dataset中已经设置好了）
        if is_main:
            # 使用所有品种的均值进行计算
            diff_range_total = np.array([pr['open_diff_arr'][target_len] for pr in target_info_arr])
            diff_range = diff_range_total.mean()
            diff_range_arr =  np.stack([pr["open_diff_arr"] for pr in target_info_arr])
            diff_range_arr = np.mean(diff_range_arr,0)
            range_class = get_simple_class(diff_range)
        else:
            # 收盘与前收盘价差作为衡量指标
            # diff_range = (price_array[-self.output_chunk_length+self.cut_len-1] - price_array[-self.output_chunk_length])/price_array[-self.output_chunk_length]*100
            # 预测结束日期的开盘与预测开始日期的开盘价差作为衡量指标
            if target_info is not None:
                diff_range = target_info['open_diff_arr'][target_len]
                range_class = get_simple_class(diff_range)
                diff_range_arr = None
            # 价差展示，从过去一直延续到预测当日
            elif target_info_arr is not None:
                diff_range_arr = np.stack([pr["open_diff_arr"] for pr in target_info_arr])     
                diff_range = diff_range_arr[:,target_len] 
                range_class = None
            else:
                diff_range_arr = None
                range_class = None
        
        return diff_range,range_class,diff_range_arr    
    
if __name__ == "__main__":
    file_path = "custom/data/aug/test_100.npy"
    pd_file_path = "custom/data/aug/test_100.pkl"
    train_path = "custom/data/aug/test100_all_train.npy"
    test_path = "custom/data/aug/test100_all_test.npy"
    # aug_data_process(file_path,train_path=train_path,test_path=test_path)
    # aug_data_view(file_path)
    # aug_data_to_pd(file_path,pd_file_path,['datetime','instrument','dayofweek','CORD5', 'VSTD5', 'WVMA5', 'label','ori_label'])\
    # aug_pd_data_view(pd_file_path)
    # compare_dataset_consistence()
    # compare_clean_data_and_continus_data(match_date=20251009)
    # compare_clean_data_and_1min_cross_data(match_date=20251009)
    coll_ana = CollResAna("custom/data/results/stats",yaml_file="custom/config/darts/workflow_pred_futures_trans_index.yaml")
    coll_ana.comprisive_stat()
       
    
