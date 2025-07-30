import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import sklearn.neighbors

from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax, \
    TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, \
    KNeighborsTimeSeries

from cus_utils.log_util import AppLogger
logger = AppLogger()

def test_ori():
    np.random.seed(0)
    n_ts_per_blob, sz, d, n_blobs = 20, 100, 1, 2
    
    # Prepare data
    X, y = random_walk_blobs(n_ts_per_blob=n_ts_per_blob,
                             sz=sz,
                             d=d,
                             n_blobs=n_blobs)
    scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))  # Rescale time series
    X_scaled = scaler.fit_transform(X)
    
    indices_shuffle = np.random.permutation(n_ts_per_blob * n_blobs)
    X_shuffle = X_scaled[indices_shuffle]
    y_shuffle = y[indices_shuffle]
    
    X_train = X_shuffle[:n_ts_per_blob * n_blobs // 2]
    X_test = X_shuffle[n_ts_per_blob * n_blobs // 2:]
    y_train = y_shuffle[:n_ts_per_blob * n_blobs // 2]
    y_test = y_shuffle[n_ts_per_blob * n_blobs // 2:]
    
    # Nearest neighbor search
    knn = KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
    knn.fit(X_train, y_train)
    dists, ind = knn.kneighbors(X_test)
    print("1. Nearest neighbour search")
    print("Computed nearest neighbor indices (wrt DTW)\n", ind)
    print("First nearest neighbor class:", y_test[ind[:, 0]])
    
    # Nearest neighbor classification
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    print("\n2. Nearest neighbor classification using DTW")
    print("Correct classification rate:", accuracy_score(y_test, predicted_labels))
    
    # Nearest neighbor classification with a different metric (Euclidean distance)
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    print("\n3. Nearest neighbor classification using L2")
    print("Correct classification rate:", accuracy_score(y_test, predicted_labels))
    
    # Nearest neighbor classification based on SAX representation
    metric_params = {'n_segments': 10, 'alphabet_size_avg': 5}
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="sax",
                                             metric_params=metric_params)
    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    print("\n4. Nearest neighbor classification using SAX+MINDIST")
    print("Correct classification rate:", accuracy_score(y_test, predicted_labels))


def test_busi_data():
    # print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
    scaler = TimeSeriesScalerMinMax(value_range=(0., 1.))
    X = np.load("custom/data/asis/data_train.npy")[:,:,:1]
    y = np.load("custom/data/asis/class_train.npy")[:,0,0]
    X_scaled = scaler.fit_transform(X)
    train_end = X_scaled.shape[0]//3 * 2
    train_end = 3000
    test_start = train_end + 1000
    X_train = X_scaled[:train_end,:,:]
    X_test = X_scaled[train_end:test_start,:,:]
    y_train = y[:train_end]
    y_test = y[train_end:test_start]
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=4, metric="softdtw")
    logger.debug("begin fit")
    knn_clf.fit(X_train, y_train)
    predicted_labels = knn_clf.predict(X_test)
    
    logger.debug("\n2. Nearest neighbor classification using DTW")
    logger.debug("Correct classification rate:{}".format(accuracy_score(y_test, predicted_labels)))

def test_pred_data():
    result_file_path = "custom/data/results/pred_coll.pkl"
    with open(result_file_path, "rb") as fin:
        result_data = pickle.load(fin)    
    result_data = result_data[(result_data['date']>=20220505)&(result_data['date']<=20220531)]
    result_data.to_csv("custom/data/results/pred_coll.csv")

def test_predResult_data():
    result_file_path = "/home/qdata/workflow/fur_sim_flow_2025/task/162/dump_data/pred_result.pkl"
    with open(result_file_path, "rb") as fin:
        result_data = pickle.load(fin)    
    result_data = result_data[(result_data['date']>=20220511)&(result_data['date']<=20220511)]
    result_data.to_csv("custom/data/results/pred_result.csv")

def update_predResult_data():
    result_file_path = "/home/qdata/workflow/fur_sim_flow_2025/task/162/dump_data/pred_result.pkl"
    with open(result_file_path, "rb") as fin:
        result_data = pickle.load(fin)    
    result_data['date'] = 20250722
    with open(result_file_path, "wb") as fout:
        pickle.dump(result_data, fout)        
              
if __name__ == "__main__":
    # test_ori()
    # test_pred_data()
    # test_predResult_data()
    update_predResult_data()
    # test_busi_data()
    
    
    