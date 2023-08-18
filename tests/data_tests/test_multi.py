from tslearn.generators import random_walk_blobs
from tslearn.svm import TimeSeriesSVC

def test_tslearn():
    
    X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    clf = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
    ret = clf.fit(X, y).predict(X)
    sv = clf.support_vectors_
    sv_sum = sum([sv_i.shape[0] for sv_i in sv])
    sv_sum == clf.svm_estimator_.n_support_.sum()
    print("sv_sum:",sv_sum)
    clf.decision_function(X).shape
    clf.predict_log_proba(X).shape
    clf.predict_proba(X).shape
    print("ok")
    

if __name__ == "__main__":
    test_tslearn()