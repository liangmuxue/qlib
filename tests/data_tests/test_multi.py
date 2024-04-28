from tslearn.generators import random_walk_blobs
from tslearn.svm import TimeSeriesSVC
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

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
    
def test_gmm():
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # X = np.array([i for i in range(100)])
    gmm = GaussianMixture(n_components=4)
    gmm.fit(X)
    clusters = gmm.predict(X)
    probs = gmm.predict_proba(X)
    print(probs[:100].round(2))
    
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
    plt.title("Gaussian Mixture Model Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    # test_tslearn()
    test_gmm()
    
    