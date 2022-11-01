from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter





def test_imbalance():        
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=1, n_redundant=1, flip_y=0,
                               n_features=2, n_clusters_per_class=1, n_samples=1000, random_state=10)
    print(Counter(y))
    # Counter({1: 900, 0: 100})
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(Counter(y_res))
    print(X_res.shape)
    
if __name__ == "__main__":
    # test_pd_index()
    test_imbalance()

    
    