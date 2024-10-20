import numpy as np
import matplotlib.pyplot as plt

import time
import warnings
import torch
import torch.nn.modules
import torch.nn
import torch.nn.functional as F 
from torch.nn.parameter import Parameter

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from itertools import cycle, islice

def test_dbscan():
    
    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    
    X = StandardScaler().fit_transform(X)
    
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    
    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    
    plt.title('Estimated number of clusters: %d' % n_clusters_)    



def test_SpectralClustering():
    from sklearn import datasets
    
    np.random.seed(0)
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    
    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)
    
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    
    plot_num = 1
    
    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}
    
    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]
    
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
    
        X, y = dataset
    
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
    
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
    
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
    
        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        optics = cluster.OPTICS(min_samples=params['min_samples'],
                                xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
    
        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('OPTICS', optics),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )
    
        for name, algorithm in clustering_algorithms:
            t0 = time.time()
    
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)
    
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)
    
            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
    
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
    
    plt.show()
    print("done")
     

def test_matirx_view():

    # create testing data which is 4x5 data
    mat = np.arange(20).reshape(4,5)
    print(mat)
    
    # Save Image Function
    fig = plt.figure(figsize=(10,8))
    ax = plt.gca()
    cax = plt.imshow(mat, cmap='viridis')
    # set up colorbar
    cbar = plt.colorbar(cax, extend='both', drawedges = False)
    cbar.set_label('Intensity',size=36, weight =  'bold')
    cbar.ax.tick_params( labelsize=18 )
    cbar.minorticks_on()
    # set up axis labels
    ticks=np.arange(0,mat.shape[0],1)
    ## For x ticks
    plt.xticks(ticks, fontsize=12, fontweight = 'bold')
    ax.set_xticklabels(ticks)
    ## For y ticks
    plt.yticks(ticks, fontsize=12, fontweight = 'bold')
    ax.set_yticklabels(ticks)
    plt.savefig('test.png', dpi = 300)
    plt.close()    
 
 
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hiddens,n_outputs):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_features,n_hiddens)
        self.predict=torch.nn.Linear(n_hiddens,n_outputs)
 
    def forward(self, x):
        x=F.relu(self.hidden(x))
        predict=F.softmax(self.predict(x))
        return predict
 
class MyNet:
    def __init__(self,n_features,n_hiddens,n_outputs,times):
        self.NeuronalNet=Net(n_features,n_hiddens,n_outputs)
        self.realX=None
        self.realY=None
        self.opitimizer=None
        self.lossFunc=None
        self.times=times
        
    def getData(self):
        temp = torch.ones(100, 2)
 
        B = torch.normal(2 * temp, 1)
 
        By = torch.ones(100)
        A = torch.normal(-2 * temp, 1)
        Ay = torch.zeros(100)
 
        self.realX = (torch.cat([A, B], 0))
        self.realY = (torch.cat([Ay, By]).type(torch.LongTensor))
 
        # plt.scatter(realX.data.numpy()[:,0],realX.data.numpy()[:,1],c=realY)
        # plt.show()
 
 
    def run(self):
        self.opitimizer=torch.optim.SGD(self.NeuronalNet.parameters(),lr=0.01)
        self.lossFunc=torch.nn.CrossEntropyLoss()
 
        for i in range(self.times):
            out=self.NeuronalNet(self.realX)
 
            loss=self.lossFunc(out,self.realY)
 
            self.opitimizer.zero_grad()
 
            loss.backward()
 
            self.opitimizer.step()
 
    def showBoundary(self):
        x_min, x_max = self.realX[:, 0].min() - 0.1, self.realX[:, 0].max() + 0.1
        y_min, y_max = self.realX[:, 1].min() - 0.1, self.realX[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
        cmap = plt.cm.Spectral
 
        X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        y_pred = self.NeuronalNet(X_test)
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.reshape(xx.shape)
 
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(self.realX[:, 0], self.realX[:, 1], c=self.realY, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("binary classifier")
        plt.show()
        
 
    def predict(self,inputData):
        #inputData should be a 1x2 matrix
        data=torch.from_numpy(np.array(inputData)).int()
        return self.NeuronalNet(data.float())
 

def test_bound():
 
    myNet = MyNet(2,18,2,1000)
    myNet.getData()
    myNet.run()
    myNet.showBoundary()
    probabilitys=list(myNet.predict([3, 3]).data.numpy())
    print("class:{}".format(1+probabilitys.index(max(probabilitys))))
 
 
 
   
      
if __name__ == "__main__":
    # test_dbscan()
    # test_SpectralClustering()
    # test_matirx_view()
    test_bound()
    
        
    
    