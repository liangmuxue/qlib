import torch 
import torch.nn as nn
import numpy as np
import time

def test_tensor():
    a = torch.randn(1, requires_grad=True).cuda()
    y = torch.tensor([31,-3],dtype=torch.float).cuda()
    n = y.where(y > 30.0 ,y.log())   
    print(n)
    
def test_embedding():
    emb = nn.Embedding(5, 3)
    t = torch.Tensor(["600004", "600006", "600007"])
    r = emb(t)                 
    print(r)
    
def test_zip():
    t00 = torch.Tensor([100, 200, 300])
    t01 = torch.Tensor([1,2])
    t10 = torch.Tensor([500, 600, 700])
    t11 = torch.Tensor([5,6])
    t0 = (t00,t01)
    t1 = (t10,t11)
    t = [t0,t1]
    # t_reverse =  [map(list,(zip(*t)))]
    t_reverse =  list(zip(*t))
    target = torch.stack(t_reverse[0]).squeeze(1)
    print(t_reverse)

def test_softmax():
    m = nn.LogSoftmax(dim=1)
    input = torch.randn(2, 3)
    print(input)
    output = m(input)
    print(output)
    index = torch.argmax(output, dim=-1)
    print(index)
    
def test_sort():
    t = torch.rand((20,6)) # 20 bbox [x, y, w, h, confidence, class]
    print(t)
    
    _, indices = torch.sort(t, descending=True, dim=0)
    # print(indices)
    b, idx_unsort = torch.sort(indices, dim=0)
    print(idx_unsort)

def test_pairwise():
    a = torch.tensor([[5.0, 3, 0, 4],[1, 6, 2, 3]])
    b = torch.einsum('ij,kj->ikj', a, a).std(dim=2)
    print(b)    

def ccc_distance_torch(x,y):
    from torchmetrics.regression import ConcordanceCorrCoef
    x = x.squeeze().transpose(1,0)
    y = y.squeeze().transpose(1,0)
    concordance = ConcordanceCorrCoef(num_outputs=x.shape[1]).to("cuda:0")
    return 1 - concordance(x, y)
        
def ccc_distance(input_ori,target_ori):
    if len(input_ori.shape)==1:
        input_with_dims = input_ori.unsqueeze(0)
    else:
        input_with_dims = input_ori
    if len(target_ori.shape)==1:
        target_with_dims = target_ori.unsqueeze(0)    
    else:
        target_with_dims = target_ori                    
    input = input_with_dims.flatten()
    target = target_with_dims.flatten()
    corr_tensor = torch.stack([input,target],dim=0)
    cor = torch.corrcoef(corr_tensor)[0][1]
    var_true = torch.var(target)
    var_pred = torch.var(input)
    sd_true = torch.std(target)
    sd_pred = torch.std(input)
    numerator = 2*cor*sd_true*sd_pred
    mse_part = MseLoss().forward(input_with_dims,target_with_dims)
    denominator = var_true + var_pred + mse_part
    ccc = numerator/denominator
    ccc_loss = 1 - ccc
    return ccc_loss  

def test_corr():
    clu = torch.rand(4,5)
    print("clu",clu)
    target = torch.rand(6,5)
    print("target",target)
    corr_tensor = torch.concat([clu,target],dim=0)
    corr = torch.corrcoef(corr_tensor)
    print("corr",corr)
    corr_real = corr[clu.shape[0]:,:clu.shape[0]]
    print("corr_real",corr_real)

class MseLoss():
    """自定义mse损失，用于设置类别权重"""
    
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean',device=None) -> None:
        self.reduction= reduction
        self.device = device

    def forward(self, input, target):
        loss_arr = (input - target) ** 2
        loss_arr = torch.mean(loss_arr,dim=1)
        # if self.reduction=="mean":
        #     mse_loss = torch.mean(loss_arr,dim=1)
        # else:
        #     mse_loss = torch.sum(loss_arr,dim=1)
        return loss_arr 
    
def test_kmeans():
    from projects.kmeans_pytorch import kmeans, kmeans_predict
    data_size, dims, num_clusters = 100000, 2, 4
    x = np.random.randn(data_size, dims) / 6
    x = torch.from_numpy(x)  
    device = torch.device('cuda:0')  
    # k-means
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='soft_dtw',device=device, 
        gamma_for_soft_dtw=0.0001,dist_func=ccc_distance_torch,iter_limit=100
    )    
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=x, num_clusters=num_clusters, distance='soft_dtw',device=device, gamma_for_soft_dtw=0.0001,dist_func=None
    # )        
    print(cluster_centers)

def test_repeat():      
    x = torch.tensor([1, 2, 3])
    x = x.repeat(1,3).squeeze()
    print(x)
    x = torch.arange(0, 10)
    print(x)
    b=torch.randperm(5)
    print(b)

def test_mul():
    feats = torch.Tensor([[.1,.2,.9],[.1,.3,.8],[.1,.2,.9],[.1,.3,.8],[.1,.2,.9],[.1,.3,.8]])
    print(feats.shape)
    sim_mat = torch.matmul(feats, torch.t(feats))
    print(sim_mat)

def test_where():
    tensor = torch.rand([10,6300]).to("cuda:0")
    for i in range(100):
        tensor = torch.rand([10,6300]).to("cuda:0")
        t1 = time.time()
        torch.where(tensor>0.9)
        t2 = time.time()
        print("time is",(t2-t1)*1000)

def test_transfer():
    arr = np.ones([1,1920,1080,3])
    for i in range(100):
        t1 = time.time()
        tensor = torch.Tensor(arr)#.to("cuda:0")
        # tensor = torch.from_numpy(arr).to("cuda:0")
        t2 = time.time()
        print("time is",(t2-t1)*1000)

def test_pca():
    k = 1
    tensor = torch.rand([128,5]).to("cuda:0")
    cov = torch.matmul(torch.t(tensor), tensor) / tensor.size(0)
    # cov = torch.cov(tensor.T)
    # tensor = tensor - torch.mean(tensor, dim=0)
    eigvals, eigvecs = torch.eig(cov, eigenvectors=True)
    idx = eigvals.sort(descending=True)[1][:k]
    transform = eigvecs[:, idx]
    rtn = torch.matmul(tensor, transform)
    print("rtn",rtn)
         
if __name__ == "__main__":
    # test_tensor()    
    # test_sort()
    # test_kmeans()
    # test_repeat()
    # test_mul()
    # test_corr()
    test_pca()
    # test_transfer()
    # test_pairwise()
    # test_embedding()
    # test_zip()
    # test_softmax()