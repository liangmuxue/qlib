import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import time
from cus_utils.metrics import pca_apply

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
    k = 2
    tensor = torch.rand([128,5]).to("cuda:0")
    rtn = pca_apply(tensor,k)
    print("rtn",rtn)
    
def test_nor():
    a = 1
    a-=0.5+0.1
    print(a)

def test_js():
    
    def js_div(p_output, q_output, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output )/2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
    
    t1 = torch.Tensor([0.1,0.2,0.3])
    t2 = torch.Tensor([0.5,0.6])
    print(js_div(t1,t2))

def test_cos():
    input1 = torch.Tensor(np.array([0,2]))
    input2 = torch.Tensor(np.array([1,30]))
    
    similarity = torch.cosine_similarity(input1, input2, dim=0)
    print(similarity) 

def test_slice():
    # x = torch.arange(15).reshape(3,-1)
    # idx = torch.tensor([1,2,3])
    #
    # idx = torch.column_stack([idx, idx+1])
    # y = torch.gather(x, 1, idx)    
    #
    # print(x)
    # print(idx)
    # print(y)
    x = torch.ones([10,5,3])
    index = torch.ones([10,2,3]).long()
    t = torch.gather(x, 1, index)  
    print(t.shape)

def test_multi_grad():
    
    # 定义模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    
    # 模拟数据
    x = torch.randn(32, 10)
    target1 = torch.randint(0, 2, (32,)).float()
    target2 = torch.randn(32, 1)
    
    # 损失函数
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    # 前向传播
    output = model(x)
    output1 = output[:, 0]  # 第一个任务输出
    output2 = output[:, 1]  # 第二个任务输出
    
    # 计算损失
    loss1 = criterion1(output1, target1)
    loss2 = criterion2(output2.unsqueeze(1), target2)
    loss_total = loss1 + loss2
    loss_total.backward(retain_graph=True)
    # 查看TOTAL梯度
    print("total backward 后的梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: gradient norm = {param.grad.norm().item():.4f}")    
    model.zero_grad()
    
    # 分别 backward - 保留计算图
    loss1.backward(retain_graph=True)
    
    # 查看第一个损失的梯度
    print("第一次 backward 后的梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: gradient norm = {param.grad.norm().item():.4f}")
            
    model.zero_grad()
    # 第二个 backward
    loss2.backward()
    
    print("\n第二次 backward 后的梯度:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: gradient norm = {param.grad.norm().item():.4f}")
    
    # 更新参数
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.step()
    optimizer.zero_grad()
        
    # def get_gradients(model, loss):
    #     """计算并返回指定损失的梯度"""
    #     # 清零之前的梯度
    #     for param in model.parameters():
    #         if param.grad is not None:
    #             param.grad.data.zero_()
    #
    #     # 计算当前损失的梯度
    #     loss.backward(retain_graph=True)
    #
    #     # 提取梯度
    #     gradients = {}
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             gradients[name] = param.grad.clone()
    #
    #     return gradients
    #
    # # 分别获取两个损失的梯度
    # gradients1 = get_gradients(model, loss1)
    # gradients2 = get_gradients(model, loss2)
    #
    # # 分析梯度
    # print("Loss1 梯度分析:")
    # for name, grad in gradients1.items():
    #     print(f"{name}: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
    #
    # print("\nLoss2 梯度分析:")
    # for name, grad in gradients2.items():
    #     print(f"{name}: mean={grad.mean().item():.6f}, std={grad.std().item():.6f}")
    #
    # # 计算梯度相似度
    # cosine_similarities = {}
    # for name in gradients1.keys():
    #     if name in gradients2:
    #         cos_sim = torch.cosine_similarity(
    #             gradients1[name].flatten(), 
    #             gradients2[name].flatten(), 
    #             dim=0
    #         )
    #         cosine_similarities[name] = cos_sim.item()
    #
    # print("\n梯度余弦相似度:")
    # for name, sim in cosine_similarities.items():
    #     print(f"{name}: {sim:.4f}")
    #

        

       
if __name__ == "__main__":
    # test_slice()
    # test_tensor()    
    # test_sort()
    # test_kmeans()
    # test_repeat()
    # test_mul()
    # test_corr()
    # test_pca()
    test_multi_grad()
    # test_js()
    # test_cos()
    # test_nor()
    # test_transfer()
    # test_pairwise()
    # test_embedding()
    # test_zip()
    # test_softmax()