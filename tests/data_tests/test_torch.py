import torch 
import torch.nn as nn

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
             
if __name__ == "__main__":
    # test_tensor()    
    test_sort()
    # test_embedding()
    # test_zip()
    # test_softmax()