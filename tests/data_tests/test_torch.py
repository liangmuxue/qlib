import torch 
import torch.nn as nn

def test_tensor():
    y = torch.tensor([31,-3],dtype=torch.float).cuda()
    n = y.where(y > 30.0 ,y.log())   
    print(n)
    
def test_embedding():
    emb = nn.Embedding(5, 3)
    t = torch.Tensor(["600004", "600006", "600007"])
    r = emb(t)                 
    print(r)
    
if __name__ == "__main__":
    # test_tensor()    
    test_embedding()