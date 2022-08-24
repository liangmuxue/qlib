import torch 

def test_tensor():
    y = torch.tensor([31,-3],dtype=torch.float).cuda()
    n = y.where(y > 30.0 ,y.log())   
    print(n)
    
if __name__ == "__main__":
    test_tensor()    