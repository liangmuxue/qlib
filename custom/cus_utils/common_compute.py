from visdom import Visdom
import torch
import numpy as np

from torch import nn
                
                
            
def mae_comp(input,target):
    loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    loss = loss_fn(input.float(), target.float())
    print(loss)
    return loss
    
if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    mae_comp(input,target)
    
       
    