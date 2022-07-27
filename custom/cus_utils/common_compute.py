from visdom import Visdom
import torch
import numpy as np

from torch import nn
                
def mae_comp(input,target):
    loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    loss = loss_fn(input.float(), target.float())
    print(loss)
    return loss


def np_qcut(arr, q):
    """ 实现类似pandas的qcut功能"""

    res = np.zeros(arr.size)
    # nanµÄ½á¹û²»²ÎÓë¼ÆËã
    na_mask = np.isnan(arr)
    res[na_mask] = np.nan
    x = arr[~na_mask]
    sorted_x = np.sort(x)
    idx = np.linspace(0, 1, q+1) * (sorted_x.size - 1)
    pos = idx.astype(int)
    fraction = idx % 1
    a = sorted_x[pos]
    b = np.roll(sorted_x, shift=-1)[pos]
    bins = a + (b - a) * fraction
    bins[0] -= 1 
    
    res[~na_mask] = np.digitize(x, bins, right=True)
    return res

if __name__ == "__main__":
    # test_normal_vis()
    input = torch.randn(3, 5)
    target = torch.randn(3, 5)
    mae_comp(input,target)
    
       
    