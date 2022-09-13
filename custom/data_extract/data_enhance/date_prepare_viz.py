import numpy as np
from cus_utils.tensor_viz import TensorViz
    
def time_ser_data_viz():
    viz_input = TensorViz(env="data_hist") 
    data_file = "/home/qdata/project/qlib/custom/data/aug/test100_all.npy"
    data = np.load(data_file,allow_pickle=True)
    price = data[:,:,-1].reshape(-1)
    price_target = data[:,15:,-1].reshape(-1)
    viz_input.viz_data_hist(price,numbins=20,win="ser20_all",title="ser20_all")
    viz_input.viz_data_hist(price_target,numbins=20,win="tar5_all",title="tar5_all")
    
if __name__ == '__main__':
     time_ser_data_viz()   