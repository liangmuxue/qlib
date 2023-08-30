from darts.models import RNNModel, ExponentialSmoothing, BlockRNNModel
from darts.models.forecasting.tcn_model import TCNModel
from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.likelihood_models import Likelihood, QuantileRegression
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.data.training_dataset import (
    MixedCovariatesTrainingDataset
)
from darts.models.forecasting.tft_submodels import (
    get_embedding_size,
)
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + np.cos(np.pi*t/t_max))/2
    
    return scheduler

class CyclicLR(_LRScheduler):
    
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]
    
class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

class LSTMReg(nn.Module):
    """lstm回归问题"""
    
    def __init__(self, input_dim, seq_len, output_dim,hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.fc1 = nn.Linear(input_dim*seq_len, hidden_dim)  # notice input shape
        self.fc2 = nn.Linear(hidden_dim,1)        
    
    def forward(self, x):
        x = x.reshape((-1, self.input_dim * self.seq_len))
        reg = nn.Sequential(
                    self.fc1,
                    nn.ReLU(),
                    self.fc2,
                )                   
        output = reg(x)
        return output                
 
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
       
    
class ClassifierTrainer():  
      
    def __init__(self,train_ds,valid_ds,input_dim=2):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.input_dim = input_dim

    def create_loaders(self,train_ds, valid_ds, bs=512, jobs=0):
        train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
        valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
        return train_dl, valid_dl
    
    def training(self):
        input_dim = self.input_dim  
        hidden_dim = 64
        layer_dim = 3
        output_dim = 4
        seq_dim = 128
        
        lr = 0.0005
        n_epochs = 1000
        train_dl, valid_dl = self.create_loaders(self.train_ds, self.valid_ds)
        iterations_per_epoch = len(train_dl)
        best_acc = 0
        patience, trials = 100, 0
        
        model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.RMSprop(model.parameters(), lr=lr)
        sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))
        
        print('Start model training')
        
        for epoch in range(1, n_epochs + 1):
            
            for i, (x_batch, y_batch) in enumerate(train_dl):
                x_batch = x_batch.float()
                model.train()
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                sched.step()
                opt.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                opt.step()
            
            model.eval()
            correct, total = 0, 0
            for x_val, y_val in valid_dl:
                x_val, y_val = [t.cuda() for t in (x_val, y_val)]
                x_val = x_val.float()
                out = model(x_val)
                preds = F.log_softmax(out, dim=1).argmax(dim=1)
                total += y_val.size(0)
                correct += (preds == y_val).sum().item()
            
            acc = correct / total
        
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}')
        
            if acc > best_acc:
                trials = 0
                best_acc = acc
                torch.save(model.state_dict(), 'best.pth')
                print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            else:
                trials += 1
                if trials >= patience:
                    print(f'Early stopping on epoch {epoch}')
                    break    
        