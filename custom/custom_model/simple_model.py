import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim  as optim
import numpy as np
from losses.crf_loss import CrfLoss

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        self.softmax = nn.LogSoftmax(1)
        
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        # out = self.hidden2(out)
        # out = self.softmax(out)
        out =self.predict(out)
        return out
    
class Trainer():    
    
    def __init__(self,n_input=2,n_hidden=20,n_output=2,batch_size=128,n_epochs=100):
        
        self.device = torch.device("cuda:0")
        # self.device = torch.device("cpu")
        self.model = Net(n_input,n_hidden,n_output)   
        # He weight initialisation
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')        
        
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion = CrfLoss(num_classes=n_output,)     
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5, eta_min=0, last_epoch=-1)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
    
    def build_dataset(self,X_total,Y_total,split):
        
        X_train = X_total[split[0]]
        X_valid = X_total[split[1]]
        y_train = Y_total[split[0]]
        y_valid = Y_total[split[1]]
        
        X_train_tensor = torch.from_numpy(X_train).type(torch.FloatTensor)
        y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)
        X_valid_tensor = torch.from_numpy(X_valid).type(torch.FloatTensor)
        y_valid_tensor = torch.from_numpy(y_valid).type(torch.LongTensor)
                
        print(X_train_tensor.shape, y_train_tensor.shape)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=2)
        valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=2)
                    
    def train(self):    
        g_epoch = 0
        for epoch in range(self.n_epochs):
            
            self.model.train()
            running_loss = 0.0
            valid_loss = 0.0
        
            for batch_idx, data in enumerate(self.train_loader):
                # get inputs and labels
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
        
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # zero the parameter gradients
                self.optimizer.zero_grad()                
                loss.backward()
                self.optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                
                # if (batch_idx+1) % 10000 == 0 or g_epoch<epoch:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch+1, self.n_epochs, batch_idx+1, len(self.train_loader), loss.item()))
            
            self.scheduler.step()      
            
                
            self.model.eval()
            for batch_idx, data in enumerate(self.valid_loader):
                # get inputs and labels
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
        
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                # print statistics
                valid_loss += loss.item()            
            
            print('Epoch [{}/{}], Train Loss: {:.4f},Valid Loss: {:.4f}'.format(epoch+1, self.n_epochs, running_loss,valid_loss))
                