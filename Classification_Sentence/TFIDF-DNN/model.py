import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15000,10000)
        self.fc2 = nn.Linear(10000,128)
        self.fc3 = nn.Linear(128,1)
        
        self.dropout5 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x         