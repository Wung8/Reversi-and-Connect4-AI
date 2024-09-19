import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=27, kernel_size=3, stride=1),
            nn.ReLU())
        
        self.fc1 = nn.Linear(4*5*27+6*7*3,100)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(100,1)

    def forward(self,x):
        x1 = self.layer1(x)

        x1 = torch.flatten(x1)

        x = torch.cat((torch.flatten(x),x1))

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class DataLoader(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = []
        for item in data:
            board,val = item[:2]
            val = torch.tensor(val).type(torch.float)
            self.data.append((board,val))
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]
        





