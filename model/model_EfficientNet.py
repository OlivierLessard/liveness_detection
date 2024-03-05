import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(1000, 128)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(128, 32)        # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.view(-1, 1000)
        # FC-1, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        # x = nn.BatchNorm2d(x)
        # FC-2, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        y1o_softmax = F.softmax(x, dim=1)
        return x, y1o_softmax