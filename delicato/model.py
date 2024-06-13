import torch
import torchvision
from torchvision.models import ResNet34_Weights
from torch import linalg as LA
import torch.nn as nn

class EuclideanDistance(torch.nn.Module):
  def __init__(self):
    super(EuclideanDistance, self).__init__()

  def forward(self, a, b):
    diff = a - b
    return LA.norm(diff, dim=1)
  

class Blobnet(torch.nn.Module):
  def __init__(self):
    super(Blobnet, self).__init__()
    self.base_model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
    self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    n_ftrs = self.base_model.fc.in_features
    self.base_model.fc = nn.Linear(n_ftrs, 256)
    self.dist = EuclideanDistance()

  def forward(self, a, b):
    r1 = self.base_model(a)
    r2 = self.base_model(b)
    d = self.dist(r1, r2)
    return d


class Signet(torch.nn.Module):
  def __init__(self):
    super(Signet, self).__init__()
    

    self.features = nn.Sequential(
      nn.Conv2d(1, 96, kernel_size=11, stride=1),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      
      nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
      nn.ReLU(),
      nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Dropout(p=0.3),
      
      nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1) ,
      nn.ReLU(),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Dropout(p=0.3),
      nn.Flatten()
    )

    self.fully_connected = nn.Sequential(
      torch.nn.Linear(17*25*256, 1024),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=0.5),

      torch.nn.Linear(1024, 128),
      #torch.nn.ReLU(),
    )

    self.dist = EuclideanDistance()

  def forward(self, a, b):
    r1 = self.features(a)
    r1 = self.fully_connected(r1)

    r2 = self.features(b)
    r2 = self.fully_connected(r2)

    d = self.dist(r1, r2)
    return d