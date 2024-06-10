import torch
import torchvision
from torch import linalg as LA

class EuclideanDistance(torch.nn.Module):
  def __init__(self):
    super(EuclideanDistance, self).__init__()

  def forward(self, a, b):
    diff = a - b
    return LA.norm(diff, dim=1)
  

class Blobnet(torch.nn.Module):
  def __init__(self):
    super(Blobnet, self).__init__()
    self.base_model = torchvision.models.resnet18(pretrained=True)
    self.base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    n_ftrs = self.base_model.fc.in_features
    self.base_model.fc = torch.nn.Linear(n_ftrs, 128)
    self.dist = EuclideanDistance()

  def forward(self, a, b):
    r1 = self.base_model(a)
    r2 = self.base_model(b)
    d = self.dist(r1, r2)
    return d


class Signet(torch.nn.Module):
  def __init__(self):
    super(Signet, self).__init__()

    conv1 = torch.nn.Conv2d(1, 96, kernel_size=11, stride=1)
    norm1 = torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
    pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
    
    conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
    norm2 = torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
    pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
    dropout2 = torch.nn.Dropout(p=0.3)

    conv3 = torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
    
    conv4 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
    #pool4 = torch.nn.MaxPool2d(kernel_size=5, stride=2)
    #pool4 = torch.nn.MaxPool2d(kernel_size=(17, 25), stride=2)
    pool4 = torch.nn.AdaptiveAvgPool2d((1, 1)) # ?
    dropout4 = torch.nn.Dropout(p=0.3)

    self.features = torch.nn.Sequential(
      conv1,
      torch.nn.ReLU(),
      norm1,
      pool1,
      conv2,
      torch.nn.ReLU(),
      norm2,
      pool2,
      dropout2,
      conv3,
      torch.nn.ReLU(),
      conv4,
      torch.nn.ReLU(),
      pool4,
      dropout4
    )

    self.fc1 = torch.nn.Linear(256, 1024)
    # self.fc1 = torch.nn.Linear(217600, 1024)
    self.fc2 = torch.nn.Linear(1024, 128)
    self.dist = EuclideanDistance()

  def forward(self, a, b):
    r1 = self.features(a)
    #print(r1.shape)
    r1 = r1.squeeze()
    #print(r1.shape)
    r1 = self.fc1(r1)
    r1 = self.fc2(r1)

    r2 = self.features(b)
    r2 = r2.squeeze()
    r2 = self.fc1(r2)
    r2 = self.fc2(r2)

    #print(r1.shape, r2.shape)

    d = self.dist(r1, r2)
    return d