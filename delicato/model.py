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
  

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        attention_scores = self.conv(x)
        attention_weights = self.sigmoid(attention_scores)

        # Apply attention to the input feature map
        attended_features = x * attention_weights

        return attended_features


class SiameseResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(SiameseResNet, self).__init__()
        self.baseModel = torchvision.models.resnet18(pretrained=pretrained)

        # Experiment with different spatial sizes based on the image resolution and signature complexity
        self.attention1 = SpatialAttention(in_channels=64)  # Spatial attention for layer 1
        self.attention2 = SpatialAttention(in_channels=128)  # Spatial attention for layer 2

        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)

        out = self.attention1(self.baseModel.layer1(out))  # Applying spatial attention to layer 1
        out = self.attention2(self.baseModel.layer2(out))  # Applying spatial attention to layer 2
        out = self.baseModel.layer3(out)  # No attention for layer 3
        out = self.baseModel.layer4(out)  # No attention for layer 4

        out = self.baseModel.avgpool(out)
        out = torch.flatten(out, 1)
        return out