from pathlib import Path
from time import time

import torch
from torchvision.transforms import v2, functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.notebook import tqdm
import torch.utils.benchmark as benchmark

#from train_test_split import init_split, generate_train_examples
#from dataset import BlobDataset
#from model import Signet

def contrastiveLoss(distance, label, margin=1, alpha=1.0, beta=1.0):
 # left = alpha * (1 - label) * torch.pow(distance, 2)
  left = alpha * (1 - label) * distance.pow(2)

  #m = torch.clamp(margin - distance, min=0.0)
  #right = beta * label * torch.pow(m, 2)

  m = torch.clamp(margin - distance, min=0.0)
  right = beta * label * m.pow(2)

  combined = left + right
  sum = combined.sum()
  return sum

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, alpha, beta, margin):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, distance, y):
        '''
        Shapes:
        -------
        x1: [B,C]
        x2: [B,C]
        y: [B,1]

        Returns:
        --------
        loss: [B,1]]
        '''
        left = self.alpha * (1-y) * distance**2
        right = self.beta * y * (torch.max(torch.zeros_like(distance), self.margin - distance)**2)
        loss = left + right
        loss = torch.mean(loss, dtype=torch.float)
        return loss

def fit(
    model,
    loaders,
    optimizer,
    criterion,
    epochs,
    device=torch.device('cpu')
):
  model.to(device)
  criterion.to(device)

  for e in range(epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = torch.tensor(0.0).to(device)

      loader = loaders[phase]
      for inputs_a, inputs_b, labels in tqdm(loader):
        t0 = time()
        inputs_a = inputs_a.to(device)
        inputs_b = inputs_b.to(device)
        labels = labels.to(device)
        print(f'to device time: {time() - t0}')

        t1 = time()
        optimizer.zero_grad()
        print(f'zero grad time: {time() - t1}')

        with torch.set_grad_enabled(phase == 'train'):
          t0 = time()
          distance = model(inputs_a, inputs_b)
          t1 = time()
          print(f'forward time: {t1 - t0}')
          loss = criterion(distance, labels)
          print(f'loss time: {time() - t1}')

          t0 = time()
          if phase == 'train':
            loss.backward()
            optimizer.step()
          t1 = time()
          print(f'backward time: {t1 - t0}')

          #t0 = time()
          #running_loss += loss.item() * inputs_a.size(0)
          #t1 = time()
          #print(f'backward time: {t1 - t0}')
      print(f'epoch {str(e).ljust(3)} {phase.ljust(7)} average loss: {running_loss / len(loader.dataset)}')



if __name__ == '__main__':
  data_path = Path() / '..' / 'data' / 'blob' 
  init_train_df, val_df, test_df = init_split(data_path, train_size=0.7, val_size=0.15, test_size=0.15, seed=77)

  train_df = generate_train_examples(init_train_df)
  val_df = generate_train_examples(val_df)


  train_transforms = v2.Compose([
    v2.Grayscale(),
    v2.Resize((155, 220)),
    v2.RandomRotation(10),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    F.invert,
    v2.Normalize((0.0907,), (0.1941,))
  ])
  
  train_ds = BlobDataset(train_df, transforms=train_transforms)
  val_ds = BlobDataset(val_df)

  train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
  val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
  loaders = { 'train': train_dl, 'val': val_dl }

  model = Signet()

  optimizer = Adam(model.parameters(), lr=0.001)
  #critertion = contrastiveLoss
  critertion = ContrastiveLoss(alpha=1, beta=1, margin=1).to(torch.device('cuda'))
  epochs = 15

  fit(model, loaders, optimizer, critertion, epochs, device=torch.device('cuda'))