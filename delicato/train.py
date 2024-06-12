from time import time

from metrics import find_best_accuracy, accuracy
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset import get_random_subset

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

def train(model, loader, optimizer, criterion, device):
  model.train()
  
  running_loss = torch.tensor(0.0).to(device)

  for i, (inputs_a, inputs_b, labels) in enumerate(tqdm(loader)):
    inputs_a = inputs_a.to(device)
    inputs_b = inputs_b.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    distance = model(inputs_a, inputs_b)
    loss = criterion(distance, labels)

    loss.backward()
    optimizer.step()
    running_loss += loss.item() * inputs_a.size(0)
  
  return running_loss

@torch.no_grad()
def eval(model, loader, criterion, device, test_threshold=None):
  model.eval()   
  running_loss = torch.tensor(0.0).to(device)

  distances = []
  all_labels = []
  for i, (inputs_a, inputs_b, labels) in enumerate(tqdm(loader)):
    inputs_a = inputs_a.to(device)
    inputs_b = inputs_b.to(device)
    labels = labels.to(device)


    distance = model(inputs_a, inputs_b)
    loss = criterion(distance, labels)

    running_loss += loss.item() * inputs_a.size(0)
    
    distances.append(distance)
    all_labels.append(labels)

  distances = torch.cat(distances)
  labels = torch.cat(all_labels)
  best_acc, t = find_best_accuracy(distances, labels)

  if test_threshold is None:
    return running_loss, best_acc, -1, t

  test_acc = accuracy(distances, test_threshold, labels)
  return running_loss, best_acc, test_acc, t
  


def fit(
    model,
    train_datasets, # list of tuples of ds and weight
    validation_datasets, # list of tuples of ds and name
    optimizer,
    scheduler,
    criterion,
    epochs,
    save_dir='models/',
    device=torch.device('cpu')
):
  model.to(device)
  criterion.to(device)

  val_loaders = [ (DataLoader(ds, batch_size=32, shuffle=False, num_workers=4), name) for ds, name, _ in validation_datasets ]
  val_super_ds = ConcatDataset([ get_random_subset(ds, w) for ds, _, w in validation_datasets ])
  val_super_loader = DataLoader(val_super_ds, batch_size=32, shuffle=False, num_workers=4)

  best_com_acc = 0
  for e in range(epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        subsets = [ get_random_subset(d, w) for d, w in train_datasets ]
        super_set = ConcatDataset(subsets)

        print(f'Length of super_set: {len(super_set)}')
        loader = DataLoader(super_set, batch_size=32, shuffle=True, num_workers=4)

        running_loss = train(model, loader, optimizer, criterion, device)
        print(f'epoch {str(e).ljust(3)} {phase.ljust(7)} average loss: {running_loss / len(loader.dataset):.3f}')

        scheduler.step()
      else:
        com_loss, best_acc, _, com_threshold = eval(model, val_super_loader, criterion, device)
        print(f'epoch {str(e).ljust(3)} {phase.rjust(7)}-combined average loss: {com_loss / len(val_super_loader.dataset):.3f} accuracy: {best_acc:.3f} threshold: {com_threshold:.3f}')

        if best_acc > best_com_acc:
          best_com_acc = best_acc
          to_save = {
            'model': model.state_dict(),
            'acc': best_com_acc,
            'threshold': com_threshold 
          }
          torch.save(to_save, save_dir + f'checkpoint_{time()}.pt')
          print(f'Saving model with combined accuracy: {best_acc} and threshold: {com_threshold}')

        for loader, name in val_loaders:
          running_loss, best_acc, test_acc, best_threshold = eval(model, loader, criterion, device, test_threshold=com_threshold)
          print(f'epoch {str(e).ljust(3)} {phase.ljust(7)}-{name} average loss: {running_loss / len(loader.dataset):.3f} best_accuracy: {best_acc:.3f} test_acc(combined threshold): {test_acc:.3f} threshold: {best_threshold:.3f}')