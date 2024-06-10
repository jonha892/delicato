import torch
from tqdm.notebook import tqdm

def contrastiveLoss(distance, label, margin=1, alpha=1.0, beta=1.0):
  left = alpha * (1 - label) * torch.pow(distance, 2)

  m = torch.clamp(margin - distance, min=0.0)
  right = beta * label * torch.pow(m, 2)
  combined = left + right
  sum = combined.sum()
  return sum

def fit(
    model,
    loaders,
    optimizer,
    criterion,
    epochs,
    device=torch.device('cpu')
):
  model.to(device)

  for e in range(epochs):
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0

      loader = loaders[phase]
      for inputs_a, inputs_b, labels in tqdm(loader):
        inputs_a = inputs_a.to(device)
        inputs_b = inputs_b.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          distance = model(inputs_a, inputs_b)
          loss = criterion(distance, labels)
          print(loss)

          if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs_a.size(0)
      print(f'epoch {str(e).ljust(3)} {phase.ljust(7)} average loss: {running_loss / len(loader.dataset)}')