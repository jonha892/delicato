from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset import get_random_subset

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_anchor_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_anchor_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.clamp(distance_anchor_positive - distance_anchor_negative + self.margin, min=0.0)
        return loss.mean()



def train_embedding(
        model,
        train_datasets,
        criterion,
        optimizer,
        scheduler,
        n_epochs=10,
        device='cuda'
):
    model.to(device)
    history = {'train_loss': []}
    print("------------------------Embedding Training--------------------------")

    best_loss = np.inf

    for epoch in range(1, n_epochs + 1):
        t0 = datetime.now()
        print(f"Beginning Epoch {epoch}/{n_epochs}...")
        train_loss = []
        model.train()

        subsets = [ get_random_subset(d, w) for d, w in train_datasets ]
        super_set = ConcatDataset(subsets)
        loader = DataLoader(super_set, batch_size=32, shuffle=True, num_workers=4)


        for i, data in (enumerate(tqdm(loader), 0)):
            anchor, positive, negative = data
            anchor = anchor.to(device=device)
            positive = positive.to(device=device)
            negative = negative.to(device=device)
            
            optimizer.zero_grad()
            anchor_embeddings = model(anchor)  
            positive_embeddings = model(positive)  
            negative_embeddings = model(negative) 

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)  # Changed `triplet_loss` to `loss_fn`
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())  # Added the loss value to `train_loss` list

        avg_epoch_loss = np.mean(train_loss)
        history['train_loss'].append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            print(f'previous best loss: {best_loss:.3f} new best loss: {avg_epoch_loss:.3f} saving model...')
            best_loss = avg_epoch_loss
            to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }
            torch.save(to_save, Path() / 'models' / 'best_embedding_model.pth')

        dt = datetime.now() - t0
        print(f'Epoch: {epoch}\tTrain Loss: {avg_epoch_loss}\tDuration: {dt}\n')

        if best_loss <= 1e-6:
            break

        # Tracking accuracy and loss in each epoch for plot
        scheduler.step()
    
    return history




def train(model, input1, input2, outputs, optimizer, loss_fn):
    # Set the model to training mode
    model.train()
    # Zero the gradients
    optimizer.zero_grad()
    # Compute the model's predictions
    predictions = model(input1, input2)
    # flatten the predictions
    predictions = torch.flatten(predictions)

    # Compute the loss
    #print(predictions.shape, outputs.shape)
    loss = loss_fn(predictions, outputs)
    # Compute the gradients
    loss.backward()
    # Update the weights
    optimizer.step()
    return loss, predictions

def train_classify(
  model,
  train_datasets,
  val_datasets,
  criterion,
  optimizer,
  scheduler,
  n_epochs=10,
  device='cuda'
):
  model.to(device)
  history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}
  print("------------------------Classify Training--------------------------")

  best_acc = 0.0

  for epoch in range(1, n_epochs + 1):
    t0 = datetime.now()
    print(f"Beginning Epoch {epoch}/{n_epochs}...")
    train_loss = []
    train_acc = []
    model.train()

    subsets = [ get_random_subset(d, w) for d, w in train_datasets ]
    super_set = ConcatDataset(subsets)
    print(f'Length of train super set: {len(super_set)}')
    train_loader = DataLoader(super_set, batch_size=32, shuffle=True, num_workers=4)

    for i, data in enumerate(tqdm(train_loader), 0):
        inputs1, inputs2, targets = data
        #print(inputs1.shape, inputs2.shape, targets.shape)
        inputs1 = inputs1.to(device=device)
        inputs2 = inputs2.to(device=device)
        targets = targets.to(device=device)

        loss, predictions = train(model, inputs1, inputs2, targets, optimizer, criterion)
        train_loss.append(loss.item())
        accuracy = (predictions.round() == targets).float().mean().item()
        train_acc.append(accuracy)

    valid_loss = []
    valid_acc = []
    model.eval()

    val_super_ds = ConcatDataset([ get_random_subset(ds, w) for ds, _, w in val_datasets ])
    val_super_loader = DataLoader(val_super_ds, batch_size=32, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_super_loader), 0):
            inputs1, inputs2, targets = data
            inputs1 = inputs1.to(device=device)
            inputs2 = inputs2.to(device=device)
            targets = targets.to(device=device)

            output = model(inputs1, inputs2)
            output = torch.flatten(output)
            loss = criterion(output, targets)
            valid_loss.append(loss.item())
            accuracy = (output.round() == targets).float().mean().item()
            valid_acc.append(accuracy)

    dt = datetime.now() - t0
    epoch_train_loss = np.mean(train_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_valid_loss = np.mean(valid_loss)
    epoch_valid_acc = np.mean(valid_acc)

    # Tracking accuracy and loss in each epoch for plot
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['valid_loss'].append(epoch_valid_loss)
    history['valid_acc'].append(epoch_valid_acc)

    if epoch_valid_acc > best_acc:
        print(f'previous best acc: {best_acc:4f} new best acc: {epoch_valid_acc:4f} saving model...')
        best_acc = epoch_valid_acc
        to_save = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'history': history,
        }
        torch.save(to_save, Path() / 'models' / 'best_classify_model.pth')
    print('Epoch: {}\t\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\nDuration: {}\tValid Loss: {:.4f}\tValid Accuracy: {:.4f}\n'.format(
          epoch, epoch_train_loss, epoch_train_acc, dt, epoch_valid_loss, epoch_valid_acc
    ))

    scheduler.step()
  
  return history