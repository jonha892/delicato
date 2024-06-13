from datetime import datetime
from pathlib import Path

import tqdm
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
    history = {'train_loss': []}
    print("------------------------Training--------------------------")

    best_loss = np.inf

    for epoch in range(1, n_epochs + 1):
        t0 = datetime.now()
        print(f"Beginning Epoch {epoch}/{n_epochs}...")
        train_loss = []
        model.train()

        subsets = [ get_random_subset(d, w) for d, w in train_datasets ]
        super_set = ConcatDataset(subsets)
        loader = DataLoader(super_set, batch_size=32, shuffle=True, num_workers=4)


        for i, data in tqdm(enumerate(loader, 0)):
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
            epoch_loss += loss.item()

        avg_epoch_loss = np.mean(train_loss)
        history['train_loss'].append(avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }
            torch.save(to_save, Path() / 'models' / 'best_embedding_model.pth')

        dt = datetime.now() - t0
        print('\nEpoch: {}\tTrain Loss: {}\tDuration: {}'.format(epoch, avg_epoch_loss, dt))

        # Tracking accuracy and loss in each epoch for plot
        scheduler.step()
    
    return history