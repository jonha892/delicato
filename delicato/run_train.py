from pathlib import Path

import torch
from torchvision.transforms import v2, functional as F
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR

from train_test_split import init_split, generate_train_examples
from train import ContrastiveLoss, fit
from dataset import BlobDataset
from model import Signet, Blobnet

if __name__ == '__main__':
  data_path_cedar = Path() / 'data' / 'blob' 
  train_df_cedar, val_df_cedar, _ = init_split(data_path_cedar, train_size=0.8, val_size=0.2, seed=77)
  train_df_cedar = generate_train_examples(train_df_cedar)
  val_df_cedar = generate_train_examples(val_df_cedar)

  data_path_icdar = Path() / 'data' / 'icdar_restructured' 
  train_df_icdar, val_df_icdar, _ = init_split(data_path_icdar, train_size=0.8, val_size=0.2, seed=77)
  train_df_icdar = generate_train_examples(train_df_icdar)
  val_df_icdar = generate_train_examples(val_df_icdar)


  train_transforms = v2.Compose([
    v2.Grayscale(),
    v2.Resize((155, 220)),
    v2.RandomRotation(10),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    F.invert,
    v2.Normalize((0.0907,), (0.1941,))
  ])
  
  train_ds_cedar = BlobDataset(train_df_cedar, transforms=train_transforms)
  val_ds_cedar = BlobDataset(val_df_cedar)

  train_ds_icdar = BlobDataset(train_df_icdar, transforms=train_transforms)
  val_ds_icdar = BlobDataset(val_df_icdar)

  print(len(train_ds_cedar), len(train_ds_icdar))
  

  #train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
  #val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
  train_datasets = [
    (train_ds_cedar, 0.1),
    #(train_ds_icdar, 1)
  ]
  val_datasets = [
    (val_ds_cedar, 'cedar', 1),
    #(val_ds_icdar, 'icdar', 1)
  ]

  #model = Signet()
  model = Blobnet()

  optimizer = Adam(model.parameters(), lr=0.001)
  #optimizer = RMSprop(model.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9, eps=1e-8)

  scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

  #critertion = contrastiveLoss
  critertion = ContrastiveLoss(alpha=1, beta=1, margin=1).to(torch.device('cuda'))
  epochs = 20

  fit(model, train_datasets, val_datasets, optimizer, scheduler, critertion, epochs, device=torch.device('cuda'))
