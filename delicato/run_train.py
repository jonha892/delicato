from pathlib import Path

import torch
from torch.nn import BCELoss
from torchvision.transforms import v2, functional as F
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR

from train_test_split import generate_classify_train_examples, generate_triplet_train_examples, init_id_split, init_split, generate_train_examples
from train import ContrastiveLoss, fit
from train_triplet import train_embedding, train_classify, TripletLoss
from dataset import BlobDataset, TripletDataset, ClassifyDataset
from model import Signet, Blobnet, SiameseResNet, LogisticSiameseRegression


def main_1():
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
    (train_ds_icdar, 1)
  ]
  val_datasets = [
    (val_ds_cedar, 'cedar', 0.1),
    (val_ds_icdar, 'icdar', 1)
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



def main_train_embedding(train_folders, train_folders_icdar, train_folders_sfd, device):
  train_df_cedar = generate_triplet_train_examples(train_folders)
  train_df_icdar = generate_triplet_train_examples(train_folders_icdar)
  train_df_sfd = generate_triplet_train_examples(train_folders_sfd)

  train_transforms = v2.Compose([
    v2.Resize((200, 300)),
    v2.RandomRotation((-5, 10)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
  ])

  train_ds = TripletDataset(train_df_cedar, transforms=train_transforms)
  train_ds_icdar = TripletDataset(train_df_icdar, transforms=train_transforms)
  train_ds_sfd = TripletDataset(train_df_sfd, transforms=train_transforms)

  train_datasets = [
    #(train_ds, 0.1),
    #(train_ds_icdar, 1.0),
    (train_ds_sfd, 1.0)
  ]

  model = SiameseResNet()

  optimizer = Adam(model.parameters(), lr=0.001)
  scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

  criterion = TripletLoss().to(device)
  n_epochs = 12

  train_embedding(
    model,
    train_datasets,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=n_epochs,
    device=device
  )

  return train_folders, val_folders

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_transforms = v2.Compose([
    v2.Resize((200, 300)),
    v2.RandomRotation((-5, 10)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True)
  ])

  val_transforms = v2.Compose([
    v2.Resize((200, 300)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
  ])


  data_path_cedar = Path() / 'data' / 'blob'
  train_folders, val_folders, _ = init_id_split(data_path_cedar, train_size=0.8, seed=77)
  classify_train_df_cedar = generate_classify_train_examples(train_folders)
  classify_val_df_cedar = generate_classify_train_examples(val_folders)
  train_ds_cedar = ClassifyDataset(classify_train_df_cedar, train_transforms)
  val_ds_cedar = ClassifyDataset(classify_val_df_cedar, val_transforms)
  
  data_path_icdar = Path() / 'data' / 'icdar_restructured'
  train_folders_icdar, val_folders_icdar, _ = init_id_split(data_path_icdar, train_size=0.8, seed=77)
  classify_train_df_icdar = generate_classify_train_examples(train_folders_icdar)
  classify_val_df_icdar = generate_classify_train_examples(val_folders_icdar)
  train_ds_icdar = ClassifyDataset(classify_train_df_icdar, train_transforms)
  val_ds_icdar = ClassifyDataset(classify_val_df_icdar, val_transforms)

  data_path_sfd = Path() / 'data' / 'sfd_restructured'
  train_folders_sfd, val_folders_sfd, _ = init_id_split(data_path_sfd, train_size=0.7, seed=77)
  classify_train_df_sfd = generate_classify_train_examples(train_folders_sfd)
  classify_val_df_sfd = generate_classify_train_examples(val_folders_sfd)
  train_ds_sfd = ClassifyDataset(classify_train_df_sfd, train_transforms)
  val_ds_sfd = ClassifyDataset(classify_val_df_sfd, val_transforms)

  print(f'train cedar: {len(train_ds_cedar)} icdar: {len(train_ds_icdar)} sfd: {len(train_ds_sfd)}')
  print(f'val   cedar: {len(val_ds_cedar)} icdar: {len(val_ds_icdar)} sfd: {len(val_ds_sfd)}')

  main_train_embedding(train_folders, train_folders_icdar, train_folders_sfd, device)
  
  checkpoint = torch.load('models/best_embedding_model.pth')
  print(f'loading model with best loss: {checkpoint["history"]["train_loss"][-1]}')
  embedding_model = SiameseResNet()
  embedding_model.load_state_dict(checkpoint['model_state_dict'])

  model = LogisticSiameseRegression(embedding_model)

  optimizer = Adam(model.parameters(), lr=0.001)
  scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
  criterion = BCELoss().to(device)


  train_datasets = [
    #(train_ds_cedar, 0.1),
    #(train_ds_icdar, 1.0)
    (train_ds_sfd, 1.0)
  ]
  
  val_datasets = [
    #(val_ds_cedar, 'cedar', 1),
    #(val_ds_icdar, 'icdar', 1)
    (val_ds_sfd, 'sfd', 1)
  ]

  n_epochs = 10

  history = train_classify(
    model,
    train_datasets,
    val_datasets,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=n_epochs,
    device=device
  )