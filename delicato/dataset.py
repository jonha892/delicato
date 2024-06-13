
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2, functional as F


init_transforms = v2.Compose([
    v2.Grayscale(),
    v2.Resize((155, 220)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    F.invert,
    v2.Normalize((0.0907,), (0.1941,))
])

def get_random_subset(dataset, fraction):
    dataset_size = len(dataset)
    subset_size = int(fraction * dataset_size)
    indices = random.sample(range(dataset_size), subset_size)
    subset = Subset(dataset, indices)
    return subset

class BlobDataset(Dataset):
    def __init__(self, df, transforms=init_transforms):
        self.df = df
        self.transforms = transforms
        

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_a = Image.open(row['path_a'])
        image_a = self.transforms(image_a)

        image_b = Image.open(row['path_b'])
        image_b = self.transforms(image_b)

        label = 0 if row['is_genuine'] else 1 # distance
        label = torch.tensor(label, dtype=torch.float32)

        return image_a, image_b, label
    
    
class TripletDataset(Dataset):
    def __init__(self, df, transforms=init_transforms):
        self.df = df
        self.transforms = transforms
        

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        anchor = Image.open(row['anchor_path'])
        anchor = self.transforms(anchor)

        positive = Image.open(row['positive_path'])
        positive = self.transforms(positive)

        negative = Image.open(row['negative_path'])
        negative = self.transforms(negative)

        return anchor, positive, negative