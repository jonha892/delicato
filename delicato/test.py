import itertools
from pathlib import Path

import torch
from torchvision.transforms import v2, functional as F
from PIL import Image

from model import LogisticSiameseRegression, SiameseResNet

if __name__ == '__main__':
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  data_path_cedar = Path() / 'data' / 'sign_1'

  genuine_files = list(data_path_cedar.glob('genuine/*'))
  forged_files = list(data_path_cedar.glob('forged/*'))

  combinations = list(itertools.product(genuine_files, forged_files))
  combinations.extend(list(itertools.combinations(genuine_files, 2)))
  
  checkpoint = torch.load(Path() / 'models' / 'best_classify_model.pth')
  model = LogisticSiameseRegression(SiameseResNet())
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)

  transformations =v2.Compose([
    v2.Grayscale(),
    v2.Resize((155, 220)),
    #v2.RandomRotation(10),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    F.invert,
    v2.Normalize((0.21,), (0.1941,))
  ])
  

  for path_a, path_b in combinations:
    #print(path_a.parent.name)
    is_genuine = 1 if path_a.parent.name == path_b.parent.name and path_a.parent.name == 'genuine' else 0
    image_a = transformations(Image.open(path_a))
    image_b = transformations(Image.open(path_b))

    image_a = image_a.unsqueeze(0).to(device)
    image_b = image_b.unsqueeze(0).to(device)

    output = model(image_a, image_b)
    output = torch.flatten(output)
    #output = torch.round(output)

    print(f' path_a: {path_a} path_b: {path_b}')
    print(f'Predicted: {output.item():.3f} Actual: {is_genuine}')