import itertools
from pathlib import Path
import random

from sklearn.model_selection import train_test_split
import pandas as pd

def process_ids(ids):
    items = []
    for i in ids:
      genuine_files = list(i.glob('genuine/*.png'))
      items.extend([{ 'path': file, 'id': i.name, 'is_genuine': True } for file in genuine_files])
      forged_files = list(i.glob('forged/*.png'))
      items.extend([{ 'path': file, 'id': i.name, 'is_genuine': False } for file in forged_files])
    return pd.DataFrame(items)

def init_split(path: Path, train_size: float, val_size = None, test_size = None, seed: int = 42):
  # get folders in the path
  folders = list(path.glob('*/'))
  if len(folders) == 0:
    raise ValueError(f'No folders found in the path {path}')

  train_ids, val_ids = train_test_split(folders, train_size=train_size, random_state=seed)
  
  test_df = None
  if val_size and test_size:
    val_size = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(val_ids, train_size=val_size, random_state=seed)
    print(f'Test: {len(test_ids)} ratio: {len(test_ids) / len(folders)}')
    test_df = process_ids(test_ids) if test_ids else None

  print(f'Train: {len(train_ids)} ratio: {len(train_ids) / len(folders)}')
  print(f'Val: {len(val_ids)} ratio: {len(val_ids) / len(folders)}')
    

  train_df = process_ids(train_ids)
  val_df = process_ids(val_ids)

  return train_df, val_df, test_df

def generate_train_examples(df: pd.DataFrame):
  ids = df['id'].unique()
  
  result = []
  for id in ids:
    rows = df[df['id'] == id]
    #print(f'ID: {id} Genuine: {len(rows[rows["is_genuine"]])} Forged: {len(rows[~rows["is_genuine"]])}')

    # get all combinations of rows
    for i in range(len(rows)):
      for j in range(i, len(rows)):
        row1 = rows.iloc[i]
        row2 = rows.iloc[j]
        result.append({
          'path_a': row1['path'],
          'path_b': row2['path'],
          'is_genuine': row1['is_genuine'] and row1['is_genuine'] == row2['is_genuine'],
          'id': id
        })

  return pd.DataFrame(result)

def init_id_split(path: Path, train_size: float, val_size = None, test_size = None, seed: int = 42):
  # get folders in the path
  folders = list(path.glob('*/'))
  if len(folders) == 0:
    raise ValueError(f'No folders found in the path {path}')
  
  train_folders, val_folders = train_test_split(folders, train_size=train_size, random_state=seed)
  test_folders = []
  if test_size:
    val_size = val_size / (val_size + test_size)
    val_folders, test_folders = train_test_split(val_folders, train_size=val_size, random_state=seed)
    print(f'Test: {len(test_folders)} ratio: {len(test_folders) / len(folders)}')
  return train_folders, val_folders, test_folders


def random_images(folders, type, seed: int = 42):
  paths = []
  for person_folder in folders:
    files = list(person_folder.glob(f'{type}/*'))
    if len(files) == 0:
      raise ValueError(f'Person folder {person_folder} does not have {type} files')
    if len(files) > 7:
      files = random.sample(files, 7)
    paths.extend(files)
  return paths

def generate_triplet_train_examples(folders, seed: int = 42):
  image_paths = random_images(folders, 'forged', seed)

  df = pd.DataFrame(columns=[ 'anchor_path', 'positive_path', 'negative_path', 'id' ])

  for person_folder in folders:
    genuine_files = list(person_folder.glob('genuine/*'))
    forged_files = list(person_folder.glob('forged/*'))
    if len(genuine_files) == 0 or len(forged_files) == 0:
      raise ValueError(f'Person folder {person_folder} does not have genuine or forged files')
    
    additional_images = random.sample(image_paths, 10)
    forged_files.extend(additional_images)

    num_combinations = min(
      (len(genuine_files) * (len(genuine_files) - 1)) // 2,
      len(genuine_files) * len(forged_files),
    )
    #print(f'Person {person_folder.name} Genuine: {len(genuine_files)} Forged: {len(forged_files)} Genuine combinations: {num_combinations} additional: {len(additional_images)}')
    #print(len(list(itertools.combinations(genuine_files, 2))))
    genuine_combinations = random.sample(
      list(itertools.combinations(genuine_files, 2)),
      num_combinations
    )
    forged_combinations = random.sample(
      list(itertools.product(genuine_files, forged_files)),
      num_combinations
    )


    data = []
    for (image_1, image_2), (_, forged) in zip(genuine_combinations, forged_combinations):
      data.append({
        'anchor_path': image_1,
        'positive_path': image_2,
        'negative_path': forged,
        'id': person_folder.name
      })
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
  return df