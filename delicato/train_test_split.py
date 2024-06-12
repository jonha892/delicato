from pathlib import Path

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