from pathlib import Path
import shutil

if __name__ == '__main__':
  in_base_path = Path() / '..' / 'data' / 'ICDAR' / 'train'
  out_base_path = Path() / '..' / 'data' / 'icdar_restructured'
  out_base_path.mkdir(exist_ok=True, parents=True)

  # get all folder in the base path with a name of three digits
  folders = list(in_base_path.glob('[0-9][0-9][0-9]'))
  print(f'Found {len(folders)} folders')

  for folder in folders:
    base_name = f'icdar_{folder.name}'

    folder_path = out_base_path / base_name
    (folder_path / 'genuine').mkdir(exist_ok=True, parents=True)
    (folder_path / 'forged').mkdir(exist_ok=True, parents=True)
    
    genuine_files = list(folder.glob('*'))
    forged_files = list( Path(str(folder) + '_forg').glob('*'))

    print(f'Folder: {folder.name} Genuine: {len(genuine_files)} Forged: {len(forged_files)}')
    if len(forged_files) == 0:
      break
    
    for file in genuine_files:
      save_path = folder_path / 'genuine' / file.name
      shutil.copy(file, save_path)

    for file in forged_files:
      save_path = folder_path / 'forged' / file.name
      shutil.copy(file, save_path)