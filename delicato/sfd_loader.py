from pathlib import Path
import shutil

if __name__ == '__main__':
  in_dir = Path() / 'data' / 'sfd' / 'Train'

  # folders
  folders = list(in_dir.glob('*/'))
  print(len(folders))
  for folder in folders:
    local_id = 'sfd_' + folder.name

    all_files = list(folder.glob('*'))
    for file in all_files:
      if 'original' in file.name or '-G-' in file.name:
        type = 'genuine'
      elif 'forgeries' in file.name or '-F-' in file.name or 'forge' in file.name:
        type = 'forged'
      else:
        raise ValueError(f'Unknown type for {file.name}')

      out_path = Path() / 'data' / 'sfd_restructured' / local_id / type / file.name
      out_path.parent.mkdir(exist_ok=True, parents=True)
      shutil.copy(file, out_path)