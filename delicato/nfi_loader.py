from collections import defaultdict
from pathlib import Path
import glob
import os

def clarify_name(path_string: str, subfolder_name: str):
  name = Path(path_string).name
  name = name.split('.')[0]

  signer = subfolder_name + '_' + name[:3]
  intended_person = subfolder_name + '_' + name[5:]
  id_ = subfolder_name + '_' + name[3:5]
  type_ = 'genuine' if signer == intended_person else 'forge' 
  return {
    'signer': signer,
    'intended_person': intended_person,
    'local_id': id_,
    'type': type_
  }

def load_all(path: Path):
  path = path / 'Dataset'
  path_string = str(path)

  # get all folder in the path
  subfolders = glob.glob(path_string + '/*/')
  print(subfolders)


  geniune_files = defaultdict(list)
  forge_files = defaultdict(list)

  for subfolder in subfolders:
    subfolder_name = os.path.basename(os.path.normpath(subfolder))
    files = glob.glob(subfolder + '**/*.png')
    print(subfolder, len(files))
    for f in files:
      # get basename of path
      description = clarify_name(f, subfolder_name)
      if description['type'] == 'genuine':
        geniune_files[description['intended_person']].append({
          'path': f,
          'local_id': description['local_id']
        })
      else:
        forge_files[description['intended_person']].append({
          'path': f,
          'local_id': description['local_id']
        })
    
  return geniune_files, forge_files