from pathlib import Path

def clarify_name(path_string: str, subfolder_name: str):
  pass

def load_all(path: Path):
  files = list(path.glob('**/*.png'))

  result = []
  for file in files:
    # get name of parent folder
    local_id = file.parent.name

    file_name = file.name
    is_genuine = file_name.startswith('original')
    #print(file_name, is_genuine)
    result.append({
      'path': file,
      'local_id': local_id,
      'is_genuine': is_genuine
    })
  return result