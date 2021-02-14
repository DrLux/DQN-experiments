import os

def check_dir(dirpath):
  if not os.path.isdir(dirpath):
    os.mkdir(dirpath)