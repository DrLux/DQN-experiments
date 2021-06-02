import utils
import datetime
import logging
import json
from pathlib import Path


class CfgMaker():

  def __init__(self):
    # Create initial folders
    self.DEV_LEVEL = "DEVELOPMENT"
    
    if self.DEV_LEVEL == "DEVELOPMENT":
      self.experiment_folder = 'experiments/results/dev'
    else: 
      self.experiment_folder = 'experiments/results/'+str(datetime.datetime.now().strftime('%d-%b-%y %H:%M:%S'))

    utils.make_dir(self.experiment_folder)
    self.dump_cfg()

  def make_cfg_logger(self):
    if self.DEV_LEVEL == "DEVELOPMENT":
      log_level = logging.DEBUG
    elif self.DEV_LEVEL == "DEBUG":
      log_level = logging.DEBUG
    else:
      log_level = logging.INFO

    log_dir = Path(self.experiment_folder) / Path("logdir")
    utils.make_dir(log_dir)
    self.cfg_logger = {
      'name'      : "cfg_logger",
      'log_dir'   : str(log_dir),
      'log_file'  : 'log.txt',
      'log_level' : log_level
    }
    return self.cfg_logger


  def make_cfg_dumper(self):
    dump_dir = Path(self.experiment_folder) / Path("data")
    utils.make_dir(dump_dir)
    self.cfg_dumper = {
      'name'      : "cfg_dumper",
      'dump_dir': str(dump_dir),
      'dump_file': 'dump.txt',
    }
    return self.cfg_dumper

  def dump_cfg(self):
    conf_dir = Path(self.experiment_folder) / Path("config") 
    utils.make_dir(conf_dir)
    path = conf_dir / 'config.json'

    with open(str(path), 'w') as fp:
      json.dump(self.make_cfg_logger(), fp, indent=4)
      json.dump(self.make_cfg_dumper(), fp, indent=4)

