import utils
import datetime
import logging


class CfgMaker():

  def __init__(self):
    # Create initial folders
    utils.make_dir("experiments")
    utils.make_dir("experiments/results")
    
    self.experiment_folder = 'experiments/results/'+str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    utils.make_dir(self.experiment_folder)

  def cfg_logger(self):
    cfg_logger = {
      'log_dir': self.experiment_folder,
      'log_file': 'logger.txt',
      'log_level': logging.DEBUG
    }
    return cfg_logger


  def cfg_dumper(self):
    cfg_dumper = {
      'dump_dir': self.experiment_folder,
      'dump_file': 'dump.txt',
    }
    return cfg_dumper

  def cfg_env(self):
    cfg_env = {
      'symbolic'            : True, #if true use symbolic data. If not, use frame
      'env_name'            : 'Pendulum-v0',
      'seed'                : 0,
      'max_episode_length ' : 1000,
      'action_repeat'       : 1,
      'bit_depth'           : 5
    }
    return cfg_env