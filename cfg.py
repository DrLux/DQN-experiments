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

        #  create folders
        utils.make_dir(self.experiment_folder)
        
        self.conf_dir = Path(self.experiment_folder) / Path("config") 
        utils.make_dir(self.conf_dir)
        self.conf_dir = self.conf_dir / 'config.json'
        self.conf_dir.unlink()


    def make_cfg_logger(self):
        log_dir = Path(self.experiment_folder) / Path("logdir")
        utils.make_dir(log_dir)
        self.cfg_logger = {
            'name'      : "cfg_logger",
            'log_dir'   : str(log_dir),
            'info_log_file'  : 'infolog.txt',
            'dbg_log_file'  : 'dbglog.txt',
            'DEV_LEVEL'     : self.DEV_LEVEL,
            }

        with open(str(self.conf_dir), 'a') as fp:
            json.dump(self.cfg_logger, fp, indent=4)
        return self.cfg_logger
    
    def make_cfg_agent(self):
        self.cfg_agent = {
            'name'      : "cfg_agent",
        }
        with open(str(self.conf_dir), 'a') as fp:
            json.dump(self.cfg_agent, fp, indent=4)
        return self.cfg_agent


    def make_cfg_experiment(self):
        self.cfg_experiment = {
            'name'      : "cfg_experiment",
        }
        with open(str(self.conf_dir), 'a') as fp:
            json.dump(self.cfg_experiment, fp, indent=4)
        return self.cfg_experiment

    def make_cfg_env(self):
        self.cfg_env = {
            'name'      :   'cfg_env',
            'env_name'  :   'CartPole-v0',#'MountainCarContinuous-v0',
            'render'    :   True,
        }
        with open(str(self.conf_dir), 'a') as fp:
            json.dump(self.cfg_env, fp, indent=4)
        return self.cfg_env


    def make_cfg_dumper(self):
        dump_dir = Path(self.experiment_folder) / Path("data")
        utils.make_dir(dump_dir)
        self.cfg_dumper = {
            'name'      : "cfg_dumper",
            'dump_dir': str(dump_dir),
            'dump_file': 'dump.txt',
        }
        with open(str(self.conf_dir), 'a') as fp:
            json.dump(self.cfg_dumper, fp, indent=4)
        return self.cfg_dumper

    def dump_cfg(self, cfg):
        with open(str(self.conf_dir), 'a') as fp:
            json.dump(cfg, fp, indent=4)


        

