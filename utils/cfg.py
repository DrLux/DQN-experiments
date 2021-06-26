from utils.utils import make_dir
import datetime
import logging
import pickle
from pathlib import Path
import json
import torch

class CfgMaker():

    def __init__(self):
        # Create initial folders
        self.DEV_LEVEL = "DEVELOPMENT"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        if self.DEV_LEVEL == "DEVELOPMENT":
            self.experiment_folder = 'experiments/results/dev'
        else: 
            self.experiment_folder = 'experiments/results/'+str(datetime.datetime.now().strftime('%d-%b-%y %H:%M:%S'))

        #  create folders
        make_dir(self.experiment_folder)
        
        self.conf_dir = Path(self.experiment_folder) / Path("config") 
        make_dir(self.conf_dir)
        self.conf_dir = self.conf_dir / 'config.pkl'
        if self.conf_dir.exists():
            self.conf_dir.unlink()
        self.all_configs = dict()

    def make_exploration_config(self):
        self.exploration_cfg = {
            'name'              : "exploration_cfg",    
            'epsilon'           : 0.1,            
            'epsilon_decay'     : 0.00005,  
            'epsilon_min'       : 0.001,
            'strategy'          : "linear"         
        }
        self.all_configs['exploration_cfg'] = self.exploration_cfg
        self.dump_cfg(self.all_configs)
        return self.exploration_cfg


    def make_dqn_net_config(self):
        self.dqn_net_cfg = {
            'name'      : "dqn_net_cfg",
            'fc1Dims'      : 1024,
            'fc2Dims'      : 512,
            'lr'           : 0.0001,
            'keys'         : ['action_range','action_dtype','num_actions','obs_shape','obs_range','obs_dtype'],
            'device'       : self.device
        }
        self.all_configs['dqn_net_cfg'] = self.dqn_net_cfg
        self.dump_cfg(self.all_configs)
        
        return self.dqn_net_cfg


    def make_cfg_logger(self):
        log_dir = Path(self.experiment_folder) / Path("logdir")
        make_dir(log_dir)
        self.cfg_logger = {
            'name'          : "cfg_logger",
            'log_dir'       : str(log_dir),
            'info_log_file' : 'infolog.txt',
            'dbg_log_file'  : 'dbglog.txt',
            'DEV_LEVEL'     : self.DEV_LEVEL,
            }

        self.all_configs['cfg_logger'] = self.cfg_logger
        self.dump_cfg(self.all_configs)

        return self.cfg_logger
    
    def make_cfg_agent(self):
        policy_type = "DQN"

        replay_cfg = {
            'name'              : 'replay_cfg',
            'replay_dim'        : 1000000,
            'replay_keys'       : ['obs_shape', 'obs_dtype', 'action_dtype']
        }

        traing_cfg = {
            'name'              : 'traing_cfg',
            'replay_cfg'        : replay_cfg,
            'batch_size'        : 64,
            'min_replay_dim'    : 2048,
        }

        self.cfg_agent = {
            'name'      : "cfg_agent",
            'train_cfg' : traing_cfg,
            'gamma'             : 0.99
        }

        self.all_configs['cfg_agent'] = self.cfg_agent
        self.all_configs['replay_cfg'] = replay_cfg
        self.all_configs['traing_cfg'] = traing_cfg
        self.dump_cfg(self.cfg_agent)

        return self.cfg_agent

    def update_cfg(self,old_dict,upd_dict):
        self.all_configs[old_dict['name']].update(upd_dict)
        self.dump_cfg(self.all_configs)
        return old_dict

    def add_key(self,old_dict, k,v):
        self.all_configs[old_dict['name']][k] = v
        self.dump_cfg(self.all_configs)
        return old_dict





    def make_cfg_experiment(self):
        self.cfg_experiment = {
            'name'              : "cfg_experiment",
            'max_allowed_steps' : 50,
            'num_episodes'      : 100,
        }

        self.all_configs['cfg_experiment'] = self.cfg_experiment
        self.dump_cfg(self.cfg_experiment)

        return self.cfg_experiment

    def make_cfg_env(self):
        self.cfg_env = {
            'name'      :   'cfg_env',
            'env_name'  :   'CartPole-v0',#'MountainCarContinuous-v0',
            'render'    :   False,
        }
        self.all_configs['cfg_env'] = self.cfg_env
        self.dump_cfg(self.cfg_env)

        return self.cfg_env


    def make_cfg_dumper(self):
        dump_dir = Path(self.experiment_folder) / Path("data")
        make_dir(dump_dir)
        self.cfg_dumper = {
            'name'      : "cfg_dumper",
            'dump_dir': str(dump_dir),
            'dump_file': 'dump.txt',
        }
        self.all_configs['cfg_dumper'] = self.cfg_dumper
        self.dump_cfg(self.cfg_env)
        return self.cfg_dumper

    def dump_cfg(self, cfg):
        with open(str(self.conf_dir), 'wb') as fp:
            pickle.dump(cfg, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_cfg(self):
        print(f"Loading configs from: {self.conf_dir}")
        self.show_configs()
        with open(str(self.conf_dir), 'rb') as handle:
            self.all_configs = pickle.load(handle)

    def show_configs(self):        
        for conf_name, configs in self.all_configs.items():
            try:
                str_conf = json.dumps(configs, indent = 4)
                print(f"\n {conf_name} -> {str_conf}")
            except (TypeError, OverflowError):
                print(f"\n {conf_name} -> {configs}")

    
