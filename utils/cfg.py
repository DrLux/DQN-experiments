from utils.utils import make_dir
import datetime
import logging
import pickle
from pathlib import Path
import json

class CfgMaker():

    def __init__(self,load=False):
        # Create initial folders
        self.DEV_LEVEL = "DEVELOPMENT"

        if self.DEV_LEVEL == "DEVELOPMENT":
            self.experiment_folder = 'experiments/results/dev'
        else: 
            self.experiment_folder = 'experiments/results/'+str(datetime.datetime.now().strftime('%d-%b-%y %H:%M:%S'))

        #  create folders
        make_dir(self.experiment_folder)
        
        self.conf_dir = Path(self.experiment_folder) / Path("config") 
        make_dir(self.conf_dir)
        self.conf_dir = self.conf_dir / 'config.json'

        self.all_configs = dict()
        self.load_cfg()


    def make_cfg_logger(self):
        log_dir = Path(self.experiment_folder) / Path("logdir")
        make_dir(log_dir)

        cfg_logger = self.all_configs['cfg_logger'] 
        cfg_logger['log_dir'] = str(log_dir)

        return cfg_logger
    
    def make_cfg_agent(self):

        ckp_dir = Path(self.experiment_folder) / Path("ckp")
        make_dir(ckp_dir)

        mem_dir = Path(self.experiment_folder) / Path("mem")
        make_dir(mem_dir)


        # Get all configs
        replay_cfg      = self.all_configs['replay_cfg']
        exploration_cfg = self.all_configs['exploration_cfg'] 
        train_cfg       = self.all_configs['train_cfg']
        cfg_agent       = self.all_configs['cfg_agent']
        
        if cfg_agent['policy_type'] == "DQN":   
            dqn_cfg  = self.all_configs['dqn_net_cfg']
        
        # Update configs
        dqn_cfg['ckp_path']     = str(ckp_dir)
        dqn_cfg['seed'] = self.all_configs['cfg_experiment']['seed']
        replay_cfg['mem_path']  = str(mem_dir)
        if "total_train_episodes" in self.all_configs['cfg_experiment']:
            exploration_cfg["total_train_episodes"] = self.all_configs['cfg_experiment']["total_train_episodes"]
        
        # Pack configs 
        train_cfg['replay_cfg'] = replay_cfg
        train_cfg['exploration_cfg'] = exploration_cfg
        cfg_agent['train_cfg'] = train_cfg       
        cfg_agent['dqn_cfg'] = dqn_cfg        
        
        return cfg_agent

    def make_cfg_experiment(self):
        cfg_experiment = self.all_configs['cfg_experiment']
        return cfg_experiment

    def make_cfg_env(self):
        cfg_env = self.all_configs['cfg_env']
        cfg_env['seed'] = self.all_configs['cfg_experiment']['seed']
        return cfg_env


    def make_cfg_dumper(self):
        dump_dir = Path(self.experiment_folder) / Path("dump")
        make_dir(dump_dir)
        
        cfg_dumper = self.all_configs['cfg_dumper'] 
        cfg_dumper['dump_dir'] = str(dump_dir)

        return cfg_dumper

    def dump_cfg(self, cfg):
        with open(str(self.conf_dir), 'w') as outfile:
            json.dump(cfg, outfile, indent=4)
    
    def load_cfg(self):
        print(f"Loading configs from: {self.conf_dir}")
        self.show_configs()
        with open(str(self.conf_dir), 'rb') as handle:
            self.all_configs = json.load(handle)

    def show_configs(self):        
        for conf_name, configs in self.all_configs.items():
            try:
                str_conf = json.dumps(configs, indent = 4)
                print(f"\n {conf_name} -> {str_conf}")
            except (TypeError, Error):
                print(f"\n {conf_name} -> {configs}")

    
    def update_cfg(self,old_dict,upd_dict):
        self.all_configs[old_dict['name']].update(upd_dict)
        #self.dump_cfg(self.all_configs)
        return old_dict

    def add_key(self,old_dict_keys, k,v):

        if len(old_dict_keys) == 1:
            self.all_configs[old_dict_keys[0]][k] = v
        if len(old_dict_keys) == 2:
            self.all_configs[old_dict_keys[0]][old_dict_keys[1]][k] = v
    
        #self.dump_cfg(self.all_configs)
        return old_dict_keys