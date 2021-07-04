from utils.utils import make_dir
import datetime
import logging
import pickle
from pathlib import Path
import json

class CfgMaker():

    def __init__(self,load=False):
        self.all_configs = dict()
        self.load_cfg()
        
        if self.all_configs['cfg_experiment']['development'] == "DEVELOPMENT":
            self.experiment_folder = 'experiments/results/dev'
        else: 
            self.experiment_folder = 'experiments/results/'+str(datetime.datetime.now().strftime('%d-%b-%y_%H:%M:%S'))

        #  create folders
        make_dir(self.experiment_folder)
        
        self.conf_dir = Path(self.experiment_folder) / Path("config") 
        make_dir(self.conf_dir)
        self.conf_dir = self.conf_dir / 'config.json'


    def extend_dict(self,dict):
        all_extensions = set()
        for ex in dict['extensions']:
            all_extensions.add(ex)
            if type(self.all_configs[ex]) is dict:
                all_extensions.update(self.all_configs[ex]['extensions'])
         
        dict['extensions'] = list(all_extensions)
        for ext_cfg in all_extensions:
            dict[ext_cfg] = self.all_configs[ext_cfg]

    def make_cfg_logger(self):

        cfg_logger = self.all_configs['cfg_logger'] 
        
        # Add other sub-dictionary
        self.extend_dict(cfg_logger)
        
        # Update configs
        cfg_logger['dev_level'] = self.all_configs['cfg_experiment']['development']

        # add experiment folder
        if "experiment_folder" in cfg_logger:
            cfg_logger["experiment_folder"] = self.experiment_folder

        return cfg_logger
    
    def make_cfg_agent(self):

        cfg_agent = self.all_configs["cfg_agent"]

        # Add other sub-dictionary
        self.extend_dict(cfg_agent)

        # Update configs
        cfg_agent['seed'] = self.all_configs['cfg_experiment']['seed']
        cfg_agent["total_train_episodes"] = self.all_configs['cfg_experiment']["total_train_episodes"]

        # add experiment folder
        if "experiment_folder" in cfg_agent:
            cfg_agent["experiment_folder"] = self.experiment_folder
        
        return cfg_agent

    

    def make_cfg_experiment(self):
        cfg_experiment = self.all_configs['cfg_experiment']

        # Add other sub-dictionary
        self.extend_dict(cfg_experiment)

        # add experiment folder
        if "experiment_folder" in cfg_experiment:
            cfg_experiment["experiment_folder"] = self.experiment_folder
        
        return cfg_experiment

    def make_cfg_env(self):
        cfg_env = self.all_configs['cfg_env']

        # Add other sub-dictionary
        self.extend_dict(cfg_env)

        # Update configs
        cfg_env['seed'] = self.all_configs['cfg_experiment']['seed']

        # add experiment folder
        if "experiment_folder" in cfg_env:
            cfg_env["experiment_folder"] = self.experiment_folder

        return cfg_env


    def make_cfg_dumper(self):        
        cfg_dumper = self.all_configs['cfg_dumper'] 
        
        # Add other sub-dictionary
        self.extend_dict(cfg_dumper)
        
        # add experiment folder
        if "experiment_folder" in cfg_dumper:
            cfg_dumper["experiment_folder"] = self.experiment_folder

        return cfg_dumper

    def dump_cfg(self):
        with open(str(self.conf_dir), 'w') as outfile:
            json.dump(self.all_configs, outfile, indent=4)
    
    def load_cfg(self):
        path = "experiments/original_config/original_config.json"
        print(f"Loading configs from: {path}")
        with open(str(path), 'rb') as handle:
            self.all_configs = json.load(handle)
        self.show_configs()

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