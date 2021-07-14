import datetime
import logging
import pickle
from pathlib import Path
import yaml
from utils.utils import delete_folder



class CfgMaker():

    def __init__(self,load=False):
        self.all_configs = dict()
        self.load_cfg_path = Path("drl_framework/original_config/")
        self.load_cfg()

        if self.all_configs['cfg_experiment']['development'] == "DEVELOPMENT":
            self.experiment_folder = Path('experiments/results/dev')
            delete_folder(self.experiment_folder) 
        else: 
            self.experiment_folder = Path('experiments/results/'+str(datetime.datetime.now().strftime('%d-%b-%y_%H:%M:%S')))

        #  create folders
        self.experiment_folder.mkdir(parents=True,exist_ok=True)
        
        self.conf_dir = Path(self.experiment_folder) / Path("config") 
        self.conf_dir.mkdir(parents=True,exist_ok=True)
         


    def extend_dict(self,dict_to_ext):
        all_extensions = set()
        for ex in dict_to_ext['extensions']:
            all_extensions.add(ex)
            if type(self.all_configs[ex]) is dict:
                all_extensions.update(self.all_configs[ex]['extensions'])
        dict_to_ext['extensions'] = list(all_extensions)
        for ext_cfg in all_extensions:
            dict_to_ext[ext_cfg] = self.all_configs[ext_cfg]

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
        cfg_agent['extensions'].append(cfg_agent['agent_class'])

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

        cfg_dumper["env_names"] = self.all_configs['cfg_env']['dump_names']
        cfg_dumper["agent_names"] = self.all_configs[(self.all_configs['cfg_agent']['agent_class'])]['dump_names']
        
        # add experiment folder
        if "experiment_folder" in cfg_dumper:
            cfg_dumper["experiment_folder"] = self.experiment_folder

        return cfg_dumper

    def dump_cfg(self):
        #with open(str(self.conf_dir), 'w') as outfile:
        #    json.dump(self.all_configs, outfile, indent=4)
        if self.all_configs:
            for k,v in self.all_configs.items():
                conf_path = self.conf_dir / (k + ".yml")

                with conf_path.open('a') as fp:
                    yaml.dump(v, fp)
    
    def load_cfg(self):
        print(f"Loading configs from: {self.load_cfg_path}")
        for conf_path in self.load_cfg_path.glob(r"*.yml"):
            with conf_path.open('r') as handle:
                self.all_configs[conf_path.stem] = yaml.load(handle, Loader=yaml.Loader)
        #with open(str(self.load_cfg_path), 'rb') as handle:
        #    self.all_configs = json.load(handle)
        #self.show_configs()

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