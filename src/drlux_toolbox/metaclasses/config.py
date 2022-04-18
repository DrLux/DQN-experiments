from pathlib import Path
import datetime
import yaml
from pprint import pprint


class Config:
    """
    cfg2load: string that contain the path to the configs to the folder that contain all the config in yaml format
    """
    def __init__(self, cfg2load):
        
        cfg2load = Path(cfg2load)
        experiment_name = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        self.experiment_folder = Path('./output/data/results/' + experiment_name)
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

        self.experiment_folder = Path('./output/data/results/' + experiment_name)
        self.experiment_folder.mkdir(parents=True, exist_ok=True)

        self.ckp_dirpath = self.experiment_folder / 'ckp_dir'
        self.ckp_dirpath.mkdir(parents=True, exist_ok=True)

        self.log_dirpath = self.experiment_folder / 'log_dir'
        self.log_dirpath.mkdir(parents=True, exist_ok=True)

        self.dashboard_dirpath = self.experiment_folder / 'dashboard'
        self.dashboard_dirpath.mkdir(parents=True, exist_ok=True)

        self.load_cfg(cfg2load)        
                


    def load_cfg(self,cfg2load):
        self.allConfigs = dict()
        for conf_path in cfg2load.glob(r"*.yml"):
            with conf_path.open('r') as handle:
                self.allConfigs[conf_path.stem] = yaml.load(handle, Loader=yaml.Loader) or {}
        

    def dump_cfg(self):
        if self.allConfigs:
            path = self.experiment_folder / "config"
            path.mkdir(parents=True, exist_ok=True)
            
            for k,v in self.allConfigs.items():
                conf_path = path / (k + ".yml")

                with conf_path.open('w') as fp:
                    yaml.dump(v, fp)


    def __str__(self):
        return str(self.allConfigs)

    def __reprs__(self):
        self.__str__()

    def addConfig(self,titleConfig,newDict):
        self.allConfigs[titleConfig] = {**self.allConfigs[titleConfig], **newDict}
        self.dump_cfg()

    @property
    def envConfig(self):
        self.allConfigs["envConfig"]["seed"] = self.seed
        self.dump_cfg()
        return self.allConfigs["envConfig"]

    def print(self):
        pprint(vars(self))

    #@property
    #def classConfig(self):
    #    self.dump_cfg()
    #    return self.allConfigs["classConfig"]
    