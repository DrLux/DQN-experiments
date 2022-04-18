from drlux_toolbox.metaclasses.config import Config
from loguru import logger
from pathlib import Path
import sys

class ConfigManager(Config):
    def __init__(self, cfg2load):
        super().__init__(cfg2load)
        logger.add(sys.stderr, format="{time} {level} {message}", filter="drlux_toolbox", level="INFO")
        logger.add( str(self.log_dirpath / Path("log.log")))

    @property
    def experimentConfig(self):
        self.dump_cfg()
        return self.allConfigs["experimentConfig"]

    @property
    def envConfig(self):
        self.allConfigs["envConfig"]["seed"] = self.allConfigs["experimentConfig"]["seed"]
        self.dump_cfg()
        return self.allConfigs["envConfig"]

    @property
    def agentConfig(self):
        self.dump_cfg()
        return self.allConfigs["agentConfig"]