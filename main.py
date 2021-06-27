from dumper import Dumper
from utils.cfg import CfgMaker
from env import Env 
from utils.log import Logger
from DQN.DQNAgent import DqnAgent
from experiment import Experiment
from utils.utils import *


if __name__ == "__main__":
    cfg = CfgMaker()
    # Get all config
    cfg_logger = cfg.make_cfg_logger()
    cfg_dumper = cfg.make_cfg_dumper()
    cfg_experiment = cfg.make_cfg_experiment()
    cfg_env = cfg.make_cfg_env()
    cfg_agent = cfg.make_cfg_agent()
    cfg.show_configs()
    
    
    #Instantiate objects
    logger = Logger(cfg_logger)
    dumper = Dumper(cfg_dumper, logger)
    env = Env(cfg_env,logger)
    info_env = env.get_agent_info()
    agent = DqnAgent(cfg_agent,info_env,logger)
    experiment = Experiment(cfg_experiment,env,agent,dumper,logger)
    
        
    try:
        experiment.train()
        experiment.test()

    except KeyboardInterrupt:
        logger.handle_kb_int()
        env.handle_kb_int()
        dumper.close()
