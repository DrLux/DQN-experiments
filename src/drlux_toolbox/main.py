from urllib3 import Retry
from toolbox.config_manager import ConfigManager
from drlux_toolbox.toolbox.agents.random_agent import RandomAgent
from drlux_toolbox.toolbox.envs.classic_control_env import ToyEnv
from loguru import logger
import random 
import numpy as np
from dumper import Dumper
#from experiment import Experiment

config_path = "./input/original_config/classic_control_env"


def init():
    conf_manager = ConfigManager(config_path)
    seed = conf_manager.experimentConfig['seed']
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    return conf_manager


conf_manager = init()
conf_manager.print()

env = ToyEnv(conf_manager.envConfig)

conf_manager.addConfig("agentConfig", {"stateSpace": env.stateSpace, "actionSpace": env.actionSpace} )
agent = RandomAgent(conf_manager.agentConfig)

dumper = Dumper()

if conf_manager.experimentConfig['temporary_run']:
    import shutil
    shutil.rmtree(str(conf_manager.experiment_folder))

#experiment = Experiment(env,agent,dumper,num_eps)


# if __name__ == '__main__':
#    try:
#        experiment.exe_experiment()
#    except KeyboardInterrupt:
#        logger.error('main_rlhev: Experiment Keyboard Interrupt')
#        #dumper.handle_kb_int()
#        env.handle_kb_int()
