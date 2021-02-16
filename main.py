import cfg as configuration
import utils 
from env import GymEnv
from tqdm import tqdm
from experiment import Exp
from agentDQN import Agent


  
def main():
  cfg = configuration.CfgMaker()
  logger = utils.setup_logger(cfg.cfg_logger())
  env = GymEnv(cfg.cfg_env())
  agent = Agent(cfg.cfg_agent(), env.action_space)
  experiment = Exp(cfg.cfg_exp(),env,agent)
  experiment.start()
 


if __name__ == "__main__":
    main()