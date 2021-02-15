import cfg as configuration
import utils 
from env import GymEnv



def init():
  cfg = configuration.CfgMaker()
  logger = utils.setup_logger(cfg.cfg_logger())
  env = GymEnv(cfg.cfg_env())
  
def main():
  init()


if __name__ == "__main__":
    main()