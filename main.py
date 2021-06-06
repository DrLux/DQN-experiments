from cfg import CfgMaker
from env import Env 
from log import Logger
from agent import Agent
from experiment import Experiment
import utils 


if __name__ == "__main__":
    try:
        cfg = CfgMaker()
        logger = Logger(cfg.make_cfg_logger())
        env = Env(cfg.make_cfg_env(),logger)
        agent = Agent(cfg.make_cfg_agent(),logger)
        experiment = Experiment(cfg.make_cfg_experiment(),env,logger)

        experiment.start()

    except KeyboardInterrupt:
        logger.handle_kb_int()
        env.handle_kb_int()
