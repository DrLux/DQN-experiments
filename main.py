from cfg import CfgMaker
from env import Env 
from log import Logger
from DQNAgent import DqnAgent
from experiment import Experiment
import utils 


if __name__ == "__main__":
    try:
        cfg = CfgMaker()
        logger = Logger(cfg.make_cfg_logger())
        env = Env(cfg.make_cfg_env(),logger)
        
        # Extract env info to update the agent config 
        update_cfg_agent = {
            'action_range' : env.get_action_range(),
            'action_dtype' : env.get_action_dtype(),
            'num_actions'  : env.get_num_acts(),
        }

        agent_cfg = cfg.make_cfg_agent()
        agent_cfg = cfg.update_cfg(agent_cfg,update_cfg_agent)

        
        agent = DqnAgent(agent_cfg,logger)

        dumper = None 
        experiment = Experiment(cfg.make_cfg_experiment(),env,agent,dumper,logger)

        cfg.show_configs()
        
        experiment.start()

    except KeyboardInterrupt:
        logger.handle_kb_int()
        env.handle_kb_int()
