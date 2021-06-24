from utils.cfg import CfgMaker
from env import Env 
from utils.log import Logger
from DQN.DQNAgent import DqnAgent
from experiment import Experiment
from utils.utils import *


if __name__ == "__main__":
    try:
        cfg = CfgMaker()
        logger = Logger(cfg.make_cfg_logger())
        env = Env(cfg.make_cfg_env(),logger)
        
        
        # Extract env info to update the agent config 
        # Action info
        action_range =  env.get_action_range()
        action_dtype =  env.get_action_dtype()
        num_actions  =  env.get_num_acts()
        

        # State info
        obs_shape    =  env.get_obs_shape()
        obs_range    =  env.get_state_range()
        obs_dtype    =  env.get_state_dtype()
        
        # get specific agent config
        net_config   =  cfg.make_dqn_net_config()
        experiment_cfg = cfg.make_cfg_experiment()

        # get exploration config
        exploration_cfg = cfg.make_exploration_config()
        cfg.add_key(exploration_cfg,'num_episode',experiment_cfg['num_episodes'])
        
        
        update_cfg_agent = {
            'action_range'  : action_range,
            'action_dtype'  : action_dtype,
            'num_actions'   : num_actions,

            'obs_shape'     : obs_shape,
            'obs_range'     : obs_range,
            'obs_dtype'     : obs_dtype,
        
            'net_config'   : net_config,
        }

        agent_cfg = cfg.make_cfg_agent()
        agent_cfg = cfg.update_cfg(agent_cfg,update_cfg_agent)
        cfg.add_key(agent_cfg['train_cfg'],'exploration_cfg',exploration_cfg)

        agent = DqnAgent(agent_cfg,logger)

        dumper = None 
        experiment = Experiment(experiment_cfg,env,agent,dumper,logger)

        cfg.show_configs()
        
        experiment.train()

    except KeyboardInterrupt:
        logger.handle_kb_int()
        env.handle_kb_int()
