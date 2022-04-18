from drlux_toolbox.metaclasses.env import Environment,StateSpace,ActionSpace
from loguru import logger
import gym

class ToyEnv(Environment):
    def __init__(self,envConfig):
        super().__init__()

        self.env = gym.make(envConfig['env_id']) 
        self.actionSpace = ToyEnvActionSpace(self.env.action_space)
        self.stateSpace  = ToyEnvStateSpace(self.env.observation_space)
        
        self.set_seed(envConfig['seed'])



    def close(self):
        logger.info("Closing Env")
        self.env.close()

    def reset(self):
        logger.info(f"Resetting env")
        self.env.reset()

    def step(self,action):
        return self.env.step(action)

    def set_seed(self,seed):
        logger.info(f"Setting {seed=} in Env")
        self.env.seed(seed)

    def getActionSpace(self):
        return self.actionSpace

    def getStateSpace(self):
        return self.stateSpace
        
    def handle_kb_int(self):
        logger.info("Handling kb interrupt in Env")
        self.close()

    def render(self):
        self.env.render()

class ToyEnvStateSpace():
    def __init__(self,observation_space):
        self.obs_space      =   observation_space
        self.range_high     =   observation_space.high
        self.range_low      =   observation_space.low
        self.dtype          =   observation_space.dtype
        self.shape          =   observation_space.shape
        self.show_state_env_info()

    def get_range_high(self):
        return self.range_high

    def get_range_low(self):
        return self.range_low

    def get_state_dtype(self):
        return self.dtype

    def get_obs_shape(self):
        return self.shape

    def show_state_env_info(self):
        logger.info("\t ################# ")
        logger.info("\t# Dump State Info Environment: ")
        logger.info(f"\t# State_space: {self.obs_space}")
        logger.info(f"\t# State Shape: {self.shape}")
        logger.info(f"\t# State dtype: {self.dtype}")
        logger.info(f"\t# State Range Low: {self.range_low}, High: {self.range_high}")
        logger.info("\t ################# \n")
    


class ToyEnvActionSpace():
    def __init__(self,action_space):
        #self.range_low     =   self.env.action_space.low
        #self.range_high    =   self.env.action_space.high
        self.action_space   =   action_space
        self.n_acts         =   action_space.n
        self.dtype          =   action_space.dtype
        self.range_high     =   action_space.n
        self.range_low      =   0            
        self.show_action_env_info()
        
    def sample_random_action(self):
        self.action_space.sample()

    def get_action_space(self):
        return self.action_space

    def get_n_acts(self):
        return self.n_acts

    def get_dtype(self):
        return self.dtype

    def get_range_high(self):
        return self.range_high

    def get_range_low(self):
        return self.range_low    

    def show_action_env_info(self):
        logger.info("\t ################# ")
        logger.info("\t# Dump Action Info Environment: ")
        logger.info(f"\t# Random Action: {self.sample_random_action()}")
        logger.info(f"\t# Action_space: {self.action_space}")
        logger.info(f"\t# Num Actions: {self.n_acts}")
        logger.info(f"\t# Action dtype: {self.dtype}")
        logger.info(f"\t# Action Range Low: {self.range_low}, High: {self.range_high}")
        logger.info("\t ################# \n")
    