import gym

# QUI MANCA TUTTA LA PARTE SUI SEED 

class State():
    def __init__(self,env, logger):
        self.env            =   env
        self.logger         =   logger
        self.obs_space     =   self.env.observation_space
        self.logger.info_log(f" Init State. Dump State_space {self.obs_space}")

    def get_state_range(self):
        return (self.obs_space.low,self.obs_space.high)

    def get_state_dtype(self):
        return self.obs_space.dtype

    def get_obs_shape(self):
        return self.obs_space.shape

    def show_state_env_info(self):
        self.logger.info_log("\n\t ################# ")
        self.logger.info_log("\t# Dump State Info Environment: ")
        self.logger.info_log(f"\t# State_space: {self.obs_space}")
        self.logger.info_log(f"\t# State Shape: {self.obs_space.shape}")
        self.logger.info_log(f"\t# State dtype: {self.obs_space.dtype}")
        self.logger.info_log(f"\t# State Range Low: {self.obs_space.low}, High: {self.obs_space.high}")
        self.logger.info_log("\t ################# \n")




class Action():
    def __init__(self,env, logger):
        self.env            =   env
        self.logger         =   logger
        self.action_space   =   self.env.action_space.__class__.__name__
        self.logger.info_log(f" Init action. Dump Action_space {self.env.action_space}", )

        self.init_action_range()
        self.init_dtype()
        self.init_num_acts()


    def get_action_range(self):
        return (self.range_low,self.range_high)

    def get_action_dtype(self):
        return self.dtype

    def get_num_acts(self):
        return self.n_acts

    def init_action_range(self):
        if self.action_space == "Discrete":
            self.range_high     =   self.env.action_space.n
            self.range_low      =   0            
        if self.action_space == "Continuous":
            self.range_high     =   self.env.action_space.high
            self.range_low      =   self.env.action_space.low
        elif self.action_space == "Box":
            raise NotImplementedError
        
            
    def init_dtype(self):
        if self.action_space == "Discrete":
            self.dtype          =   self.env.action_space.dtype
        elif self.action_space == "Continuous":
            raise NotImplementedError
        elif self.action_space == "Box":
            raise NotImplementedError

    
    def init_num_acts(self):
        if self.action_space == "Discrete":
            self.n_acts = self.env.action_space.n
        if self.action_space == "Continuous":
            raise NotImplementedError
        elif self.action_space == "Box":
            raise NotImplementedError

    
    def sample_random_action(self):
        return self.env.action_space.sample()

    def show_action_env_info(self):
        self.logger.info_log("\n\t ################# ")
        self.logger.info_log("\t# Dump Action Info Environment: ")
        self.logger.info_log(f"\t# Random Action: {self.sample_random_action()}")
        self.logger.info_log(f"\t# Action_space: {self.action_space}")
        self.logger.info_log(f"\t# Num Actions: {self.n_acts}")
        self.logger.info_log(f"\t# Action dtype: {self.dtype}")
        self.logger.info_log(f"\t# Action Range Low: {self.range_low}, High: {self.range_high}")
        self.logger.info_log("\t ################# \n")





class Env():

    def __init__(self,cfg_env, logger):
        self.env = gym.make(cfg_env['env_name']).unwrapped
        self.logger = logger
        self.render_flag = cfg_env['render']
        self.action = Action(self.env, self.logger)
        self.state = State(self.env, self.logger)
        self.show_info_env()

    def render(self):
        if self.render_flag:
            self.env.render()

    def reset(self):
        self.logger.info_log("Resetting Environment")
        init_obs = self.env.reset()
        return init_obs 
         
    def sample_action(self):
        return self.action.sample_random_action()

    def step(self, action):
        if self.render_flag:
            self.env.render()
        new_state,rew,done,info = self.env.step(action)
        return new_state,rew,done

    def handle_kb_int(self):
        self.logger.info_log("Interrupting env")
        self.close()

    def get_action_range(self):
        return self.action.get_action_range()

    def get_action_dtype(self):
        return self.action.get_action_dtype()

    def get_num_acts(self):
        return self.action.get_num_acts()

    def show_info_env(self):
        self.logger.info_log("\t# Dump Info Environment: ")
        self.action.show_action_env_info()
        self.state.show_state_env_info()

    def get_obs_shape(self):
        return self.state.get_obs_shape()

    def get_state_range(self):
        return self.state.get_state_range()

    def get_state_dtype(self):
        return self.state.get_state_dtype()

    def close(self):
        self.env.close()