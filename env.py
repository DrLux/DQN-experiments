import gym

#class State():
    
class Action():
    def __init__(self,env, logger):
        self.env            =   env
        self.logger         =   logger
        self.action_space   =   self.env.action_space.__class__.__name__
        self.logger.info_log(" Init action. Dump Action_space ", self.action_space)

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
            self.n_acts         =   len(self.env.action_space.shape)
            if self.n_acts == 0:
                self.n_acts = 1
        if self.action_space == "Continuous":
            raise NotImplementedError
        elif self.action_space == "Box":
            raise NotImplementedError

    
    def sample_random_action(self):
        return self.env.action_space.sample()

    def show_action_env_info(self):
        self.logger.info_log("\n\t ################# ")
        self.logger.info_log("\t# Dump Info Environment: ")
        self.logger.info_log(f"\t# Action_space: {self.action_space}")
        self.logger.info_log(f"\t# Num Actions: {self.n_acts}")
        self.logger.info_log(f"\t# Action dtype: {self.dtype}")
        self.logger.info_log(f"\t# Action Range Low: {self.range_low}, High: {self.range_high}")
        self.logger.info_log("\t ################# \n")



class Env():

    def __init__(self,cfg_env, logger):
        self.env = gym.make(cfg_env['env_name'])
        self.logger = logger
        self.render = cfg_env['render']
        self.action = Action(self.env, self.logger)
        self.show_info_env()

        

    def reset(self):
        self.logger.info_log("Resetting Environment")
        init_obs = self.env.reset()
        return init_obs 
         
    def sample_action(self):
        return self.action.sample_random_action()

    def step(self, action):
        if self.render:
            self.env.render()
        new_state,rew,done,info = self.env.step(action)
        return new_state,rew,done

    def handle_kb_int(self):
        self.info_log("Interrupting env")
        self.close()

    def get_action_range(self):
        return self.action.get_action_range()

    def get_action_dtype(self):
        return self.action.get_action_dtype()

    def get_num_acts(self):
        return self.action.get_num_acts()

    def show_info_env(self):
        self.action.show_action_env_info()


    def close(self):
        self.env.close()
    

    