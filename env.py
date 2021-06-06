import gym

#class State():
    
class Action():
    def __init__(self,env, logger):
        self.env            =   env
        self.logger         =   logger
        self.action_space   =   self.env.action_space.__class__.__name__
        self.logger.info_log("Action_space: ", self.action_space)

        #self.n_acts         =   self.env.action_space.n
        #self.env.action_space.shape

        def get_range(self):
            if self.action_space == "Discrete":
                assert 1 == 2
            elif self.action_space == "Continuous":
                self.range_high     =   self.env.action_space.high
                self.range_low      =   self.env.action_space.low
            elif self.action_space == "Box":
                self.range_low = self.env.action_space.low
                self.range_high = self.env.action_space.high
            
            self.logger.info_log("Range_low: ", self.range_low)
            self.logger.info_log("Range_high: ", self.range_high)

        def get_dtype(self):
            if self.action_space == "Discrete":
                assert 1 == 2
            elif self.action_space == "Continuous":
                self.dtype          =   self.env.action_space.dtype
            elif self.action_space == "Box":
                self.dtype = self.env.action_space.dtype
            self.logger.info_log("Dtype: ", self.dtype)

        
        def sample_random_action(self):
            return self.env.action_space.sample()





class Env():

    def __init__(self,cfg_env, logger):
        self.env = gym.make(cfg_env['env_name'])
        self.logger = logger
        self.render = cfg_env['render']
        self.action = Action(self.env, self.logger)
        

        self.get_action_space()

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

    def get_action_space(self):
        return self.action


    def close(self):
        self.env.close()
    