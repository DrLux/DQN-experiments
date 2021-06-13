import numpy as np

class Agent:
    def __init__(self,cfg,logger):
        self.logger = logger
        self.action_range = cfg['action_range']
        self.action_dtype = cfg['action_dtype']
        self.num_actions  = cfg['num_actions']
        self.logger.info_log(f" Init Agent")


    def sample_random_action(self):
        action = np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(self.num_actions)).astype(self.action_dtype.name)
        if self.num_actions == 1:
            action = action[0]
        return action