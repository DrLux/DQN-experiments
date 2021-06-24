import numpy as np 


class Exploration_strategy():
    def __init__(self,exploration_cfg,logger):
        self.epsilon            = exploration_cfg['epsilon']            # chance of random action
        self.epsilon_decay      = exploration_cfg['epsilon_decay']      # how much the chance shrinks each step
        self.epsilon_min        = exploration_cfg['epsilon_min']        # minimum for the chance, so you never fully stop exploring
        self.num_episode        = exploration_cfg['num_episode']
        self.strategy           = exploration_cfg['strategy']
        self.logger             = logger


    def get_eploration_flag(self):
        if np.random.random() < self.epsilon:
            return True
        else:
            return False

        
  
    def decay_exp(self):
        if self.strategy == "linear":
            self.epsilon -= self.epsilon_decay  # shrink
        if self.epsilon < self.epsilon_min: # clamp
            self.epsilon = self.epsilon_min