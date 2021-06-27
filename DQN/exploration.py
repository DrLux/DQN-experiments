import numpy as np 


class Exploration_strategy():
    def __init__(self,exploration_cfg,logger):
        self.epsilon            = exploration_cfg['epsilon']            # chance of random action
        self.epsilon_decay      = exploration_cfg['epsilon_decay']      # how much the chance shrinks each step
        self.epsilon_min        = exploration_cfg['epsilon_min']        # minimum for the chance, so you never fully stop exploring
        self.total_train_episodes        = exploration_cfg['total_train_episodes']
        self.strategy           = exploration_cfg['strategy']
        self.logger             = logger


    def exploration_step_flag(self):
        return np.random.random() < self.epsilon

        
  
    def decay_exp(self):
        if self.epsilon > self.epsilon_min: # clamp
            if self.strategy == "linear":
                self.epsilon -= self.epsilon_decay  # shrink
        
        #self.epsilon -= self.epsilon_decay
        #if self.epsilon < self.epsilon_min:
        #    self.epsilon = self.epsilon_min
        