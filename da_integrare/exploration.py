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
        random_value = np.random.random() 
        self.logger.dbg_log(f"Exploration sample {random_value} and epsilon is {self.epsilon}.")
        return  random_value < self.epsilon

    def decay_exp(self):
        self.logger.dbg_log(f"Exploration trying to decay exploration rate.")
        if self.epsilon > self.epsilon_min: # clamp
            self.logger.dbg_log(f"Exploration decaying exploration rate.")
            if self.strategy == "linear":
                self.epsilon -= self.epsilon_decay  # shrink
        