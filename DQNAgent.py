from agent import *

class DQN_Train_Agent(TrainAgent):
    def __init__(self,train_cfg,logger):
        pass

    def sample_random_action(self):
        return super(DQN_Train_Agent, self).sample_random_action()
    
    def chooseAction(self,obs):
        raise NotImplementedError()

class DQN_Eval_Agent(EvalAgent):
    def __init__(self,eval_cfg,logger):
        pass

    def sample_random_action(self):
        return super(DQN_Eval_Agent, self).sample_random_action()

    def chooseAction(self,obs):
        raise NotImplementedError()


class DqnAgent(Agent):
    
    def __init__(self,cfg,logger):
        self.logger = logger
        self.action_range = cfg['action_range']
        self.action_dtype = cfg['action_dtype']
        self.num_actions  = cfg['num_actions']

        ### Init Agents
        train_cfg = None
        self.train_agent = DQN_Train_Agent(train_cfg,logger)
        
        eval_cfg = None
        self.eval_agent = DQN_Eval_Agent(train_cfg,logger)
        
        self.agent_state = self.train_agent 
        #self.logger.info_log(f" Init Agent")

    def set_agent_state(self,state):
        if state == "Train":
            self.agent_state = self.train_agent 

    def sample_random_action(self):
        return super(DqnAgent, self).sample_random_action()
        
    
    def chooseAction(self,obs):
        self.agent_state.chooseAction(obs)

    
    
