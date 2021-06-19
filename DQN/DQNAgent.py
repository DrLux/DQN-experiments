from agent import *
from DQN.network import Network

class DQN_Train_Agent(TrainAgent):
    def __init__(self,train_cfg,logger):
        pass

    def sample_random_action(self):
        return super(DQN_Train_Agent, self).sample_random_action()
    
    def chooseAction(self,obs):
        raise NotImplementedError()

class DQN_Eval_Agent(EvalAgent):
    def __init__(self,eval_cfg,logger, dqn):
        self.dqn = dqn
        pass

    def sample_random_action(self):
        return super(DQN_Eval_Agent, self).sample_random_action()

    def chooseAction(self,obs):
        obs = torch.tensor(obs).float() #cast state into tensor
        obs = obs.to(self.dqn.device) # and put it on the gpu/cpu
        obs = obs.unsqueeze(0)  # add batch size

        qValues = self.dqn(obs) # pass it through the network to get your estimations
        action = torch.argmax(qValues) # pick the highest
        return action.item()  # return an int instead of a tensor containing the index of the best action


class DqnAgent(Agent):
    
    def __init__(self,cfg,logger):
        self.logger = logger
        self.action_range = cfg['action_range']
        self.action_dtype = cfg['action_dtype']
        self.num_actions  = cfg['num_actions']

        self.obs_shape    = cfg['obs_shape']
        self.obs_range    = cfg['obs_range']
        self.obs_dtype    = cfg['obs_dtype']

        
        update_cfg_net = {
                            'action_range'     : self.action_range,
                            'action_dtype'     : self.action_dtype,
                            'num_actions'      : self.num_actions,
                            'obs_shape'        : self.obs_shape,
                            'obs_range'        : self.obs_range,
                            'obs_dtype'        : self.obs_dtype                                                   
                        }

        self.cfg_net      = cfg['net_config']
        self.cfg_net.update(update_cfg_net)

        self.dqn = Network(self.cfg_net)
        assert 1 == 2

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

    
    
