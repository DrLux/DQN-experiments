from agent import *
from DQN.network import Network
import torch
from DQN.replay_buffer import vect_ReplayBuffer
from utils.utils import print_dict

class DQN_Train_Agent(TrainAgent):
    def __init__(self,agent_cfg,logger):
        self.state_name = "training"
        self.agent_cfg = agent_cfg
        self.train_cfg = self.agent_cfg['train_cfg']
        
        self.net_cfg = agent_cfg['net_config']
        self.dqn = Network(self.net_cfg)

        self.replay_cfg = self.__get_replay_cfg()
        self.memory = vect_ReplayBuffer(self.replay_cfg)

        self.batch_size = self.train_cfg['batch_size']
        self.min_replay_dim = self.train_cfg['min_replay_dim']
        
    
    

    def __get_replay_cfg(self):        
        for key in self.train_cfg['replay_cfg']['replay_keys']:
            self.train_cfg['replay_cfg'].update({key : self.agent_cfg[key]})
        return self.train_cfg['replay_cfg']


    def sample_random_action(self):
        return super(DQN_Train_Agent, self).sample_random_action()
    
    def chooseAction(self,obs):
        qValues = self.dqn(obs) # pass it through the network to get your estimations
        action = torch.argmax(qValues) # pick the highest
        action =  action.item() # return an int instead of a tensor containing the index of the best action
        
        import random
        # 10% of the time the agent picks an action at random, and ignores its own q values
        chanceOfAsparagus = random.randint(1, 10)
        if chanceOfAsparagus == 1:  #   10% chance
            action = random.randint(0, 1)

        return action  


    def learn(self,obs, action, reward, next_obs, done): 
        
        self.dqn.optimizer.zero_grad()  # resets all the tensor derivatives

        reward = torch.tensor(reward).float().detach().to(self.dqn.device)  # put the reward in tensor form and detach it so we dont backprop through it
        
        qValues = self.dqn(obs) # predict what reward each action will get
        nextQValues = self.dqn(next_obs)

        qValues = self.dqn(obs)
        nextQValues = self.dqn(next_obs)

        predictedValueOfNow = qValues[0][action]    #   interpret the past
        futureActionValue = nextQValues[0].max()    #   interpret the future

        trueValueOfNow = reward + futureActionValue * (1 - done)  # td function

        loss = self.dqn.loss(trueValueOfNow, predictedValueOfNow)

        loss.backward()
        self.dqn.optimizer.step()



class DQN_Eval_Agent(EvalAgent):
    def __init__(self,eval_cfg,logger):
        self.net_cfg = eval_cfg['net_config']
        self.dqn = Network(self.net_cfg)

        self.state_name = "evaluation"
        pass

    def sample_random_action(self):
        return super(DQN_Eval_Agent, self).sample_random_action()

    def chooseAction(self,obs):
        qValues = self.dqn(obs) # pass it through the network to get your estimations
        action = torch.argmax(qValues) # pick the highest
        return action.item()  # return an int instead of a tensor containing the index of the best action


class DqnAgent(Agent):
    
    def __init__(self,cfg,logger):
        self.logger = logger
        self.cfg = cfg
        self.__get_net_cfg()
        
        ### Init Agents
        self.train_agent = DQN_Train_Agent(self.cfg,logger)
        
        eval_cfg = self.cfg_net
        self.eval_agent = DQN_Eval_Agent(self.cfg,logger)
        
        self.agent_state = self.train_agent 
        self.logger.info_log(f" Init Agent in state {self.agent_state.state_name}")

    def __get_net_cfg(self):        
        self.cfg_net      = self.cfg['net_config']
        for key in self.cfg_net['keys']:
            self.cfg_net.update({key : self.cfg[key]})

    def set_agent_state(self,flag):
        if flag == "Train":
            self.agent_state = self.train_agent 

    def sample_random_action(self):
        return super(DqnAgent, self).sample_random_action()
    
    def chooseAction(self,obs):
        return self.agent_state.chooseAction(obs)

    def learn(self,obs, action, reward, next_obs, done):
        if self.agent_state.state_name == "training":
            self.train_agent.learn(obs, action, reward, next_obs, done) 
        else:
            pass