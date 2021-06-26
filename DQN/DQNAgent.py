from agent import *
from DQN.network import Network
import torch
from DQN.replay_buffer import vect_ReplayBuffer
from utils.utils import print_dict
from DQN.exploration import Exploration_strategy

class DQN_Train_Agent(TrainAgent):
    def __init__(self,agent_cfg,logger):      

        self.state_name = "training"
        self.agent_cfg = agent_cfg
        self.action_range = agent_cfg['action_range']
        self.num_actions = agent_cfg['num_actions']
        self.train_cfg = self.agent_cfg['train_cfg']
        self.action_dtype = self.agent_cfg['action_dtype']
        self.gamma = self.agent_cfg['gamma']

        
        self.net_cfg = agent_cfg['net_config']
        self.dqn = Network(self.net_cfg)

        self.replay_cfg = self.__get_replay_cfg()
        self.memory = vect_ReplayBuffer(self.replay_cfg)

        self.batch_size = self.train_cfg['batch_size']
        self.min_replay_dim = self.train_cfg['min_replay_dim']

        exploration_cfg = self.train_cfg['exploration_cfg']
        self.exploration = Exploration_strategy(exploration_cfg,logger)


    def __get_replay_cfg(self):        
        for key in self.train_cfg['replay_cfg']['replay_keys']:
            self.train_cfg['replay_cfg'].update({key : self.agent_cfg[key]})
        return self.train_cfg['replay_cfg']


    def sample_random_action(self):
        action = super(DQN_Train_Agent, self).sample_random_action()
        action = action[0]
        return action 
    
    def chooseAction(self,obs):
        if self.exploration.get_eploration_flag():
            action = self.sample_random_action()
        else:
            obs = torch.tensor(obs).float().detach()
            qValues = self.dqn(obs) # pass it through the network to get your estimations
            action = torch.argmax(qValues) # pick the highest
            action =  action.item() # return an int instead of a tensor containing the index of the best action

        return action  

    def get_epsilon(self):
        return self.exploration.epsilon


    def learn(self,state, action, reward, next_state, done): 
        self.storeMemory(state, action, reward, next_state, done)
        if self.memory.memCount < self.min_replay_dim:   
            return
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        states  = torch.tensor(states , dtype=torch.float32).to(self.dqn.device)
        actions = torch.tensor(actions, dtype=torch.long   ).to(self.dqn.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.dqn.device)
        next_state = torch.tensor(new_states, dtype=torch.float32).to(self.dqn.device)
        dones   = torch.tensor(dones  , dtype=torch.bool   ).to(self.dqn.device)

        self.dqn.optimizer.zero_grad()  # resets all the tensor derivatives

        batchIndices = np.arange(self.batch_size, dtype=np.int64)  # i learned how to spell indices
        qValue = self.dqn(states)[batchIndices, actions]

        next_qValue = self.dqn(next_state)        #   values of all actions
        next_qValue = torch.max(next_qValue, dim=1)[0] #   extract greedy action value
        next_qValue[dones] = 0.0                    #   filter out post-terminal states

        qTarget = reward + self.gamma * next_qValue
        loss = self.dqn.loss(qTarget, qValue)

        self.dqn.optimizer.zero_grad()
        loss.backward()
        self.dqn.optimizer.step()

        self.exploration.decay_exp()

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

class DQN_Eval_Agent(EvalAgent):
    def __init__(self,eval_cfg,logger):
        self.net_cfg = eval_cfg['net_config']
        self.dqn = Network(self.net_cfg)

        self.state_name = "evaluation"
        pass

    def sample_random_action(self):
        return super(DQN_Eval_Agent, self).sample_random_action()

    def chooseAction(self,obs):
        obs = obs.unsqueeze(0)  # add batch size
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

    def learn(self,state, action, reward, nextState, done):
        if self.agent_state.state_name == "training":
            self.train_agent.learn(state, action, reward, nextState, done) 
        else:
            pass

    def get_epsilon(self):
        if self.agent_state.state_name == "training":
            return self.train_agent.get_epsilon()        
        else:
            pass
