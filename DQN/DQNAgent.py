from drl_framework.agent import *
from DQN.network import Network
from DQN.vect_memory import vect_ReplayBuffer
from utils.utils import print_dict
from DQN.exploration import Exploration_strategy
from collections import defaultdict
from pathlib import Path
import torch
from utils.utils import make_dir


class DQN_Train_Agent(TrainAgent):
    def __init__(self,agent_cfg,logger):      

        self.state_name = "training"
        self.agent_cfg = agent_cfg
        self.action_range = agent_cfg['action_range']
        self.num_actions = agent_cfg['num_actions']
        self.train_cfg = self.agent_cfg['train_cfg']
        self.action_dtype = self.agent_cfg['action_dtype']
        self.gamma = self.agent_cfg['gamma']
        self.step_info_dump = defaultdict(list)
        self.logger = logger

        self.net_cfg = agent_cfg['dqn_net_cfg']
        self.dqn = Network(self.net_cfg,self.state_name,logger) 


        self.replay_cfg = self.__get_replay_cfg()
        self.memory = vect_ReplayBuffer(self.replay_cfg,self.logger)

        self.batch_size = self.train_cfg['batch_size']
        self.min_replay_dim = self.train_cfg['min_replay_dim']

        exploration_cfg = self.__get_exploration_cfg()
        self.exploration = Exploration_strategy(exploration_cfg,logger)

        
    def get_info_dump(self):
        temp_dict = self.step_info_dump 
        self.logger.dbg_log("Agent get the temp_dict")
        self.step_info_dump = dict()
        return temp_dict


    def __get_replay_cfg(self):
        for key in self.agent_cfg['replay_cfg']['keys_from_agent']:
            self.agent_cfg['replay_cfg'].update({key : self.agent_cfg[key]})
        return self.agent_cfg['replay_cfg']

    def __get_exploration_cfg(self):
        for key in self.agent_cfg['exploration_cfg']['keys_from_agent']:
            self.agent_cfg['exploration_cfg'].update({key : self.agent_cfg[key]})
        return self.agent_cfg['exploration_cfg']


    def sample_random_action(self):
        self.logger.dbg_log("Agent sampling random action")
        action = super(DQN_Train_Agent, self).sample_random_action()
        action = action[0]
        return action 
    
    def chooseAction(self,obs):
        exploration_step_flag = self.exploration.exploration_step_flag() 
        if exploration_step_flag:
            action = self.sample_random_action()
            dump_qvalue = None
        else:
            self.logger.dbg_log("Agent Calculate best action")
            obs = torch.tensor(obs).float().detach()
            obs = obs.unsqueeze(0)  # add batch size
            qValues = self.dqn(obs) # pass it through the network to get your estimations            
            action = torch.argmax(qValues) # pick the highest
            action =  action.item() # return an int instead of a tensor containing the index of the best action
            dump_qvalue = torch.max(qValues).item()     

        self.step_info_dump['greedy'] = not exploration_step_flag
        self.step_info_dump['qvalue'] = dump_qvalue
        return action  

    def get_epsilon(self):
        epsilon = self.exploration.epsilon        
        return epsilon


    def learn(self,state, action, reward, next_state, done): 
        self.storeMemory(state, action, reward, next_state, done)
        if self.memory.memCount < self.min_replay_dim:
            loss = 0 
            self.step_info_dump['epsilon'] = self.get_epsilon()
            self.logger.dbg_log("Agent doesn't learn nothing in this step")
        else:
        
            state, action, reward, new_state, done = self.memory.sample(self.batch_size)

            state          = torch.tensor(state , dtype=torch.float32).to(self.dqn.device)
            action         = torch.tensor(action, dtype=torch.long   ).to(self.dqn.device)
            reward         = torch.tensor(reward, dtype=torch.float32).to(self.dqn.device)
            next_state      = torch.tensor(new_state, dtype=torch.float32).to(self.dqn.device)
            done           = torch.tensor(done  , dtype=torch.bool   ).to(self.dqn.device)

            batchIndices = np.arange(self.batch_size, dtype=np.int64)  # i learned how to spell indices
            qValue = self.dqn(state)[batchIndices, action]

            next_qValue = self.dqn(next_state)        #   values of all actions
            next_qValue = torch.max(next_qValue, dim=1)[0] #   extract greedy action value
            next_qValue[done] = 0.0                    #   filter out post-terminal states

            qTarget = reward + self.gamma * next_qValue
            loss = self.dqn.loss(qTarget, qValue)

            self.dqn.optimizer.zero_grad()
            loss.backward()
            self.dqn.optimizer.step()
            self.logger.dbg_log("Agent compute backpropagation")


            self.exploration.decay_exp()
            self.logger.dbg_log("Agent decay esploration.")


        self.step_info_dump['loss'] = loss
        self.step_info_dump['epsilon'] = self.get_epsilon()

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

    def save_checkpoint(self,episode):
        return self.dqn.save_model(episode)

    def load_checkpoint(self,ckp_name):
        self.dqn.load_model(ckp_name)

    def handle_kb_int(self):
        self.logger.info_log("Received keyboard interrupt. Closing Train Env")
        self.dqn.handle_kb_int()
        self.memory.handle_kb_int()  

    def dump_memory(self,episode):
        self.memory.dump_memory(episode)

class DQN_Eval_Agent(EvalAgent):

    def __init__(self,eval_cfg,logger):
        self.net_cfg = eval_cfg['dqn_net_cfg']
        self.state_name = "evaluation"
        self.dqn = Network(self.net_cfg,self.state_name,logger)

    
    def load_checkpoint(self,ckp_name):
        self.dqn.load_model(ckp_name)

    def sample_random_action(self):
        return super(DQN_Eval_Agent, self).sample_random_action()

    def chooseAction(self,obs):
        obs = torch.tensor(obs).float().detach()
        qValues = self.dqn(obs) # pass it through the network to get your estimations
        action = torch.argmax(qValues).item() # pick the highest return an int instead of a tensor containing the index of the best action

        return action  

    def handle_kb_int(self):
        self.logger.info_log("Received keyboard interrupt. Closing Eval Agent")


class DqnAgent(Agent):
    
    def __init__(self,cfg,info_env,logger):
        cfg.update(info_env)
        
        self.logger = logger
        self.cfg = cfg

        self.__get_net_cfg()

        ### Init Agents
        self.train_agent = DQN_Train_Agent(self.cfg,logger)
        self.eval_agent = DQN_Eval_Agent(self.cfg,logger)
        self.agent_state = self.train_agent 
        self.logger.info_log(f" Init Agent in state {self.agent_state.state_name}")
        

        self.ckp_dir = Path(self.cfg['experiment_folder']) / self.cfg['ckp_dirname']
        make_dir(self.ckp_dir)

      
        

    def __get_net_cfg(self):        
        self.dqn_cfg      = self.cfg['dqn_net_cfg']
        for key in self.dqn_cfg['keys_from_agent']:
            self.dqn_cfg.update({key : self.cfg[key]})

    def set_agent_state(self,flag):
        if flag == "Train":
            self.agent_state = self.train_agent 
        else:
            self.agent_state = self.eval_agent 


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

    
    def save_checkpoint(self,episode):
        if self.agent_state.state_name == "training":
            return self.train_agent.save_checkpoint(episode)        
        else:
            pass

    def get_info_dump(self):
        return self.agent_state.get_info_dump()

    def load_checkpoint(self,ckp_name):
        self.agent_state.load_checkpoint(ckp_name)

    def handle_kb_int(self):
        self.agent_state.handle_kb_int()

    def dump_memory(self,episode):
        if self.agent_state.state_name == "training":
            self.agent_state.dump_memory(episode)
        else:
            pass
