from abc import ABC, abstractmethod
import numpy as np
 

    
class Agent(ABC):
    _agents =  {}
    
    @classmethod
    def register(cls,sub_class):
        cls._agents[sub_class.__name__] = sub_class

    @classmethod
    def select_agent(cls, agent_type, **kwargs):
        return cls._agents[agent_type](**kwargs)

    @abstractmethod
    def sample_random_action(self):
        action = np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(self.num_actions)).astype(self.action_dtype)
        if self.num_actions == 1:
            action = action[0]
        return action
    
    @abstractmethod
    def chooseAction(self,obs):
        raise NotImplementedError()

    
def registerAgent(cls):
    Agent.register(cls)
    return cls


class EvalAgent(Agent):
    @abstractmethod
    def chooseAction(self,obs):
        raise NotImplementedError()

    @abstractmethod
    def sample_random_action(self):
        action = np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(self.num_actions)).astype(self.action_dtype)
        if self.num_actions == 1:
            action = action[0]
        return action


class TrainAgent(Agent):
    @abstractmethod
    def chooseAction(self,obs):
        raise NotImplementedError()

    @abstractmethod
    def sample_random_action(self):
        action = np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(self.num_actions)).astype(self.action_dtype)
        if self.num_actions == 1:
            action = action[0]
        return action
