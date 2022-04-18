import gym
import abc

class Environment(metaclass=abc.ABCMeta):

    def __init__(self,env_id,seed):
        raise NotImplementedError()
        #self.env = gym.make('CartPole-v0') 
        #self.actionSpace = ActionSpace(self.env)
        #self.stateSpace = StateSpace()
        #self.set_seed(seed)
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError()
        #self.env.close()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
        #self.env.reset()

    @abc.abstractmethod
    def step(self,action):
        raise NotImplementedError()
        #return self.env.step(action)

    @abc.abstractmethod
    def set_seed(self,seed):
        raise NotImplementedError()

    @abc.abstractmethod
    def getActionSpace(self):
        #return self.actionSpace
        raise NotImplementedError()

    @abc.abstractmethod
    def getStateSpace(self):
        #return self.stateSpace
        raise NotImplementedError()

    @abc.abstractmethod
    def handle_kb_int(self):
        #self.close()
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self):
        #self.env.render()
        raise NotImplementedError()

class StateSpace(metaclass=abc.ABCMeta):
    def __init__(self):
        pass 

class ActionSpace(metaclass=abc.ABCMeta):
    def __init__(self,env):
        self.env = env

    @abc.abstractmethod
    def sample_action(self):
        return self.env.action_space.sample()