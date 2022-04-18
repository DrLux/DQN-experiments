import gym
import abc

class Environment(gym.Env):
    def __init__(self, *args, **kwargs):
        super(Environment, self).__init__()

    def close(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self,action):
        raise NotImplementedError()

    def set_seed(self,seed):
        raise NotImplementedError()

    def getActionSpace(self):
        raise NotImplementedError()

    def getStateSpace(self):
        raise NotImplementedError()

    def handle_kb_int(self):
        raise NotImplementedError()

    def render(self):
        raise NotImplementedError()

class StateSpace(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_range_high(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_range_low(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_dtype(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_obs_shape(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def show_state_env_info(self):
        raise NotImplementedError()

class ActionSpace(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample_random_action(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_action_space(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_n_acts(self):
        raise NotImplementedError()
    

    @abc.abstractmethod
    def get_dtype(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_range_high(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_range_low(self):
        raise NotImplementedError()