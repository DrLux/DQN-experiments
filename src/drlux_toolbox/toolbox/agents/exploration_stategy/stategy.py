import abc

class StrategyRandom(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargsr):
        pass

    @abc.abstractmethod
    def act(self, priority, action_space):
        """
        Method used by the Agent to choose a legal action during every step
        :param priority: priority to be given to each legal action in the action space
        :param action_space: actual action_space of the environment
        :return: tuple (action: index of the action chosen, info: dictionary about the exploration choice)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self):
        """
        Update the strategy parameters
        :return: info: dictionary about the exploration strategy
        """
        raise NotImplementedError()
