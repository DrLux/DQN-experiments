import abc


class Agent(metaclass=abc.ABCMeta):

    ##############################
    #  Methods that need to be implemented
    ##############################

    # @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    # @abc.abstractmethod
    def choose_action(self, obs):
        return self.actionSpace.sample_action(), "info action"

    # @abc.abstractmethod
    def ops_end_episode(self):
        return {}

    # @abc.abstractmethod
    def update_model_episode(self, ep_counter):
        return {}

    # @abc.abstractmethod
    def update_exp_strategy_episode(self, ep_counter):
        return {}

    # @abc.abstractmethod
    def update_exp_strategy_step(self, step_counter):
        return {}

    # @abc.abstractmethod
    def ops_step(self, tr):
        return {}

    # @abc.abstractmethod
    def update_model_step(self, step_counter):
        return {}

    # @abc.abstractmethod
    def init_episode(self):
        return {}

    # @abc.abstractmethod
    def init_experiment(self):
        return {}

    #########################################
    #   Optional Methods
    ########################################

    def save(self):
        print("ho salvato")
        pass
