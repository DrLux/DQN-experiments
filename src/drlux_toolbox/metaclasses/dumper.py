import abc


class Dumper:
    
    @abc.abstractmethod
    def init_experiment(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_episode(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def ops_step(self, tr, step_counter,
                 info_agent_ops_step, info_choice, info_env, info_model_step, info_strategy_step):
        raise NotImplementedError()

    @abc.abstractmethod
    def ops_episode(self, ep_counter,
                    info_agent_end, info_model_episode, info_strategy_episode):
        raise NotImplementedError()

    @abc.abstractmethod
    def end_experiment(self):
        raise NotImplementedError()