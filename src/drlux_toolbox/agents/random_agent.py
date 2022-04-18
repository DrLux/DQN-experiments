from drlux_toolbox.metaclasses.agent import Agent


class RandomAgent(Agent):
    def __init__(self):
        print("Agente creato")
        self.save()

    def choose_action(self, obs):
        return self.actionSpace.sample_action(), "info action"
