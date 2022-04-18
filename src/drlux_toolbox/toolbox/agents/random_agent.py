from drlux_toolbox.metaclasses.agent import Agent
from loguru import logger


class RandomAgent(Agent):
    def __init__(self, agentConfig):
        self.stateSpace = agentConfig["stateSpace"]
        self.actionSpace = agentConfig["actionSpace"]

    def choose_action(self, obs):
        return self.actionSpace.sample_action(), "info action"
