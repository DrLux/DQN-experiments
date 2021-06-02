import memory

class Agent:
    def __init__(self,cfg,experienceReplayBuffer):
        self.buffer = experienceReplayBuffer
        #self.action_size = cfg['action_size']

    def get_action(self,state):
        pass