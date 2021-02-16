class Exp():

    def __init__(self,cfg,env,agent):
        self.env = env
        self.agent = agent

    def start(self):
        state = self.env.reset()
        action = self.env.sample_random_action()

        print(state)
