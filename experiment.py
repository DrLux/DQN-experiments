class Experiment():

    #def __init__(self,cfg,env,agent,dumper,logger):
    def __init__(self,cfg,env,logger):
        self.env    = env
        self.logger = logger
        #self.agent  = agent
        #self.dumper
        #self.trainer
        #self.validation
        #self.test

    def start(self):
        state = self.env.reset()
        done = False
        #while not done:
        #    action = self.agent.sample_random_action()
        #    new_state,rew,done,info = self.env.step(action)
