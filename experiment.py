class Experiment():

    def __init__(self,cfg,env,agent,dumper,logger):
        self.env    = env
        self.logger = logger
        self.agent  = agent
        self.max_allowed_steps =  cfg['max_allowed_steps']
        #self.dumper
        #self.trainer
        #self.validation
        #self.test

    def start(self):
        state = self.env.reset()
        done = False
        while not done and self.max_allowed_steps > 0:
            action = self.agent.sample_random_action()
            new_state,rew,done = self.env.step(action)
            self.max_allowed_steps -= 1

        print("Finito")
        self.env.close()