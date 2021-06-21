
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

        self.env.close()


    def train(self):
        highScore = -100000
        episode = 0
        while True:
            done = False
            state = self.env.reset()

            score, frame = 0, 1
            while not done:
                self.env.render()

                action = self.agent.chooseAction(state)
                state_, reward, done = self.env.step(action)
                
                self.agent.learn(state, action, reward, state_, done)
                state = state_

                score += reward
                frame += 1

            highScore = max(highScore, score)

            print(( "ep {}: high-score {:12.3f}, "
                    "score {:12.3f}, last-episode-time {:4d}").format(
                episode, highScore, score,frame))

            episode += 1