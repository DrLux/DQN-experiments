
class Experiment():

    def __init__(self,cfg,env,agent,dumper,logger):
        self.env    = env
        self.logger = logger
        self.agent  = agent
        self.max_allowed_steps =  cfg['max_allowed_steps']
        self.num_episodes = cfg['num_episodes']
        self.dumper = dumper
        
        
   
    def train(self):
        episode = 0
        episode_info = dict()
        all_steps = 0
        
        #tqdm
        for episode in range(self.num_episodes):
            done = False
            state = self.env.reset()

            cump_rew = 0
            step_episode = 0
            while not done:
                step_agent_info = None
                step_experiment_info = dict()

                action = self.agent.chooseAction(state)
                new_state, reward, done = self.env.step(action)                
                self.agent.learn(state, action, reward, new_state, done)
                state = new_state

                cump_rew += reward
                step_episode += 1
                all_steps += 1

                step_experiment_info['step'] = all_steps
                step_experiment_info['reward'] = reward
                step_experiment_info['done'] = done


                step_agent_info = self.agent.get_info_dump()
                self.dumper.plot_step_info(step_experiment_info,step_agent_info)

            print(( "ep {:4d}: score {:12.3f}, epsilon {:5.3f}").format(episode, cump_rew, self.agent.get_epsilon()))
            
            episode_info['episode'] = episode
            episode_info['cump_rew'] = cump_rew
            episode_info['len'] = step_episode
            episode_info['epsilon'] = self.agent.get_epsilon()

            self.dumper.plot_episode_info(episode_info)

        self.dumper.plot_experiment_info()
        
        self.env.close()
        self.dumper.close()