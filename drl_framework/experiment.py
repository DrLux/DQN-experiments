from pathlib import Path
from tqdm import tqdm
from utils.utils import set_seeds

class Experiment():

    def __init__(self,cfg,env,agent,dumper,logger):
        self.logger = logger
        set_seeds(cfg['seed'])
        self.logger.info_log(f"Set experiment seed at {cfg['seed']} \n")
        
        self.env    = env
        self.agent  = agent
        self.max_allowed_steps =  cfg['max_allowed_steps']
        self.total_train_episodes = cfg['total_train_episodes']
        self.total_test_episodes = cfg['total_test_episodes']
        self.dumper = dumper
        self.save_ckp_int = cfg['save_ckp_int']
        self.last_ckp_name = None
        
    def test(self):
        self.env.set_render(True)
        self.agent.set_agent_state("Test")
        self.logger.dbg_log(f"Experiment start test. Total episodes {self.total_test_episodes}")


        if self.last_ckp_name:
            self.agent.load_checkpoint(self.last_ckp_name)
            self.logger.dbg_log(f"Experiment load checkpoint {self.last_ckp_name}")
        else:
            ckp_folder = self.agent.cfg['dqn_cfg']['ckp_path']
            list_ckps = list(Path(ckp_folder).glob("*.ckp"))
            list_ckps.sort()
            if list_ckps != []:
                last_ckp_name = list_ckps[-1].name
                self.agent.load_checkpoint(str(last_ckp_name))
                self.logger.dbg_log(f"Experiment find itself checkpoint {self.last_ckp_name}")
            else:
                self.logger.dbg_log(f"Experiment do not load any checkpoint")


        for episode in tqdm(range(self.total_test_episodes)):
            self.logger.dbg_log(f"Experiment start test episode: {episode}")

            done = False
            state = self.env.reset()
            
            while not done:
                action = self.agent.chooseAction(state)
                new_state, reward, done = self.env.step(action)      
                state = new_state   
            

   
    def train(self):
        self.logger.dbg_log(f"Experiment start training. Total episodes {self.total_train_episodes}")

        self.env.set_render(False)
        episode = 0
        episode_info = dict()
        all_steps = 0

        for episode in tqdm(range(self.total_train_episodes+1)):
            
            done = False
            analize_episode = False
            state = self.env.reset()
            ep_step = 0

            ########
            #   INIT EPISODE
            #######
            while not done:
                self.logger.dbg_log(f"Experiment start train episode: {episode}")

                action = self.agent.chooseAction(state)
                new_state, reward, done = self.env.step(action)                
                self.agent.learn(state, action, reward, new_state, done)
                state = new_state
                ep_step +=1
                all_steps += 1

            #######
            # PLOT INFORMATION
            #######
            agent_info_ep = self.agent.get_ep_info_dump()
            env_info_ep = self.env.get_ep_info_dump()

            if episode == self.total_train_episodes:
                self.dumper.analyze_episode(env_info_ep,agent_info_ep,episode)

            self.dumper.dump_info(env_info_ep,agent_info_ep, episode,all_steps)

            if episode % self.save_ckp_int == 0 and episode != 0:
                self.last_ckp_name = self.agent.save_checkpoint(episode)
                self.agent.dump_memory(episode)
                self.agent.save_checkpoint(episode)
                self.logger.info_log(f"Episode {episode}. Create ckp {self.last_ckp_name} and dumper memory!")

            #print(f"{episode=} {ep_step=}")
        
        ##############
        # END EXPERIMENT
        ###############à
        self.dumper.plot_experiment_info(episode)
        self.agent.dump_memory(episode)
        self.agent.save_checkpoint(episode)

        self.env.close()
        self.dumper.close()
