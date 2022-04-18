from collections import namedtuple
import numpy as np
import tqdm

from utils.profiler import timer 


# Aggiugnere un profiling automatico degli episodi
class Experiment:
    def __init__(self,env,agent,dumper,num_eps=1000):
        self.dumper = dumper
        self.env = env 
        self.agent = agent 
        self.num_eps = num_eps

    
    #@time_profiler
    def exe_experiment(self): 

        self.hook_init_experiment()

        for ep_count in tqdm.trange(1, self.num_eps +1):
            self.hook_init_episode()
            self.exe_episode()
            self.hook_end_episode(ep_count)

        self.exe_episode(True)

        self.hook_end_experiment()
        self.env.close()

    #@time_profiler
    def exe_episode(self, isTestEpisode=False):
        tr = namedtuple('Transition', ['obs','action','reward','done','new_obs'])

        # init episode stuff
        tr.obs = self.env.reset()
        tr.done = False

        # compute cycle inside episode
        if isTestEpisode:
            self.exe_test_cycle(tr)
        else:
            self.exe_cycle(tr)


    def exe_test_cycle(self,tr):
        step_counter = 0 
        
        while not tr.done:
            step_counter += 1
            self.env.render()
            tr.action, info_choice = self.agent.choose_action(tr.obs)
            tr.new_obs, tr.reward, tr.done, info_env = self.env.step(tr.action)
            tr.obs = tr.new_obs  
            self.hook_update_step(tr,step_counter,info_choice,info_env)

        
    def exe_cycle(self,tr):
        step_counter = 0 
        
        while not tr.done:
            step_counter += 1
            
            tr.action, info_choice = self.agent.choose_action(tr.obs)
            tr.new_obs, tr.reward, tr.done, info_env = self.env.step(tr.action)
            tr.obs = tr.new_obs  
            self.hook_update_step(tr,step_counter,info_choice,info_env)
        

    def hook_end_experiment(self):
        self.dumper.end_experiment()

    def hook_end_episode(self,ep_counter):
        info_agent_end = self.agent.ops_end_episode()
        
        info_update_model_episode = self.agent.update_model_episode(ep_counter)
        info_update_strategy_episode = self.agent.update_exp_strategy_episode(ep_counter)

        self.dumper.ops_end_episode(ep_counter, info_agent_end,
                                    info_update_model_episode, info_update_strategy_episode)   


    def hook_update_step(self,tr,step_counter,info_choice,info_env):
        info_agent_ops_step = self.agent.ops_step(tr)
        
        info_update_model_step = self.agent.update_model_step(step_counter)
        info_update_strategy_step = self.agent.update_exp_strategy_step(step_counter)
        
        self.dumper.ops_step(tr, step_counter,
                            info_agent_ops_step, info_choice, info_env,
                            info_update_model_step, info_update_strategy_step)
        

    def hook_init_episode(self):
        self.dumper.init_episode()
        self.agent.init_episode()


