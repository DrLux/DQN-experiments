from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
#import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Dumper():
    def __init__(self,cfg_dumper,logger):
        self.logger = logger 
        self.dump_dir = Path(cfg_dumper['experiment_folder']) / cfg_dumper['dump_dirname']
        self.dump_dir.mkdir(parents=True,exist_ok=True) 
        self.writer = SummaryWriter(self.dump_dir)
        self.logger.info_log(f"Started tensorboard session at: {self.writer.get_logdir()}")
        self.labels_barplot = cfg_dumper['labels_barplot']
        self.names_to_dump = cfg_dumper['env_names'] + cfg_dumper['agent_names']
        
        self.step_grain_dump = cfg_dumper['step_grain_dump']
        self.episode_grain_dump = cfg_dumper['episode_grain_dump']
        self.experiment_grain_dump = cfg_dumper['experiment_grain_dump']
        self.dump_to_file = cfg_dumper['dump_to_file']

        self.data_to_dump = cfg_dumper['data_to_dump']


        if self.experiment_grain_dump:
            self.buffer = defaultdict(list)
        
    def close(self):
        self.writer.close()

    def handle_kb_int(self):
        self.close()

    def plot_scalar(self,label,y,x):
        self.writer.add_scalar(label, y, x)

    def plot_image(self,label,img,step):
        self.writer.add_image(label, img, step)

    def plot_text(self,cell_title,text,step):
        self.writer.add_text(cell_title,text,step)

    def dump_info(self,env_info_ep,step_agent_info,episode,total_steps):
        if self.step_grain_dump:
            self.dump_steps(env_info_ep,step_agent_info,total_steps)
            self.logger.dbg_log(f"Dumped step_grain_dump")
        if self.episode_grain_dump:
            self.dump_ep_info(env_info_ep,step_agent_info,episode)
            self.logger.dbg_log(f"Dumped episode_grain_dump")
        if self.experiment_grain_dump:
            self.dump_experiment(env_info_ep,step_agent_info,episode)
            self.logger.dbg_log(f"Dumped experiment_grain_dump")
        if self.dump_to_file:
            self.dump_data(env_info_ep,step_agent_info)
    
    
    def dump_data(self,env_info_ep,step_agent_info):
        env_info_ep.update(step_agent_info)

        for dtd in self.data_to_dump:
            self.buffer[dtd] += env_info_ep[dtd]
    
    def dump_steps(self,env_info_ep,step_agent_info,total_steps):
        env_info_ep.update(step_agent_info)
        start_index = total_steps - len(env_info_ep["reward"])

        for vtd in self.data_to_dump:
            for i, val in enumerate(np.array(env_info_ep[vtd])):
                #print(f"{vtd=} | {i=} | {start_index} |  {val=}")
                self.plot_scalar(f'steps/{vtd}', val, start_index+i)
    
    def dump_experiment(self,env_info_ep,step_agent_info,episode):
        # Collect info for entire experiment
        self.buffer['greedy'] = env_info_ep['greedy']

    def dump_ep_info(self,env_info_ep,step_agent_info,episode):
        rewards     = np.array(env_info_ep['reward'])
        losses      = np.array(step_agent_info['loss'])
        qupdates    = np.array(step_agent_info['qupdate'])
        epsilons    = np.array(step_agent_info['epsilon'])
        qvalues     = np.array(step_agent_info['qvalue'])


        # Env statistics
        self.plot_scalar(f"episode/length",len(rewards),episode)
        self.plot_scalar(f"episode/mean_reward",np.mean(rewards),episode)
        self.plot_scalar(f"episode/max_reward",np.amax(rewards),episode)
        self.plot_scalar(f"episode/min_reward",np.amin(rewards),episode)
        self.plot_scalar(f"episode/std_reward",np.std(rewards),episode)

        # Agent statistics  
        self.plot_scalar(f"episode/mean_epsilon",np.mean(epsilons), episode)
        self.plot_scalar(f"episode/mean_qvalues",np.mean(qvalues),  episode)
        self.plot_scalar(f"episode/mean_qupdates",np.mean(qupdates),episode)
        self.plot_scalar(f"episode/mean_losses",np.mean(losses),    episode)
        

    
    def analyze_episode(self,env_info_ep,step_agent_info,episode):    
        env_info_ep.update(step_agent_info)

        # Plot for each episode
        for dtd in self.data_to_dump:
            for i, val in enumerate(np.array(env_info_ep[dtd])):
                self.plot_scalar(f"analyze_episode_{episode}/{dtd}",val,i)
            

    def plot_experiment_info(self,episode):
        pass
        #for k,l in self.buffer.items():
        #    self._make_bar_plot(f"{k}_over_{episode}_eps",l,self.labels_barplot[k])
        #    self.logger.dbg_log(f"Dumper created {k}.")

    def _make_bar_plot(self,title,list, labels):

        store_path = self.dump_dir / (f"{title}.png")
        true_count = sum(list)
        false_count = len(list) - true_count

        plt.bar(labels, [true_count,false_count])
        plt.savefig(str(store_path))
        plt.close()

