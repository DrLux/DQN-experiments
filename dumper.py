from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
#import matplotlib.pyplot as plt
from collections import defaultdict




class Dumper():
    def __init__(self,cfg_dumper,logger):
        self.logger = logger 
        self.dump_dir = cfg_dumper['dump_dir']
        self.writer = SummaryWriter(self.dump_dir)
        self.logger.info_log(f"Started tensorboard session at: {self.writer.get_logdir()}")
        self.buffer = defaultdict(list)
                

    def close(self):
        self.writer.close()

    def plot_scalar(self,label,y,x):
        self.writer.add_scalar(label, y, x)

    def plot_image(self,label,img,step):
        self.writer.add_image(label, img, step)

    def plot_text(self,cell_title,text,step):
        self.writer.add_text(cell_title,text,step)

    def get_df_data(self, exp_id):
        experiment = tb.data.experimental.ExperimentFromDev(exp_id)
        df = experiment.get_scalars()
        print(df)

    def plot_step_info(self,step_experiment_info,step_agent_info):
        step_experiment_info.update(step_agent_info)
        
        self.buffer['greedy'].append(step_experiment_info['greedy'])

        self.plot_scalar("step/reward",step_experiment_info['reward'],step_experiment_info['step'])
        self.plot_scalar("step/loss",step_experiment_info['loss'],step_experiment_info['step'])
        self.plot_scalar("step/epsilon",step_experiment_info['epsilon'],step_experiment_info['step'])
        

        if step_experiment_info['done']:
            self.plot_scalar("step/done",step_experiment_info['done'],step_experiment_info['step'])
        
        if step_experiment_info['greedy']:
            self.plot_scalar("step/qvalue",step_experiment_info['qvalue'],step_experiment_info['step'])

    
    def plot_episode_info(self,episode_info):

        self.plot_scalar("episode/cump_rew",episode_info['cump_rew'],episode_info['episode'])
        self.plot_scalar("episode/length",episode_info['len'],episode_info['episode'])
        self.plot_scalar("episode/epsilon",episode_info['epsilon'],episode_info['episode'])


    def plot_experiment_info(self):
        for k,l in self.buffer.items():
            self._make_bar_plot(k,l)

    def _make_bar_plot(self, title,list):
        from collections import Counter
        import numpy as np
        import matplotlib.pyplot as plt
        store_path = self.dump_dir + f"/{title}.png"

        labels, values = zip(*Counter(list).items())

        indexes = np.arange(len(labels))
        width = 1

        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig(str(store_path))
        plt.close()