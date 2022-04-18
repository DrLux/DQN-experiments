from drlux_toolbox.agents.random_agent import RandomAgent
#from env.env import Environment
#from config import Config
#from dumper import Dumper
#from loguru import logger
#from experiment import Experiment

# num_eps=100

#env = Environment()
agent = RandomAgent()
#dumper = Dumper()
#experiment = Experiment(env,agent,dumper,num_eps)


# if __name__ == '__main__':
#    try:
#        experiment.exe_experiment()
#    except KeyboardInterrupt:
#        logger.error('main_rlhev: Experiment Keyboard Interrupt')
#        #dumper.handle_kb_int()
#        env.handle_kb_int()
