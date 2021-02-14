import logging
import os
import time 

'''
How to use logger: 
    logger.info("")
    logger.error("")
    logger.warning("")
'''

def setup_logger(config):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('logger')
    log_handler = logging.FileHandler(os.path.join(config['log_dir'], config['log_file']))
    log_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(log_handler)
    logger.setLevel(logging.getLevelName(config['log_level']))

    #logger.info('Created logger')
    #logger.error('Logging error')

    return logger

def make_dir(dirpath):
  path = os.path.join(os.getcwd(), dirpath)
  if not os.path.isdir(path):
    os.mkdir(path)

class Chronometer():
  def __init__(self):
    self.result = dict()  

  def set_checkpoint(self, cname):
    self.result[cname] = time.time()  

  def take_time(self,cname):
    self.result[cname] = time.time() - self.result[cname] 

  def show_result(self):
    for k,v in self.result.items():
      print('Elapsed time for {} = {:1.5f}'.format(k,v))

  

