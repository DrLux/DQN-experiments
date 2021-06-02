import logging
import time 
import tracemalloc
from pathlib import Path


class Logger():
    def __init__(self, cfg):
        path = Path(cfg['log_dir']) / Path(cfg['log_file'])
        self.logger = logging.getLogger('logger')
        log_handler = logging.FileHandler(str(path), "w")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(cfg['log_level']) #debug, info, warning, error, critical
        tracemalloc.start()
        self.info_log("Start logger")
    
    def dbg_log(self, text, params=None):
        memory_used, _ = tracemalloc.get_traced_memory()
        memory_used = memory_used / 10 ** 6
        if params:
            self.logger.debug(f' Ram: {memory_used} MB | {text} \n Dump: {params}')
        else:
            self.logger.debug(f' Ram: {memory_used} MB | {text}')

    def info_log(self,text,params=None):
        if params:
            self.logger.info(f' {text} \n Dump: {params}')
            print(f' {text} \n Dump: {params}')
        else:
            print(text)
            self.logger.info(text)


    def close(self):
        self.info_log("Closing logger")
        tracemalloc.stop()

    def handle_kb_int(self):
        self.info_log("Received keyboard interrupt")
        self.close()

