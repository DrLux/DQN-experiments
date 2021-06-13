import logging
import time 
import tracemalloc
from pathlib import Path


class Logger():
    def __init__(self, cfg):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        tracemalloc.start()
        self.DEV_LEVEL = cfg['DEV_LEVEL']

        if self.DEV_LEVEL == "DEVELOPMENT":        
            # DEBUG logger
            self.debug_logger = logging.getLogger('debug_logger')
            path = Path(cfg['log_dir']) / Path(cfg['dbg_log_file'])
            dbg_log_handler = logging.FileHandler(str(path), "w")
            dbg_log_handler.setFormatter(formatter)
            self.debug_logger.addHandler(dbg_log_handler)
            self.debug_logger.setLevel(logging.DEBUG) #debug, info, warning, error, critical
            self.dbg_log("Start debug logger")
        
        # INFO logger
        self.info_logger = logging.getLogger('info_logger')
        path = Path(cfg['log_dir']) / Path(cfg['info_log_file'])
        info_log_handler = logging.FileHandler(str(path), "w")
        info_log_handler.setFormatter(formatter)
        self.info_logger.addHandler(info_log_handler)
        self.info_logger.setLevel(logging.INFO) #debug, info, warning, error, critical
        self.info_log("Start info logger")

        

        
    
    def dbg_log(self, text, params=None):
        memory_used, _ = tracemalloc.get_traced_memory()
        memory_used = memory_used / 10 ** 6
        if params:
            self.debug_logger.debug(f' Ram: {memory_used} MB | {text} \n Dump: {params}')
        else:
            self.debug_logger.debug(f' Ram: {memory_used} MB | {text}')

    def info_log(self,text,params=None):
        if self.DEV_LEVEL == "DEVELOPMENT":  
            self.dbg_log(text)

        if params:
            self.info_logger.info(f' {text} \n Params: {params}')
            print(f' {text} \n Params: {params}')
        else:
            print(text)
            self.info_logger.info(text)


    def close(self):
        self.info_log("Closing logger")
        tracemalloc.stop()

    def handle_kb_int(self):
        self.info_log("Received keyboard interrupt")
        self.close()

