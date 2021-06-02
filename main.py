from cfg import CfgMaker
from log import Logger
import utils 


def init():
  cfg = CfgMaker()
  return cfg

if __name__ == "__main__":
  try:
    cfg = init()
    logger = Logger(cfg.make_cfg_logger())
  except KeyboardInterrupt:
    logger.handle_kb_int()