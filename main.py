import cfg as configuration
import utils 




def init():
  cfg = configuration.CfgMaker()
  logger = utils.setup_logger(cfg.cfg_logger())
  
  

def main():
  init()
  test_pool()


if __name__ == "__main__":
    main()