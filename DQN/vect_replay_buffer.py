import numpy as np
from pathlib import Path

class vect_ReplayBuffer(): 
    def __init__(self, replay_cfg):

        replay_dim      = replay_cfg['replay_dim']
        obs_shape       = replay_cfg['obs_shape']
        obs_dtype       = replay_cfg['obs_dtype']
        action_dtype    = replay_cfg['action_dtype']
        self.mem_path   = replay_cfg['mem_path']


        self.memSize = replay_dim
        self.memCount = 0

        self.stateMemory        = np.zeros((self.memSize, *obs_shape), dtype=np.float32) #tuple unpacking in the array allocation -> *obs_shape
        self.actionMemory       = np.zeros( self.memSize,               dtype=np.int64)
        self.rewardMemory       = np.zeros( self.memSize,               dtype=np.float32)
        self.nextStateMemory    = np.zeros((self.memSize, *obs_shape), dtype=np.float32)
        self.doneMemory         = np.zeros( self.memSize,               dtype=np.bool)

    def storeMemory(self, state, action, reward, nextState, done):
        memIndex = self.memCount % self.memSize 
        
        self.stateMemory[memIndex]      = state
        self.actionMemory[memIndex]     = action
        self.rewardMemory[memIndex]     = reward
        self.nextStateMemory[memIndex]  = nextState
        self.doneMemory[memIndex]       = done

        self.memCount += 1

    def sample(self, sampleSize):
        memMax = min(self.memCount, self.memSize)
        batchIndecies = np.random.choice(memMax, sampleSize, replace=False)

        states      = self.stateMemory[batchIndecies]
        actions     = self.actionMemory[batchIndecies]
        rewards     = self.rewardMemory[batchIndecies]
        nextStates  = self.nextStateMemory[batchIndecies]
        dones       = self.doneMemory[batchIndecies]

        return states, actions, rewards, nextStates, dones

    def dump_memory(self, episode):
        mem_filename = "mem_{episode}.npz"
        path = Path(self.mem_path) / mem_filename 

        np.savez_compressed(str(path),
                            memSize = self.memSize,
                            memCount = self.memCount,
                            stateMemory = self.stateMemory,
                            actionMemory = self.actionMemory,
                            rewardMemory = self.rewardMemory,
                            nextStateMemory = self.nextStateMemory,
                            doneMemory = self.doneMemory)


        self.logger.info_log(f"Storing memory at {path}")
        
    def load_memory(self, full_path=None):
        if full_path is None:
            list_mems = list(Path(self.mem_path).glob("*.ckp"))
            list_mems.sort()
            assert list_mems != [], "Impossible to load any dumper memory" 
            mem_filename = list_mems[-1].name
            full_path = Path(self.mem_path) / mem_filename 
            
                

        raw_data = np.load(str(full_path), allow_pickle=True)
        
        self.memSize            = raw_data['memSize']
        self.memCount           = raw_data['memCount']
        self.stateMemory        = raw_data['stateMemory']
        self.actionMemory       = raw_data['actionMemory']
        self.rewardMemory       = raw_data['rewardMemory']
        self.nextStateMemory    = raw_data['nextStateMemory']
        self.doneMemory         = raw_data['doneMemory']
        
        self.logger.info_log(f"Recover memory from {full_path}")

    def handle_kb_int(self):
        path = ""
        self.logger.info_log(f"Received keyboard interrupt. Storing memory at {path}")

