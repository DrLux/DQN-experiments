import numpy as np

class vect_ReplayBuffer(): 
    def __init__(self, replay_cfg):

        replay_dim      = replay_cfg['replay_dim']
        obs_shape       = replay_cfg['obs_shape']
        obs_dtype       = replay_cfg['obs_dtype']
        action_dtype    = replay_cfg['action_dtype']

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