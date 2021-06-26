import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym       
import math
import numpy as np
import random

class ReplayBuffer():
    def __init__(self, maxSize, stateShape):
        self.memSize = maxSize
        self.memCount = 0

        self.stateMemory        = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.actionMemory       = np.zeros( self.memSize,               dtype=np.int64  )
        self.rewardMemory       = np.zeros( self.memSize,               dtype=np.float32)
        self.nextStateMemory    = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.doneMemory         = np.zeros( self.memSize,               dtype=np.bool   )

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

class Network(torch.nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent():
    def __init__(self, lr, inputShape, numActions):
        self.network = Network(lr, inputShape, numActions)
        self.memory = ReplayBuffer(maxSize=10000, stateShape=inputShape)
        self.minMemorySize = 1024
        self.batchSize = 64
        self.gamma = 0.99

        self.epsilon = 0.1
        self.epsilon_decay = 0.00005
        self.epsilon_min = 0.001

    def chooseAction(self, observation):
        if np.random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            state = torch.tensor(observation).float().detach()
            state = state.to(self.network.device)
            state = state.unsqueeze(0)

            qValues = self.network(state)
            action = torch.argmax(qValues).item()
        return action

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

    def learn(self):
        if self.memory.memCount < self.minMemorySize:
            return

        states, actions, rewards, states_, dones = self.memory.sample(self.batchSize)
        states  = torch.tensor(states , dtype=torch.float32).to(self.network.device)
        actions = torch.tensor(actions, dtype=torch.long   ).to(self.network.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.network.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.network.device)
        dones   = torch.tensor(dones  , dtype=torch.bool   ).to(self.network.device)

        batchIndices = np.arange(self.batchSize, dtype=np.int64)
        qValue = self.network(states)[batchIndices, actions]

        qValues_ = self.network(states_)
        qValue_ = torch.max(qValues_, dim=1)[0]
        qValue_[dones] = 0.0

        qTarget = reward + self.gamma * qValue_
        loss = self.network.loss(qTarget, qValue)

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

if __name__ == '__main__':
    env = gym.make('CartPole-v1').unwrapped
    agent = Agent(lr=0.001, inputShape=(4,), numActions=2)

    highScore = -math.inf
    episode = 0
    numSamples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            # env.render()

            action = agent.chooseAction(state)
            state_, reward, done, info = env.step(action)
            agent.storeMemory(state, action, reward, state_, done)
            agent.learn()
            
            state = state_

            numSamples += 1

            score += reward
            frame += 1

        highScore = max(highScore, score)

        print(( "ep {:4d}: high-score {:12.3f}, "
                "score {:12.3f}, epsilon {:5.3f}").format(
            episode, highScore, score, agent.epsilon))

        episode += 1