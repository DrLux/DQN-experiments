import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

class Network(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.obs_shape = cfg['obs_shape']
        self.num_actions = cfg['num_actions']
        self.fc1Dims = cfg['fc1Dims']
        self.fc2Dims = cfg['fc2Dims']
        assert 1 == 2

        #   layers
        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.num_actions)  # output 1 outcome estimate per action

        #   pytorch stuff
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x