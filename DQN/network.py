import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.obs_shape      = cfg['obs_shape']
        self.num_actions    = cfg['num_actions']
        self.fc1Dims        = cfg['fc1Dims']
        self.fc2Dims        = cfg['fc2Dims']
        self.lr             = cfg['lr']
        self.device         = cfg['device']
        self.create_model()


        #   pytorch stuff
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.to(self.device)
        self.__get_info_network()

    def __get_info_network(self):
        parms = [p for p in self.parameters()]
        #print(f"Parameters: {len(parms)}")
        print(f"obs_shape:      {self.obs_shape}")
        print(f"num_actions:    {self.num_actions}")

    def create_model(self):
        #   layers
        self.fc1 = nn.Linear(*self.obs_shape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.num_actions)  # output 1 outcome estimate per action


    def forward(self, x):
        # Preprocess input
        x = torch.tensor(x).float().detach()  # detach it so we dont backprop through it
        x = x.to(self.device) # and put it on the gpu/cpu
        x = x.unsqueeze(0)  # add batch size
        
        # pass input to the net
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x