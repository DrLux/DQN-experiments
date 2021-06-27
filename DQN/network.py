import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class Network(torch.nn.Module):
    def __init__(self, cfg,mode,logger):
        super().__init__()
        self.obs_shape      = cfg['obs_shape']
        self.num_actions    = cfg['num_actions']
        self.fc1Dims        = cfg['fc1Dims']
        self.fc2Dims        = cfg['fc2Dims']
        self.lr             = cfg['lr']
        self.ckp_path       = cfg['ckp_path']
        if cfg['device']:
            self.device = cfg['device']
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.set_net_mode(mode)
        self.logger = logger 


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
        x = x.to(self.device) # and put it on the gpu/cpu

        # pass input to the net
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x    

    def set_net_mode(self,mode):
        if mode == "training":
            self.train()
        else:
            self.eval()

    def save_model(self,episode=None):
        ckp = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }

        if episode:
            ckp['episode'] = episode

        ckp_name = f"ep_{episode}.ckp"
        path = Path(self.ckp_path) / (ckp_name)
        torch.save(ckp, str(path))
        self.logger.info_log(f"Storing ckp at {path}")

        return ckp_name


    def load_model(self, ckp_name):
        path =  Path(self.ckp_path) / ckp_name
        checkpoint = torch.load(str(path))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.logger.info_log(f"Loading ckp from {path}")
        
        if "episode" in checkpoint:
            episode =  checkpoint['episode']
        else:
            episode = 0
        return episode