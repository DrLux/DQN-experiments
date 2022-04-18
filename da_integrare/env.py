#import cv2
import numpy as np
import torch
import gym

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']

# controllare che i seed siano tutti fissati

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

def _images_to_observation(images, bit_depth):
    #images = TF.to_tensor(np.asarray(images.copy()))
    images = torch.tensor(images.copy().transpose(2, 0, 1), dtype=torch.float32)  #put channel first and technical fix with .copy()
    preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class GymEnv():
  
  def __init__(self, env_cfg):
    import logging
    gym.logger.set_level(logging.ERROR)  # Ignore warnings from Gym logger
    self.symbolic = env_cfg['symbolic']
    self._env = gym.make(env_cfg['env_name'])
    self._env.seed(env_cfg['seed'])
    self.max_episode_length = env_cfg['max_episode_length']
    self.action_repeat = env_cfg['action_repeat']
    self.bit_depth = env_cfg['bit_depth']

  def reset(self):
    # Reset internal timer
    self.t = 0  
    state = self._env.reset()
    if self.symbolic:
      #return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
      return np.expand_dims(np.array(state, dtype=np.float32), axis=0)
    #else:
    #  return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
  
  def step(self, action):
    #action = action.detach().numpy() deve arrivarmi già in numpy
    reward = 0
    done = False
    k = 0
    while k < self.action_repeat and done==False:
      k +=1 
      state, reward_k, done, _ = self._env.step(action)
      reward += reward_k
      self.t += 1  # Increment internal timer
      done = done or self.t == self.max_episode_length
    if self.symbolic:
      observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    else:
      observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
    return observation, reward, done

  def render(self):
    self._env.render()

  def close(self):
    self._env.close()

  @property
  def observation_size(self):
    return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

  @property
  def action_size(self):
    return self._env.action_space.shape[0]

  @property
  def action_space(self):
    return self._env.action_space.n 
    #return float(self._env.action_space.low[0]), float(self._env.action_space.high[0])

  # Sample an action randomly from a uniform distribution over all valid actions
  def sample_random_action(self):
    return self._env.action_space.sample()

# Wrapper for batching environments together
class EnvBatcher():
  def __init__(self, env_class, env_args, env_kwargs, n):
    self.n = n
    self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
    self.dones = [True] * n

  # Resets every environment and returns observation
  def reset(self):
    observations = [env.reset() for env in self.envs]
    self.dones = [False] * self.n
    return torch.cat(observations)

 # Steps/resets every environment and returns (observation, reward, done)
  def step(self, actions):
    done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
    observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
    dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
    self.dones = dones
    observations, rewards, dones = torch.cat(observations), torch.tensor(rewards, dtype=torch.float32), torch.tensor(dones, dtype=torch.uint8)
    observations[done_mask] = 0
    rewards[done_mask] = 0
    return observations, rewards, dones

  def close(self):
    [env.close() for env in self.envs]
