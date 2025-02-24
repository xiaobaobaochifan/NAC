import torch
import numpy as np
from torch.distributions.normal import Normal
from gymnasium.spaces import Box, Discrete



class CostFunction:
    
    def __init__(self, env_fn, act_partition='default', state_gen_runs=100, lc=5, tc=0.1, device=torch.device('cpu')):
        self.env = env_fn()
        self.state_gen_runs = state_gen_runs
        self.lc = lc
        self.tc = tc
        self.device = device
        act_space = Discrete(n=self.env.action_space.n)
        self.act_dim = act_space.n
        if act_partition == 'default':
            # Create a default tensor if none is provided
            self.act_partition = torch.cat([torch.ones(self.act_dim // 2, device=device, dtype=torch.float32), 
                                            torch.zeros(self.act_dim - self.act_dim // 2, device=device, dtype=torch.float32)])        
        else:
            # Use the provided tensor, ensure it is on the correct device
            self.act_partition = torch.tensor(act_partition, device=device, dtype=torch.float32)
        
    def obs_sampling(self):

        obs_space = Box(low=self.env.observation_space.low, high=self.env.observation_space.high,
                    shape=self.env.observation_space.shape, dtype=np.float32)

        # Calculate the range for each dimension
        ranges = torch.tensor(np.minimum(obs_space.high, 100) - np.maximum(obs_space.low, -100))

        # Generate random samples from uniform distribution and scale them
        samples = torch.rand(self.state_gen_runs, obs_space.shape[0], device=self.device, dtype=torch.float32)
        samples = samples * ranges + torch.tensor(np.maximum(obs_space.low, -100), device=self.device, dtype=torch.float32)

        return samples
    
    def duo(self, ac_old, ac_new):

        with torch.no_grad():
            samples = self.obs_sampling()

        _, dists_old = ac_old.pi(samples, with_prob=True)
        _, dists_new = ac_new.pi(samples, with_prob=True)

        thresholds = self.act_partition

        vols_old = torch.sum(dists_old * thresholds, dim=-1)
        vols_new = torch.sum(dists_new * thresholds, dim=-1)

        costs = self.lc * torch.abs(vols_old - vols_new) + self.tc * (torch.min(vols_old, vols_new) + torch.min(1-vols_old, 1-vols_new))
        average_cost = torch.mean(costs)

        return average_cost


