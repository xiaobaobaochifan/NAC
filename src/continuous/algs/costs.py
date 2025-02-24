import torch
import numpy as np
from torch.distributions.normal import Normal
from gymnasium.spaces import Box



class CostFunction:
    
    def __init__(self, env_fn, act_partition='default', state_gen_runs=100, lc=5, tc=0, device=torch.device('cpu')):
        self.env = env_fn()
        self.state_gen_runs = state_gen_runs
        self.lc = lc
        self.tc = tc
        self.device = device
        act_space = Box(low=self.env.action_space.low, high=self.env.action_space.high,
                    shape=self.env.action_space.shape, dtype=np.float32)
        self.act_dim = act_space.shape[0]
        if  act_partition == 'default':
            # Create a default tensor if none is provided
            self.act_partition = [0] + [float(100)]*(self.act_dim-1)
        else:
            # Use the provided tensor, ensure it is on the correct device
            self.act_partition = act_partition
        # self.act_limit = max(np.maximum(np.abs(self.env.action_space.high), np.abs(self.env.action_space.low)))


    def joint_probability(self, ac, obs):
        """
        Calculate the joint probability of each variable being below its threshold,
        returning a single scalar representing the product of all defined probabilities.
        """
        dist = ac.pi(obs)[2]
        mean = dist.mean[0]
        std = dist.stddev[0]

        dists = [Normal(mean[i], std[i]) for i in range(len(mean))]

        probs = 1  # Start with a scalar 1.0 for multiplication
        for i, threshold in enumerate(self.act_partition):
            if threshold != float('inf'):
                threshold_tensor = torch.tensor([threshold], dtype=mean.dtype)
                prob = dists[i].cdf(threshold_tensor)
                probs *= prob.item() 

        return probs
    
    def obs_sampling(self):

        obs_space = Box(low=self.env.observation_space.low, high=self.env.observation_space.high,
                    shape=self.env.observation_space.shape, dtype=np.float32)

        # Calculate the range for each dimension
        ranges = torch.tensor(np.minimum(obs_space.high, 100) - np.maximum(obs_space.low, -100))

        # Generate random samples from uniform distribution and scale them
        samples = torch.rand(self.state_gen_runs, obs_space.shape[0])
        samples = samples * ranges + torch.tensor(np.maximum(obs_space.low, -100))
        samples = torch.unbind(samples.unsqueeze(1))

        return samples

    def duo_leg(self, ac_old, ac_new):

        with torch.no_grad():
            samples = self.obs_sampling() 

            probs_old = torch.tensor([self.joint_probability(ac_old, sample) for sample in samples])
            probs_new = torch.tensor([self.joint_probability(ac_new, sample) for sample in samples])
            costs = self.lc * torch.abs(probs_old - probs_new) + self.tc * (torch.min(probs_old, probs_new)
                                                                            + torch.min(1-probs_old, 1-probs_new))
            cost = costs.mean()

        return cost
            
    def joint_probability_combine(self, ac, obs):
        # Extract the distribution parameters
        dist = ac.pi(obs)[2]

        thresholds = torch.tensor(self.act_partition)

        # Compute the CDF for all dimensions at once
        cdf_values = dist.cdf(thresholds)

        # Calculate the product of probabilities across all dimensions
        joint_prob = torch.prod(cdf_values)

        return joint_prob
    
    def duo_combine(self, ac_old, ac_new):

        with torch.no_grad():
            samples = self.obs_sampling() 

            probs_old = torch.tensor([self.joint_probability_combine(ac_old, sample) for sample in samples])
            probs_new = torch.tensor([self.joint_probability_combine(ac_new, sample) for sample in samples])
            costs = self.lc * torch.abs(probs_old - probs_new) + self.tc * (torch.min(probs_old, probs_new)
                                                                            + torch.min(1-probs_old, 1-probs_new))
            cost = costs.mean()

        return cost
    
    def duo(self, ac_old, ac_new):

        with torch.no_grad():
            samples = self.obs_sampling()

        dists_old = [ac_old.pi(obs)[2] for obs in samples]
        dists_new = [ac_new.pi(obs)[2] for obs in samples]

        thresholds = torch.tensor(self.act_partition)

        cdfs_old = torch.stack([dist.cdf(thresholds) for dist in dists_old]).squeeze(1)
        cdfs_new = torch.stack([dist.cdf(thresholds) for dist in dists_new]).squeeze(1)

        vols_old = torch.prod(cdfs_old, dim=1)
        vols_new = torch.prod(cdfs_new, dim=1)

        costs = self.lc * torch.abs(vols_old - vols_new)+ self.tc * (torch.min(vols_old, vols_new)
                                                                        + torch.min(1-vols_old, 1-vols_new))
        average_cost = torch.mean(costs)
        return average_cost


