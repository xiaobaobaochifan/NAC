import numpy as np
# import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class DiscreteMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, nn.Identity)
        self.logits = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False, with_prob=False):
        # Pass the observation through the network
        net_out = self.net(obs)
        # Calculate the logits
        logits = self.logits(net_out)
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            # Choose the action that maximizes the probability
            actions = torch.argmax(probs, dim=-1)
        else:
            # Sample actions according to the probability distribution
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # Check which dimension is correct for actions input to env.step()
        
        if not with_prob:
            probs = None
            
        return actions, probs
    

# Define the Q-network (critic) for discrete action spaces
class DiscreteMLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        return self.q(obs)
    

class DiscreteMLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = DiscreteMLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = DiscreteMLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = DiscreteMLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
