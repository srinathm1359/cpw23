import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """
    The agent takes in a state defined by the following information:
    1. The healths of everyone in the opponent team
    2. The healths of everyone in our team
    3. The ammos of everyone in the opponent team
    4. The ammos of everyone in our team
    """
    def __init__(self, envs):
        super().__init__()
        self.our_health_encoder = nn.Sequential(
            layer_init(nn.Linear(3, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 16)),
        )
        self.opp_health_encoder = nn.Sequential(
            layer_init(nn.Linear(3, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 16)),
        )
        self.our_ammo_encoder = nn.Sequential(
            layer_init(nn.Linear(3, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 16)),
        )
        self.opp_ammo_encoder = nn.Sequential(
            layer_init(nn.Linear(3, 16)),
            nn.Tanh(),
            layer_init(nn.Linear(16, 16)),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)