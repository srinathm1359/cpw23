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

    The actions outputted by the agent is a dictionary of {agent_0: [], agent_1: [], agent_2: []}
    For each of three friendly agents:
        List value in dictionary of agent_i is vector of length 7 (with two ones), concatenating the following three
            One hot action for each agent: [no gun, shoot player 1, player 2, player 3] (i.e. [0, 1, 0, 0])
            One hot action among: [shield, reload] (either [0, 1] or [1, 0]) 
               ^ (in the case of shooting one player being the 1 in the above, either of [1 0] or [0 1] 
                    should result in no shielding or reloading happening)
            one non-one-hot value of how much ammo is used by agent, such as [7], [0], or [3]
             ^ assume arbitrary value if agent doesnt shoot
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