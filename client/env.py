import numpy as np
import os
import pathlib
import glob
import cv2
from collections import namedtuple

import gym
from gym import spaces

class Env007(gym.Env):
    """
    Action for each agent: [no gun, shoot player 1, player 2, player 3]
    other action: [shield, reload]
    """

    def __init__(self):
        self.action_space = spaces.Discrete(4)

        # Initialize Observations
        self.team1_healths = np.array([5,5,5])
        self.team1_ammos = np.array([0,0,0])
        self.team2_healths = np.array([5,5,5])
        self.team2_ammos = np.array([0,0,0])

        # Settings
        self.horizon = 1000


    def reset(self):
        self.team1_healths = np.array([5,5,5])
        self.team1_ammos = np.array([0,0,0])
        self.team2_healths = np.array([5,5,5])
        self.team2_ammos = np.array([0,0,0])

    def step(self, action):
        pass

    def get_obs(self):
        return self._obs.copy()
    
    def render(self):
        print('#############')
        print(f'Team 1 Healths: {self.team1_healths}')
        print(f'Team 1 Ammos: {self.team1_ammos}')
        print(f'Team 2 Healths: {self.team2_healths}')
        print(f'Team 2 Ammos: {self.team2_ammos}')
        print('#############')
