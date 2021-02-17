import gym
from gym import spaces

#skeleton for a new environment

class OTTWorld(gym.Env):
    def __init__(self):
        print('world initialised')

    def step(self):
        print('you took a step')

    def reset(self):
        print('world reset')

    def render(self, mode=None):
        pass
