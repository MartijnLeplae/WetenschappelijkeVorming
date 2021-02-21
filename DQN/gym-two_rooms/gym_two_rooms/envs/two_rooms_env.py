import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random as rnd

# KerasRL imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory


class TwoRoomsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        '''
        Observation:
            Type: Discrete(1)
            Num     Observation               Range
            0       Room                      [0, 1]

            Room layout:
              |0 | 1 |

        Actions:
            Type: Discrete(2)
            Num   Action
            0     Go Left
            1     Go Right
        '''

        self.state = []
        self.n_rooms = 2
        self.steps_taken = 0
        self.step_limit = 200  # Limit of the nb. of actions in one episode
        self.episode_length = 10  # Nb of actions in one episode
        self.sequence = [round(rnd.random()) for _ in range(self.episode_length)]
        self.history_length = 5

        self.observation_space = spaces.Discrete(self.history_length)  # Room
        self.action_space = spaces.Discrete(2)  # Left (1) or Right (2)
        # self.history = [0 for _ in range(self.history_length)]

    def step(self, action):
        self.steps_taken += 1
        new_room = 0 if action == 0 else 1
        self.state.append(new_room)

        if len(self.state) > len(self.sequence):
            self.state = self.state[-len(self.sequence):]

        if self.state == self.sequence[:len(self.state)]:
            reward = 1
            if len(self.sequence) == self.steps_taken:
                reward = 5
        else:
            reward = -1

        done = self.steps_taken >= self.step_limit
        # self.current_hist_rep = list(self.history)

        # observation = (current room, last `history_length' nb. of  states)
        # s = new_room, self.history[-self.history_length:]

        return self._get_state_repr(), reward, done, {}

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    def _get_state_repr(self):
        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        if len(self.state) < self.history_length:
            return np.array([0]*(self.history_length-len(self.state)) + self.state)
        else:
            return np.array(self.state[-self.history_length:])

    def reset(self):
        self.steps_taken = 0
        self.state = []
        # self.history = [0 for _ in range(self.history_length)]
        # self.current_hist_rep = list(self.history)

        return self._get_state_repr()

    def render(self, mode='human'):
        # TODO: add graphical representation
        pass

    def close(self):
        pass
