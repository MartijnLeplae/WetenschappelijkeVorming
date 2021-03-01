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
        """
        Observations
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
        """

        self.state = []
        self.n_rooms = 2
        self.steps_taken = 0
        self.sequence = [1,1,1,0,1,0,1,1,1,1,0]
        # self.sequence = [round(rnd.random()) for _ in range(10)]
        self.episode_length = len(self.sequence)  # Nb of actions in one episode
        self.history_length = 3

        self.observation_space = spaces.Discrete(self.history_length)
        self.action_space = spaces.Discrete(2)  # Left (1) or Right (2)

    def step(self, action):
        self.steps_taken += 1
        done = self.steps_taken >= self.episode_length
        new_room = 0 if action == 0 else 1
        self.state.append(new_room)

        if len(self.state) > len(self.sequence):
            self.state = self.state[-len(self.sequence):]

        if self.state == self.sequence[:len(self.state)]:  # Newly added room is correct
            reward = 2
            if len(self.sequence) == len(self.state):  # We've learned the sequence successfully
                reward = 5
        elif ''.join(map(str, self.state)) in ''.join(map(str, self.sequence)) and len(self.state) > self.history_length:
            # Correct subsequence of the sequence to be learned, but not at the correct spot / in the big picture
            reward = 1
        else:
            reward = -1

        return self._get_state_repr(), reward, done, {}

        # self.current_hist_rep = list(self.history)

        # observation = (current room, last `history_length' nb. of  states)
        # s = new_room, self.history[-self.history_length:]

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

        print('Current: ' + ' '.join(map(str, self.state[:-1])), end=' ')
        print('\u001b[31m' + str(self.state[-1]) + '\u001b[0m')  # Output last digit in the color red
        print('Goal:    ' + ' '.join(map(str, self.sequence)))

    def close(self):
        pass
