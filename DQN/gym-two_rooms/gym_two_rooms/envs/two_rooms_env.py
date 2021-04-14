import math
from collections import Counter

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
        # self.sequence = [1,1,1,0,1,0,1,1,1,1,0]  # Example sequence with repeating subsequence
        self.sequence = [round(rnd.random()) for _ in range(10)]
        self.episode_length = len(self.sequence)  # Nb of actions in one episode
        self.history_length = 3
        self.TOGGLE = 1
        self.step_size = 1

        self.observation_space = spaces.Discrete(self.history_length)
        self.action_space = spaces.Discrete(2)  # Left (1) or Right (2)

    def set_user_parameters(self, **params: dict):
        """
        This method sets the parameters given to values
        given by the user. The parameters that are set are
        those used by testFile.py
        """
        
        assert params, "params variable can't be None"
        for p, val in params.items():
            setattr(self, p, val)

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
        elif ''.join(map(str, self.state)) in ''.join(map(str, self.sequence)) and len(
                self.state) > self.history_length:
            # Correct subsequence of the sequence to be learned, but not at the correct spot / in the big picture
            reward = 1
        else:
            reward = -1

        return self._get_state_repr(self.TOGGLE, self.step_size), reward, done, {}

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    """ 
    @param: slider: parameter ∈ [0,1]:
        0: state representation is done using usual state memorisation
        1: state representation is done using bag of words
        otherwise: 
            hybrid representation: slider represents the percentage of the representation that is given in a bag of words
    @param: step: parameter to specify how many states should be left in-between the sampled history states
    """

    def _get_state_repr(self, slider=0.0, step=1):
        if slider > 0:
            assert self.history_length >= self.n_rooms

        # Take last `history_length' states; equal to self.state if history_length > len(self.state)
        # And take step parameter into account
        state_slice = self.state[-(self.history_length * step)::step]

        ratio = math.floor(slider * len(state_slice))
        bag_part = state_slice[:ratio]
        counter = Counter(bag_part)
        bag_of_words = [counter[i] for i in range(self.n_rooms)]
        self.bag_ratio = len(bag_of_words)

        zeros = [0] * (self.history_length - len(state_slice) - len(bag_of_words))

        states = state_slice[-(self.history_length - len(bag_of_words)):]
        self.state_ratio = len(states)

        return np.array(bag_of_words + zeros + states)

        # Old version
        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        # if len(self.state) < self.history_length:
        #     return np.array([0]*(self.history_length-len(self.state)) + self.state)
        # else:
        #     return np.array(self.state[-self.history_length:])

    def reset(self):
        self.steps_taken = 0
        self.state = []

        return self._get_state_repr(self.TOGGLE)

    def render(self, mode='human'):
        print('Current: ' + ' '.join(map(str, self.state[:-1])), end=' ')
        print('\u001b[31m' + str(self.state[-1]) + '\u001b[0m')  # Output last digit in the color red
        if self.steps_taken == self.episode_length:
            print('Goal:    ' + ' '.join(map(str, self.sequence)))

    def close(self):
        pass

    def get_name(self):
        state_ratio = math.floor(self.TOGGLE * self.history_length)
        bag_ratio = self.history_length - state_ratio
        return f'Toggle:{self.TOGGLE}-States:{state_ratio}-BoW:{bag_ratio}-StepSize:{self.step_size}.png'
