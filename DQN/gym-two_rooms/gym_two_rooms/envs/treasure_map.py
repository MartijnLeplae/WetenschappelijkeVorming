import math
from collections import Counter

import gym
from gym import spaces
import numpy as np
import WetenschappelijkeVorming.DQN.TestableEnv as TestableEnv

# map actions to indices of self.actions (see below)
LEFT, RIGHT, UP, DOWN, INTERACT, RESET = 0, 1, 2, 3, 4, 5

# map rooms to corresponding numbers, too
TOP_ROOM, LEFT_ROOM, CENTER_ROOM, RIGHT_ROOM, BOTTOM_ROOM = 0, 1, 2, 3, 4

INVALID_ACTION = -2
MOVE = 0
ACQUIRED_ITEM = 2
NEUTRAL_ACTION = 0
SOLD_TREASURE = 8
RESET_PENALTY = -4


class TreasureMapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Observations
            Type: Discrete(1)
            Num     Observation               Range
            0       Room                      [0, nb_rooms]

            (
            1       Interactable              treasure=0, guide=1, map=2, equipment=3, jewelry=4
                                              At the moment there are no empty states (see below),
                                              so there's no None-observation either.
            )

            Treasure world layout:
                 | t |             | 0 |
              |g | m | e|   =   |1 | 2 | 3|
                 | j |             | 4 |
            where 0: t = treasure,
                  1: g = guide,
                  2: m = map,
                  3: e = equipment and
                  4: j = jewelry.

        Actions:
            Type: Discrete(6)
            Num   Action
            0     Go Left
            1     Go Right
            2     Go Up
            3     Go Down
            4     (Try to) Interact with an interactable object (treasure, guide, map, equipment or jewelry).
            5     Reset
        """

        self.state = []
        self.nb_rooms = 5
        self.steps_taken = 0
        self.sequence = [LEFT_ROOM, TOP_ROOM, BOTTOM_ROOM]
        self.sequence2 = [RIGHT_ROOM, TOP_ROOM, BOTTOM_ROOM]
        self.episode_length = 75  # len(self.sequence)  # Nb of actions in one episode
        self.history_length = 20
        self.repr_length = 4
        self.TOGGLE = 0
        self.step_size = 1

        self.room = CENTER_ROOM
        self.acquired_items = ['map']  # Keep a list of the items the agent has collected thus far

        self.observation_space = spaces.Discrete(self.history_length)
        self.actions = ['left', 'right', 'up', 'down', 'interact', 'reset']
        self.action_space = spaces.Discrete(6)  # move left, right, up, down; interact or reset

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
        room = self.room

        new_room = 0
        reward = 0

        # Determine new room:
        if action == LEFT:
            reward = MOVE
            if room not in [CENTER_ROOM, RIGHT_ROOM]:
                reward = INVALID_ACTION
                new_room = room
            elif room == CENTER_ROOM:
                new_room = LEFT_ROOM
            elif room == 3:
                new_room = CENTER_ROOM
        elif action == RIGHT:
            reward = MOVE
            if room not in [LEFT_ROOM, CENTER_ROOM]:
                reward = INVALID_ACTION
                new_room = room
            elif room == LEFT_ROOM:
                new_room = CENTER_ROOM
            elif room == CENTER_ROOM:
                new_room = RIGHT_ROOM
        elif action == UP:
            reward = MOVE
            if room not in [CENTER_ROOM, BOTTOM_ROOM]:
                reward = INVALID_ACTION
                new_room = room
            elif room == CENTER_ROOM:
                new_room = TOP_ROOM
            elif room == BOTTOM_ROOM:
                new_room = TOP_ROOM
        elif action == DOWN:
            reward = MOVE
            if room not in [TOP_ROOM, CENTER_ROOM]:
                reward = INVALID_ACTION
                new_room = room
            elif room == TOP_ROOM:
                new_room = CENTER_ROOM
            elif room == CENTER_ROOM:
                new_room = BOTTOM_ROOM
        elif action == INTERACT:
            if room == TOP_ROOM:  # Treasure room
                if 'guide' in self.acquired_items or 'equipment' in self.acquired_items:
                    if 'treasure' not in self.acquired_items:
                        self.acquired_items.append('treasure')
                        reward = ACQUIRED_ITEM
                    else:
                        reward = INVALID_ACTION
                else:
                    reward = INVALID_ACTION
            elif room == LEFT_ROOM:  # Guide room
                # Agent starts off with map in its self.acquired_items.
                if 'guide' not in self.acquired_items:
                    if 'equipment' not in self.acquired_items:
                        reward = ACQUIRED_ITEM
                    else:
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append('guide')
                else:
                    reward = INVALID_ACTION
            elif room == CENTER_ROOM:  # Map room, agent starts off with map in its self.acquired_items.
                reward = NEUTRAL_ACTION
            elif room == RIGHT_ROOM:  # Equipment room, agent starts off with map in its self.acquired_items.
                if 'equipment' not in self.acquired_items:
                    if 'guide' not in self.acquired_items:
                        reward = ACQUIRED_ITEM
                    else:
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append('equipment')
                else:
                    reward = INVALID_ACTION
            elif room == BOTTOM_ROOM:  # Jewelry room
                if ('guide' in self.acquired_items or 'equipment' in self.acquired_items) and \
                        'jewelry' not in self.acquired_items:
                    self.acquired_items.append('jewelry')
                    reward = SOLD_TREASURE
                else:
                    reward = INVALID_ACTION
        elif action == RESET:
            reward = RESET_PENALTY
            self.reset()

        self.state.append(new_room)
        self.room = new_room

        if len(self.state) > len(self.sequence):
            self.state = self.state[-len(self.sequence):]

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
            assert self.history_length >= self.nb_rooms

        # Take last `history_length' states; equal to self.state if history_length > len(self.state)
        # And take step parameter into account
        state_slice = self.state[-(self.history_length * step)::step]

        ratio = math.floor(slider * len(state_slice))
        bag_part = state_slice[:ratio]
        counter = Counter(bag_part)
        bag_of_words = [counter[i] for i in range(self.nb_rooms)]
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
        self.acquired_items = ['map']

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
