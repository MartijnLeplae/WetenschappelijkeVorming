import math
from collections import Counter
from datetime import datetime
import gym
from gym import spaces
import numpy as np

# map actions to indices of self.actions (see below)
LEFT, RIGHT, UP, DOWN, INTERACT, RESET = 0, 1, 2, 3, 4, 5

# map rooms to corresponding numbers, too
TOP_ROOM, LEFT_ROOM, CENTER_ROOM, RIGHT_ROOM, BOTTOM_ROOM = 0, 1, 2, 3, 4

# Variable names for the items:
# TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY = 'treasure', 'guide', 'map', 'equipment', 'jewelry'
TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY = 0, 1, 2, 3, 4

INVALID_ACTION = -2
MOVE = 0
ACQUIRED_ITEM = 2
NEUTRAL_ACTION = 0
EFFICIENTLY_SOLD_TREASURE = 6
INEFFICIENTLY_SOLD_TREASURE = 4
RESET_PENALTY = -2


# TODO
#  Try: done = JEWELRY in self.state
#  Decide between:
#   - No observation when the agent moves, or
#   - An observation when the agent moves.
#  Remove map as given item at start to increase difficulty (when the easy version works).


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
        # Ideally, the agent would only need to take 3 actions to sell a treasure.
        self.episode_length = 75  # 75  # len(self.sequence)  # Nb of actions in one episode
        # self.history_length = 20
        self.repr_length = 3  # 8
        self.TOGGLE = 0
        self.nb_BOW_states = math.floor(self.TOGGLE * self.repr_length)
        self.nb_regular_states = self.repr_length - self.nb_BOW_states
        self.step_size = 1

        # When a moves generates an observation:
        # self.sequence = [LEFT_ROOM, TOP_ROOM, BOTTOM_ROOM]
        # self.sequence2 = [RIGHT_ROOM, TOP_ROOM, BOTTOM_ROOM]

        # When a move doesn't generate an observation -> only interactions generate observations:
        self.sequence = [GUIDE, TREASURE, JEWELRY]
        self.sequence2 = [EQUIPMENT, TREASURE, JEWELRY]

        self.current_room = CENTER_ROOM
        self.acquired_items = [MAP]  # Keep a list of the items the agent has collected thus far

        self.observation_space = spaces.Discrete(self.repr_length)
        self.actions = ['left', 'right', 'up', 'down', 'interact', 'reset']
        # self.actions = ['left', 'right', 'up', 'down', 'interact']
        self.action_space = spaces.Discrete(len(self.actions))  # move left, right, up, down; interact or reset

    def set_user_parameters(self, **params: dict):
        """
        This method sets the parameters given to values
        given by the user. The parameters that are set are
        those used by testFile.py
        """

        assert params, "params variable can't be None"
        for p, val in params.items():
            setattr(self, p, val)

    # Using the obtained items as observations
    def step(self, action):
        self.steps_taken += 1
        if len(self.state) >= len(self.sequence):
            self.state = []
            self.acquired_items = [MAP]
        # if JEWELRY in self.acquired_items:
        #     self.state = []
        #     self.acquired_items = [MAP]
        room = self.current_room

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
            elif room == RIGHT_ROOM:
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
                new_room = CENTER_ROOM
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
                if GUIDE in self.acquired_items or EQUIPMENT in self.acquired_items:
                    if TREASURE not in self.acquired_items:
                        self.acquired_items.append(TREASURE)
                        reward = ACQUIRED_ITEM
                        self.state.append(TREASURE)  # TODO should the agent also observe something he already has?
                    else:
                        reward = NEUTRAL_ACTION
                else:
                    reward = INVALID_ACTION
            elif room == LEFT_ROOM:  # Guide room
                # Agent starts off with map in its self.acquired_items.
                if GUIDE not in self.acquired_items:
                    if EQUIPMENT not in self.acquired_items:  # has neither GUIDE nor EQUIPMENT
                        reward = ACQUIRED_ITEM
                    else:
                        # already has EQUIPMENT, but not GUIDE
                        # Obtaining GUIDE when already having EQUIPMENT is not efficient
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append(GUIDE)
                    self.state.append(GUIDE)
                else:
                    reward = INVALID_ACTION
            elif room == CENTER_ROOM:  # Map room, agent starts off with map in its self.acquired_items.
                reward = NEUTRAL_ACTION  # Map doesn't play a role yet as of right now
                # self.state.append(MAP)
            elif room == RIGHT_ROOM:  # Equipment room, agent starts off with map in its self.acquired_items.
                if EQUIPMENT not in self.acquired_items:
                    if GUIDE not in self.acquired_items:
                        reward = ACQUIRED_ITEM
                    else:
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append(EQUIPMENT)
                    self.state.append(EQUIPMENT)
                else:
                    reward = INVALID_ACTION
            elif room == BOTTOM_ROOM:  # Jewelry room
                if bool(GUIDE in self.acquired_items) ^ bool(EQUIPMENT in self.acquired_items):
                    # n.b.: ^ is the XOR operator for booleans
                    if TREASURE not in self.acquired_items:
                        # No treasure to sell!
                        reward = INVALID_ACTION
                    else:
                        # Reward efficient decisions (i.e. acquiring only either EQUIPMENT or GUIDE to obtain TREASURE)
                        reward = EFFICIENTLY_SOLD_TREASURE
                        self.acquired_items.append(JEWELRY)
                        self.state.append(JEWELRY)
                        # done = True
                elif GUIDE in self.acquired_items and EQUIPMENT in self.acquired_items:
                    if TREASURE not in self.acquired_items:
                        # No treasure to sell!
                        reward = INVALID_ACTION
                    else:
                        # Sold treasure, but in an inefficient manner
                        reward = INEFFICIENTLY_SOLD_TREASURE
                        # Treasure can only be found with EQUIPMENT or GUIDE
                        self.acquired_items.append(JEWELRY)
                        self.state.append(JEWELRY)
                        # done = True
                else:  # Hasn't acquired EQUIPMENT nor GUIDE
                    reward = INVALID_ACTION
        elif action == RESET:
            reward = RESET_PENALTY
            self.reset()

        self.current_room = new_room
        # done = self.steps_taken >= self.episode_length
        done = JEWELRY in self.state
        if done:
            with open('state_log', mode='w') as f:
                f.write(str(self.state) + '\n')
            print(self.state)

        return self._get_state_repr(self.TOGGLE, self.step_size), reward, done, {}

    # Using the visited rooms as observations
    def step_original(self, action):
        self.steps_taken += 1
        done = self.steps_taken >= self.episode_length
        # done = JEWELRY in self.acquired_items
        room = self.current_room

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
            elif room == RIGHT_ROOM:
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
                new_room = CENTER_ROOM
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
                if GUIDE in self.acquired_items or EQUIPMENT in self.acquired_items:  # TODO set this to XOR
                    if TREASURE not in self.acquired_items:
                        self.acquired_items.append(TREASURE)
                        reward = ACQUIRED_ITEM
                    else:
                        reward = NEUTRAL_ACTION
                else:
                    reward = INVALID_ACTION
            elif room == LEFT_ROOM:  # Guide room
                # Agent starts off with map in its self.acquired_items.
                if GUIDE not in self.acquired_items:
                    if EQUIPMENT not in self.acquired_items:  # has neither GUIDE nor EQUIPMENT
                        reward = ACQUIRED_ITEM
                    else:
                        # already has EQUIPMENT, but not GUIDE
                        # Obtaining GUIDE when already having EQUIPMENT is not efficient
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append(GUIDE)
                else:
                    reward = INVALID_ACTION
            elif room == CENTER_ROOM:  # Map room, agent starts off with map in its self.acquired_items.
                reward = NEUTRAL_ACTION  # Map doesn't play a role yet as if right now
            elif room == RIGHT_ROOM:  # Equipment room, agent starts off with map in its self.acquired_items.
                if EQUIPMENT not in self.acquired_items:
                    if GUIDE not in self.acquired_items:
                        reward = ACQUIRED_ITEM
                    else:
                        reward = NEUTRAL_ACTION
                    self.acquired_items.append(EQUIPMENT)
                else:
                    reward = INVALID_ACTION
            elif room == BOTTOM_ROOM:  # Jewelry room
                if bool(GUIDE in self.acquired_items) ^ bool(EQUIPMENT in self.acquired_items):
                    # n.b.: ^ is the XOR operator for booleans
                    if TREASURE not in self.acquired_items:
                        # No treasure to sell!
                        reward = INVALID_ACTION
                    else:
                        # Reward efficient decisions (i.e. only acquiring either EQUIPMENT or GUIDE to obtain TREASURE)
                        reward = EFFICIENTLY_SOLD_TREASURE
                        self.acquired_items.append(TREASURE)
                        # done = True
                elif GUIDE in self.acquired_items and EQUIPMENT in self.acquired_items:
                    if TREASURE not in self.acquired_items:
                        # No treasure to sell!
                        reward = INVALID_ACTION
                    else:
                        # Sold treasure, but in an inefficient manner
                        reward = INEFFICIENTLY_SOLD_TREASURE
                        # Treasure can only be found with EQUIPMENT or GUIDE
                        self.acquired_items.append(TREASURE)
                        # done = True
                else:  # Hasn't acquired EQUIPMENT nor GUIDE
                    reward = INVALID_ACTION
        elif action == RESET:
            reward = RESET_PENALTY
            self.reset()

        self.state.append(new_room)
        self.current_room = new_room

        # This has the side effect of restricting the repr_length of the agent to `len(self.sequence)` == 3
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
            assert self.repr_length >= self.nb_rooms

        # Take last `history_length' states; equal to self.state if history_length > len(self.state)
        # And take step parameter into account
        state_slice = self.state[-(self.repr_length * step)::step]

        ratio = math.floor(slider * len(state_slice))
        bag_part = state_slice[:ratio]
        counter = Counter(bag_part)
        # bag_of_words = [counter[i] for i in range(self.nb_rooms)]
        bag_of_words = [counter[i] for i in range(len(bag_part))]
        self.nb_BOW_states = len(bag_of_words)

        # zeros = [0] * (self.history_length - len(state_slice) - len(bag_of_words))

        states = state_slice[-(self.repr_length - self.nb_BOW_states):]
        self.nb_regular_states = len(states)

        zeros = [0] * (self.repr_length - self.nb_BOW_states - self.nb_regular_states)

        return np.array(zeros + bag_of_words + states)

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
        self.acquired_items = [MAP]

        return self._get_state_repr(self.TOGGLE)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_name(self):
        time = datetime.now().strftime('%H:%M %d-%m-%Y')
        return f'Episode-Length:{self.episode_length}-' \
               + f'Toggle:{self.TOGGLE}-' \
               + f'States:{self.nb_regular_states}-' \
               + f'BoW:{self.nb_BOW_states}-' \
               + f'StepSize:{self.step_size}-' \
               + f'{time}.png'
