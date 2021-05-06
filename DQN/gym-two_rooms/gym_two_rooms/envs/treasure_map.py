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
ITEMS = [TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY]

INVALID_ACTION = -2
MOVE = 0
ACQUIRED_ITEM = 2
NEUTRAL_ACTION = 0
EFFICIENTLY_SOLD_TREASURE = 6
INEFFICIENTLY_SOLD_TREASURE = 4
RESET_PENALTY = -2

NB_PREV_STATES = 3  # 5  # 6



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
        self.episode_length = 20  # 300  # 75  # 25  # len(self.sequence)  # Nb of actions in one episode

        self.TOGGLE = 0
        self.step_size = 2

        # When a moves generates an observation:
        # self.sequence = [LEFT_ROOM, TOP_ROOM, BOTTOM_ROOM]
        # self.sequence2 = [RIGHT_ROOM, TOP_ROOM, BOTTOM_ROOM]

        # When a move doesn't generate an observation -> only interactions generate observations:
        self.sequence = [GUIDE, TREASURE, JEWELRY]
        self.sequence2 = [EQUIPMENT, TREASURE, JEWELRY]

        self.current_room = CENTER_ROOM
        self.acquired_items = [MAP]  # Keep a list of the items the agent has collected thus far

        # Use the get_state_repr method that uses a constant array size, or add extra elements to repr according to
        # the variables set underneath?
        self.use_in_place_repr = False  # False

        # Ideally, the agent would only need to take 3 actions to sell a treasure.
        self.repr_length = NB_PREV_STATES

        self.N_STATES = False  # False
        self.BOW = True  # False
        self.MOST_USED = False  # False
        self.INTERVAL = False  # True
        self.HISTORY_SUM = False

        if not self.use_in_place_repr:
            self.construct_repr_length()

        # self.nb_BOW_states = math.floor(self.TOGGLE * self.repr_length)
        self.nb_BOW_states = len(ITEMS)  # 0
        self.nb_regular_states = self.repr_length - self.nb_BOW_states

        # The observation consists of the current room the agent is in + history rep = 1 + self.repr_length
        self.observation_space = spaces.Discrete(self.repr_length + 1)  # spaces.Discrete(self.repr_length)
        # self.actions = ['left', 'right', 'up', 'down', 'interact', 'reset']
        self.actions = ['left', 'right', 'up', 'down', 'interact']
        self.action_space = spaces.Discrete(len(self.actions))  # move left, right, up, down; interact or reset

    def construct_repr_length(self):
        self.repr_length = 0
        if self.N_STATES:
            self.repr_length += NB_PREV_STATES
        if self.BOW:
            self.repr_length += len(ITEMS)
        if self.MOST_USED:
            self.repr_length += 1
        if self.INTERVAL:
            self.repr_length += NB_PREV_STATES
        if self.HISTORY_SUM:
            self.repr_length += 1
        self.observation_space = spaces.Discrete(self.repr_length + 1)  # spaces.Discrete(self.repr_length)

    def set_user_parameters(self, **params: dict):
        """
        This method sets the parameters given to values
        given by the user. The parameters that are set are
        those used by testFile.py
        """

        assert params, "params variable can't be None"
        for p, val in params.items():
            setattr(self, p, val)
        if not self.use_in_place_repr:
            self.construct_repr_length()

    # Using the obtained items as observations
    def step(self, action):
        self.steps_taken += 1
        # if len(self.state) >= len(self.sequence) or JEWELRY in self.acquired_items:
        if JEWELRY in self.acquired_items:
            self.state = []
            self.acquired_items = [MAP]
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
        done = (self.steps_taken >= self.episode_length)  # or JEWELRY in self.state

        return self._get_state_repr(self.TOGGLE, self.step_size), reward, done, {}

    """
    @param: slider: parameter ∈ [0,1]:
        0: state representation is done using usual state memorisation
        1: state representation is done using bag of words
        otherwise: 
            hybrid representation: slider represents the percentage of the representation that is given in a bag of words
    @param: step: parameter to specify how many states should be left in-between the sampled history states
    """

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    def _get_state_repr(self, slider=0.0, step=1):
        if self.use_in_place_repr:
            return self.get_in_place_state_repr(slider, step)
        return self.get_extended_state_repr()

    def get_in_place_state_repr(self, slider, step):
        if slider > 0:
            assert self.repr_length >= self.nb_rooms

        # Take last `NB_PREV_STATES` states; equal to self.state if NB_PREV_STATES > len(self.state)
        # And take step parameter into account
        state_slice = self.state[-(self.repr_length * step)::step]

        ratio = math.floor(slider * len(state_slice))
        # bag_part = state_slice[:ratio]

        # counter = Counter(bag_part)
        # bag_of_words = [counter[i] for i in range(len(bag_part))]
        bag_of_words = [self.state.count(i) for i in range(len(ITEMS))][:self.nb_BOW_states]
        # self.nb_BOW_states = len(bag_of_words)

        states = state_slice[-(self.repr_length - self.nb_BOW_states):]
        self.nb_regular_states = len(states)

        zeros = [len(ITEMS)] * (self.repr_length - self.nb_BOW_states - self.nb_regular_states)
        # with open('step_output.txt', mode='a') as f:
        #     f.write(str([self.current_room]) + " " + str(zeros) + " " + str(bag_of_words) + " " + str(states) + "state: " + str(self.state) + "(slice: " + str(state_slice) + ' )\n')

        return np.array([self.current_room] + zeros + bag_of_words + states)

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    def get_extended_state_repr(self):
        def get_bow():
            count = []
            for i in range(len(ITEMS)):
                count.append(self.state.count(i))
            return np.array(count)

        def most_used():
            count = get_bow()
            return np.array([np.argmax(np.array(count)) + 1])

        def get_hist_interval():
            interval = -self.step_size  # -2
            hist = []
            lim = max(-1, len(self.state) - NB_PREV_STATES * 2)
            for i in range(len(self.state) - 1, lim, interval):
                hist.append(self.state[i])
            result = np.array([0] * (NB_PREV_STATES - len(hist)) + hist)
            return result

        def get_states():
            if len(self.state) < NB_PREV_STATES:
                return np.array([0] * (NB_PREV_STATES - len(self.state)) + self.state)
            else:
                return np.array(self.state[-NB_PREV_STATES:])

        def get_hist_sum():
            count = get_bow()
            return np.array([np.nansum(count)])

        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        # 1. construct states vector
        states = np.array([])
        if self.N_STATES:
            states = get_states()
        # 2. Construct count vector
        bow = np.array([])
        if self.BOW:
            bow = get_bow()
        # 3. Construct most-used vector
        mu = np.array([])
        if self.MOST_USED:
            mu = np.array(most_used())
        # 4. Construct interval vector
        inter = np.array([])
        if self.INTERVAL:
            inter = get_hist_interval()
        hist_sum = np.array([])
        if self.HISTORY_SUM:
            hist_sum = get_hist_sum()

        return np.concatenate([[self.current_room], states, bow, mu, inter, hist_sum])

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
        time = datetime.now().strftime('%H:%M:%S:%f %d-%m-%Y')
        if self.use_in_place_repr:
            return f'BASELINE_BOW_INPLACE' \
                   + f'EpsLen:{self.episode_length}-' \
                   + f'States:{self.repr_length - self.nb_BOW_states}-' \
                   + f'BoW:{self.nb_BOW_states}-' \
                   + f'StepSize:{self.step_size}-' \
                   + f'{time}'
        else:
            name = 'epsgr' + str(self.episode_length)
            if self.N_STATES:
                name += f'States:{self.repr_length}'
            if self.INTERVAL:
                name += f'interval:{self.repr_length}'
            if self.BOW:
                name += 'bow'
            if self.MOST_USED:
                name += 'mu'
            name += '-' + time
            return name

#            return f'BASELINE_BOW-' \
#                + f'EpsLen:{self.episode_length}-' \
#                + f'N_States:{self.N_STATES}:{self.repr_length}-' \
#                + f'BoW:{self.BOW}-' \
#                + f'Interval:{self.INTERVAL}-' \
#                + f'MU:{self.MOST_USED}-' \
#                + f'HIST_SUM:{self.HISTORY_SUM}-' \
#                + f'{time}'

# + f'Toggle:{self.TOGGLE}-'
