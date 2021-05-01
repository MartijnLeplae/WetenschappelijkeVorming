import math
from datetime import datetime
import gym
from gym import spaces
import numpy as np
from random import choice

# map actions to indices of self.actions (see below)
LEFT, RIGHT, UP, DOWN, INTERACT, RESET = 0, 1, 2, 3, 4, 5

# map rooms to corresponding numbers, too
TOP_ROOM, LEFT_ROOM, CENTER_ROOM, RIGHT_ROOM, BOTTOM_ROOM = 0, 1, 2, 3, 4
ROOMS = [TOP_ROOM, LEFT_ROOM, CENTER_ROOM, RIGHT_ROOM, BOTTOM_ROOM ]

# Variable names for the items:
# TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY = 'treasure', 'guide', 'map', 'equipment', 'jewelry'
TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY = 0, 1, 2, 3, 4
ITEMS = [TREASURE, GUIDE, MAP, EQUIPMENT, JEWELRY]

# Rewards
INVALID_ACTION = DUPLICATE_ITEM = -2
MOVE = -1
ACQUIRED_ITEM = 2
NEUTRAL_ACTION = 0
EFFICIENTLY_SOLD_TREASURE = 6
INEFFICIENTLY_SOLD_TREASURE = 4

# History length
NB_PREV_STATES = 3  # 5


# TODO
#  Remove map as given item at start to increase difficulty (when the easy version works).
#  Let the agent start off in a random room


class TreasureMapHardEnv(gym.Env):
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
        self.episode_length = 100  # 75  # 25  # len(self.sequence)  # Nb of actions in one episode
        # Ideally, the agent would only need to take 3 actions to sell a treasure.
        self.repr_length = NB_PREV_STATES

        self.TOGGLE = 0
        self.step_size = 2

        self.nb_BOW_states = 5  # math.floor(self.TOGGLE * self.repr_length)
        self.nb_regular_states = self.repr_length - self.nb_BOW_states

        # When a move doesn't generate an observation -> only interactions generate observations:
        # In this hard version, the agent must also acquire a MAP first
        self.sequence = [MAP, GUIDE, TREASURE, JEWELRY]
        self.sequence2 = [MAP, EQUIPMENT, TREASURE, JEWELRY]

        # In this hard version, the agent starts off in a random room
        self.current_room = choice(ROOMS)

        # Keep a list of the items the agent has collected thus far.
        # In this hard version, the agent doesn't start off with the MAP.
        self.acquired_items = []

        # The observation consists of the current room the agent is in + history rep
        self.observation_space = spaces.Discrete(self.repr_length + 1)  # spaces.Discrete(self.repr_length)
        self.actions = ['left', 'right', 'up', 'down', 'interact']
        self.action_space = spaces.Discrete(len(self.actions))  # move left, right, up, down; interact or reset

        # Use the get_state_repr method that uses a constant array size, or add extra elements to repr according to
        # the variables set underneath?
        self.use_in_place_repr = False

        self.N_STATES = True
        self.BOW = False
        self.MOST_USED = False
        self.INTERVAL = False

        if not self.use_in_place_repr:
            self.construct_repr_length()

    def construct_repr_length(self):
        self.repr_length = 0  # len(ITEMS)
        if self.N_STATES:
            self.repr_length += NB_PREV_STATES
        if self.BOW:
            self.repr_length += len(ITEMS)
        if self.MOST_USED:
            self.repr_length += 1
        if self.INTERVAL:
            self.repr_length += NB_PREV_STATES
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
        if JEWELRY in self.acquired_items:
            self.state = []
            self.acquired_items = []
            # Start off in random room
            self.current_room = choice(ROOMS)
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
                        reward = ACQUIRED_ITEM
                        self.acquired_items.append(TREASURE)
                        self.state.append(TREASURE)
                    else:  # In the hard version, the agent receives, too, for trying to obtain an item more than once
                        reward = DUPLICATE_ITEM
                else:
                    reward = INVALID_ACTION
            elif room == LEFT_ROOM:  # Guide room
                # Agent does NOT start off with map in its self.acquired_items.
                if MAP in self.acquired_items:
                    if GUIDE not in self.acquired_items:
                        if EQUIPMENT not in self.acquired_items:  # has neither GUIDE nor EQUIPMENT
                            reward = ACQUIRED_ITEM
                        else:
                            # already has EQUIPMENT, but not GUIDE
                            # Obtaining GUIDE when already having EQUIPMENT is not efficient
                            reward = NEUTRAL_ACTION
                        self.acquired_items.append(GUIDE)
                        self.state.append(GUIDE)
                    else:  # Already has guide
                        reward = DUPLICATE_ITEM
                else:  # Doesn't yet have MAP
                    reward = INVALID_ACTION
            elif room == CENTER_ROOM:  # Map room, agent does NOT start off with map in its self.acquired_items.
                if MAP not in self.acquired_items:
                    reward = ACQUIRED_ITEM
                    self.acquired_items.append(MAP)
                    self.state.append(MAP)
                else:
                    reward = DUPLICATE_ITEM
            elif room == RIGHT_ROOM:  # Equipment room, agent does NOT start off with map in its self.acquired_items.
                if MAP in self.acquired_items:
                    if EQUIPMENT not in self.acquired_items:
                        if GUIDE not in self.acquired_items:
                            reward = ACQUIRED_ITEM
                        else:
                            # Obtaining EQUIPMENT when already having a GUIDE is inefficient
                            reward = NEUTRAL_ACTION+1
                        self.acquired_items.append(EQUIPMENT)
                        self.state.append(EQUIPMENT)
                    else:
                        reward = DUPLICATE_ITEM
                else:
                    reward = INVALID_ACTION
            elif room == BOTTOM_ROOM:  # Jewelry room
                if bool(GUIDE in self.acquired_items) ^ bool(EQUIPMENT in self.acquired_items):
                    # n.b.: 1. ^ is the XOR operator for booleans
                    #       2. If the agent has GUIDE or EQUIPMENT, this implies he must have MAP too
                    #       3. The agent cannot already have JEWELRY, as it gets reset in the beginning of the
                    #          step method if it already had JEWELRY in its acquired_items.
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
        # elif action == RESET:
        #     reward = RESET_PENALTY
        #     self.reset()

        self.current_room = new_room
        done = self.steps_taken >= self.episode_length  # or JEWELRY in self.state

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

        # Take last `history_length' states; equal to self.state if history_length > len(self.state)
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

        zeros = [0] * (self.repr_length - self.nb_BOW_states - self.nb_regular_states)

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
        return np.concatenate([[self.current_room], states, bow, mu, inter])

    def reset(self):
        self.steps_taken = 0
        self.state = []
        self.acquired_items = []
        self.current_room = choice(ROOMS)
        return self._get_state_repr(self.TOGGLE)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_name(self):
        time = datetime.now().strftime('%H:%M %d-%m-%Y')
        if self.use_in_place_repr:
            return f'EpsLen:{self.episode_length}-' \
                   + f'States:{self.repr_length - self.nb_BOW_states}-' \
                   + f'BoW:{self.nb_BOW_states}-' \
                   + f'StepSize:{self.step_size}-' \
                   + f'{time}'
        else:
            return f'EpsLen:{self.episode_length}-' \
                   + f'N_States:{self.N_STATES}:{self.repr_length}-' \
                   + f'BoW:{self.BOW}-' \
                   + f'Interval:{self.INTERVAL}-' \
                   + f'MU:{self.MOST_USED}-' \
                   + f'{time}'

            
