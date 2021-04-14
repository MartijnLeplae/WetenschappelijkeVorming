import gym
from gym import spaces
import numpy as np
import re

# COLORS
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BACKCOLOR = (91, 54, 13)
WHITE = (255, 255, 255)
PINK = (255, 103, 253)
GREEN = (0, 255, 0)
BLUE_ISH = (105, 103, 253)
BLACK = (0, 0, 0)
# LOCATION VARIABLES (x-position of the elements)
BARRY_START = 0
BUTTONS = [100, 200, 300]
# OTHER VARS
STEP_SIZE = 50 # 25 # 10 # How big are the steps of Barry?
# DIMENSION VARIABLES
WORLD_DIM = [500,500]
BARRY_DIM = [10, 75]
BUTTON_DIM = [25, 10]
# COLOR VARIABLES
WORLD_COLOR = WHITE
BARRY_COLOR = PINK
BUTTONS_COLOR = [BLUE, RED, GREEN]
################
CODE = '123' # The code barry has to learn
# REWARD VARIABLES
BASE = 0
UNVALID_ACTION = -1
WRONG_BUTTON = -2
GOOD_BUTTON = 1
CODE_COMPLETE = 6
# HISTORY REPRESENTATION
POLICY = 'epsgr:' # this is just appended to the name_string, does not have effect otherwise
NB_PREV_STATES = 3
N_STATES = True  # add normal history of length n_prev_states?
MOST_USED = False  # add most used action?
BOW = False  # add a Bag-off-words?
INTERVAL = False  # add self.interval of history of n_prev_states with one state skipped?
EPISODE_LENGTH = 25


"""
BarryWorld is a environment where Barry wants to fill a bucket of water.
To do this he has to press buttons in a certain order. Barry can move around and can 
press a button if he is close enough to it. 
He can see this vector every step:
    -> [delta_bucket1, delta_bucket2, (delta_bucket3), 'extra information about last pressed buttons']
    Where delta denotes the distance with Barry.
    The extra information can be toggled in the constructor. For example the environment can give Barry
    a vector with the last n pressed buttons.
"""
class BarryWorld(gym.Env):
    def __init__(self):
        ############ Graphical vars ###############
        self.graphics_initialised = False
        self.screen = None
        self.clock = None
        self.btn_dim = BUTTON_DIM
        self.barry_dim = BARRY_DIM
        self.btn_col = BUTTONS_COLOR
        self.barry_col = PINK
        self.has_pressed = False
        self.dimensions = WORLD_DIM
        self.buttons = np.array(BUTTONS)
        self.barry = BARRY_START
        self.step_length = STEP_SIZE
        #############################################
        self.code = CODE
        self.int_code = [int(i) for i in list(re.sub("[^0-9]", "", self.code))]  # list with int representation of the code

        # Keeps track of the buttons pressed
        self.state = []
        self.reward_state = rewardState(self.code)

        self.score = 0  # keeps track of full code completions

        self.actions = ['left', 'right', 'press']

        self.nb_prev_states = NB_PREV_STATES
        self.n_states = N_STATES
        self.most_used = MOST_USED
        self.bow = BOW
        self.interval = INTERVAL

        self.repr_length = len(BUTTONS)
        self.construct_repr_length()

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = self.repr_length

        self.steps = 0

        self.episode_length = EPISODE_LENGTH

    def construct_repr_length(self):
        if self.n_states:
            self.repr_length += self.nb_prev_states
        if self.bow:
            self.repr_length += len(BUTTONS)
        if self.most_used:
            self.repr_length += 1
        if self.interval:
            self.repr_length += self.nb_prev_states

    def set_user_parameters(self, **params: dict):
        """
        This method sets the parameters to values
        given by the user. The parameters that are set
        are those used by testSuite.py.
        """

        assert params, "params variable can't be None"
        for p, val in params.items():
            setattr(self, p, val)
        self.construct_repr_length()

    def step(self, action):
        self.has_pressed = False
        if len(self.state) >= len(self.code):
            self.state = []
            self.reward_state.reset()
        reward = BASE
        # A. Barry moves around
        if action != 2:
            direction = -1 if action == 0 else 1
            newbarry = self.barry + (direction * self.step_length)
            if self.dimensions[0] > newbarry > 0:
                self.barry = newbarry
        observation = self.get_observation()
        # B. Barry wants to press a button
        if action == 2:
            dx = abs(observation)
            closest = min(dx)
            # Barry can press buttons as close as 5-length away
            if closest <= 5:
                self.has_pressed = True
                self.state.append(dx.argmin()+1)
                reward, code_complete, flush = self.reward_state.step(self.state[-1])
                if flush:
                    self.state = []
                if code_complete:
                    self.reward_state.reset()
                    self.score += 1
            else:
                # Barry, you cant press a button if you're not near one!
                reward = UNVALID_ACTION
        self.steps += 1
        done = False if self.steps < self.episode_length else True
        return self._get_state_repr(), reward, done, {}

    # def code_complete(self):
    #     if len(self.state) < len(self.code):
    #         return False
    #     last_presses = self.state[-len(self.code):]
    #     return last_presses == self.int_code
    #
    # # Return the reward Barry gets assuming the last pressed button was appended to self.state
    # def reward_from_move(self):
    #     max_r = 6
    #     r = BASE
    #     c = self.int_code
    #     s = self.state
    #     if len(s) > len(c):
    #         s = s[-len(c):]
    #     elif len(c) > len(s):
    #         c = c[:len(s)]
    #     if c == s:
    #         r = GOOD_BUTTON
    #     else:
    #         self.state = []
    #         r = WRONG_BUTTON
    #     return r

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    def _get_state_repr(self):
        def get_bow():
            count = []
            for i in range(1, len(BUTTONS)+1):
                count.append(self.state.count(i))
            return np.array(count)

        def most_used():
            count = get_bow()
            return np.array([np.argmax(np.array(count))+1])

        def get_hist_interval():
            interval = -2
            hist = []
            lim = max(-1, len(self.state)-self.nb_prev_states*2)
            for i in range(len(self.state)-1, lim, interval):
                hist.append(self.state[i])
            result = np.array([0] * (self.nb_prev_states - len(hist)) + hist)
            return result

        def get_states():
            if len(self.state) < self.nb_prev_states:
                return np.array([0] * (self.nb_prev_states - len(self.state)) + self.state)
            else:
                return np.array(self.state[-self.nb_prev_states:])

        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        # 1. construct states vector
        states = np.array([])
        if self.n_states:
            states = get_states()
        # 2. Construct count vector
        bow = np.array([])
        if self.bow:
            bow = get_bow()
        # 3. Construct most-used vector
        mu = np.array([])
        if self.most_used:
            mu = np.array(self.most_used())
        # 4. Construct self.interval vector
        inter = np.array([])
        if self.interval:
            inter = get_hist_interval()
        return np.concatenate([self.get_observation(), states, bow, mu, inter])

    # This methods returns what Barry sees in this state
    def get_observation(self):
        return self.buttons - self.barry

    def reset(self):
        self.state = []
        self.steps = 0
        self.score = 0
        self.barry = BARRY_START
        self.reward_state.reset()
        return self._get_state_repr()

    def render(self, mode=None):
        import pygame as pg
        def draw_buttons():
            for i, btn in enumerate(self.buttons):
                width = self.btn_dim[0]
                height = self.btn_dim[1]
                xpos = btn - width/2
                ypos = self.dimensions[1] - height
                col = self.btn_col[i]
                pg.draw.rect(self.screen, col, (xpos, ypos, width, height))

        def draw_barry():
            height = self.barry_dim[1]
            width = self.barry_dim[0]
            ypos = self.dimensions[1] - height
            border = 0
            if not self.has_pressed:
                border = 5
            pg.draw.rect(self.screen, self.barry_col, (self.barry, ypos, width, height), border)

        def draw_info():
            myfont = pg.font.SysFont("Comic Sans MS", 25)
            textsurface = myfont.render(f"Step: {self.steps}/{self.episode_length} Score:{self.score}", True, BLACK)
            location = [10, 10]
            self.screen.blit(textsurface, location)

        def draw_state():
            myfont = pg.font.SysFont("Comic Sans MS", 25)
            textsurface = myfont.render(f"State: {self.state[-len(self.code):]}", True, BLACK)
            location = [170, 10]
            self.screen.blit(textsurface, location)

        def draw_code():
            myfont = pg.font.SysFont("Comic Sans MS", 25)
            textsurface = myfont.render(f"Code: {self.code}", True, BLACK)
            location = [350, 10]
            self.screen.blit(textsurface, location)

        if not self.graphics_initialised:
            pg.init()
            self.screen = pg.display.set_mode((self.dimensions[0], self.dimensions[1]))
            pg.display.set_caption('Barry-World')
            self.clock = pg.time.Clock()
            self.graphics_initialised = True
        self.screen.fill(WHITE)
        draw_buttons()
        draw_barry()
        draw_info()
        draw_state()
        draw_code()
        pg.display.update()
        self.clock.tick(7) # How many FPS?


    def get_name(self):
        name = POLICY + str(self.episode_length)
        if self.n_states:
            name += f'States:{self.nb_prev_states}'
        if self.interval:
            name += f'interval:{self.nb_prev_states}'
        if self.bow:
            name += 'bow'
        if self.most_used:
            name += 'mu'
        name += '-' + self.code
        return name


class rewardState:
    """
    Class to emulate a finite state machine for rewards.
    Only supports simple regexes (only 'symbols', * and | but no simicolons)

    examples:
        type '123' if you mean '123'
        type '12*3' if you mean 1(2)*3
        type 12|13 if you mean 1(2|1)3
        type 1x3 if you mean 1(wildcard)3
    assumptions:
        - correct use of syntax: e.g. not '*', '12|', '2*', ...
    """
    def __init__(self, regex):
        self.pos = 0  # the current index in the regex (so this means: what input am i expecting?)
        self.wildcard = 'x'
        self.regex = regex

    def step(self, input):
        """
        Make a step in the state-machine.

        :param int input: The symbol to use as input
        :return: Reward from the given step.
        """
        input = str(input)
        size = len(self.regex)
        flush = False
        if self.pos + 1 >= size:
            if input == self.regex[self.pos]:
                return CODE_COMPLETE, True, True
            else:
                return WRONG_BUTTON, False, True
        else:
            goodmove = False
            if input == self.regex[self.pos]:
                goodmove = True
            if self.regex[self.pos+1] == '|':
                goodmove = goodmove or (input == self.regex[self.pos+2])
                self.pos += 3
            elif self.regex[self.pos+1] == '*':
                if not goodmove:
                    if input == self.regex[self.pos+2]:
                        goodmove = True
                        self.pos += 3
            elif goodmove:
                self.pos += 1
            if goodmove:
                reward = GOOD_BUTTON
            else:
                reward = WRONG_BUTTON
                self.reset()
                flush = True
            complete = goodmove and (self.pos > size)
            return reward, complete, flush

    def reset(self):
        self.pos = 0




