import gym
from gym import spaces
import numpy as np

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
BUCKET_POS = 400
# OTHER VARS
STEP_SIZE = 50 # 25 # 10 # How big are the steps of Barry?
# DIMENSION VARIABLES
WORLD_DIM = [500,500]
BARRY_DIM = [10, 75]
BUTTON_DIM = [25, 10]
BUCKET_DIM = [75, 200]
# COLOR VARIABLES
WORLD_COLOR = WHITE
BARRY_COLOR = PINK
BUTTONS_COLOR = [BLUE, RED, GREEN]
################
CODE = '1231232' # The code barry has to learn
# REWARD VARIABLES
BASE = 0
UNVALID_ACTION = -1
WRONG_BUTTON = -2
GOOD_BUTTON = 1
CODE_COMPLETE = 6
# HISTORY REPRESENTATION
POLICY = 'linan5-1:' # this is just appended to the name_string, does not have effect otherwise
NB_PREV_STATES = 4
N_STATES = False  # add normal history of length n_prev_states?
MOST_USED = False  # add most used action?
BOW = True  # add a Bag-off-words?
INTERVAL = False  # add interval of history of n_prev_states with one state skipped?
EPISODE_LENGTH = 75


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
        self.bucket = 0
        self.bucket_pos = BUCKET_POS
        self.bucket_dim = BUCKET_DIM
        #############################################
        self.code = CODE
        self.int_code = [int(i) for i in list(self.code)]  # list with int representation of the code

        # Keeps track of the buttons pressed
        self.state = []

        self.actions = ['left', 'right', 'press']

        repr_length = len(BUTTONS)
        if N_STATES:
            repr_length += NB_PREV_STATES
        if BOW:
            repr_length += len(BUTTONS)
        if MOST_USED:
            repr_length += 1
        if INTERVAL:
            repr_length += NB_PREV_STATES

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = repr_length

        self.steps = 0

        self.episode_length = EPISODE_LENGTH

    def step(self, action):
        self.has_pressed = False
        if len(self.state) >= len(self.code):
            self.state = []
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
                reward = self.reward_from_move()
                if self.code_complete():
                    reward = CODE_COMPLETE
                    self.state = [] # Start over
                    # Put some water in the bucket
                    self.fill_bucket()
            else:
                # Barry, you cant press a button if you're not near one!
                reward = UNVALID_ACTION
        self.steps += 1
        done = False if self.steps < self.episode_length else True
        return self._get_state_repr(), reward, done, {}

    def code_complete(self):
        if len(self.state) < len(self.code):
            return False
        last_presses = self.state[-len(self.code):]
        return last_presses == self.int_code

    # Return the reward Barry gets assuming the last pressed button was appended to self.state
    def reward_from_move(self):
        max_r = 6
        r = BASE
        c = self.int_code
        s = self.state
        if len(s) > len(c):
            s = s[-len(c):]
        elif len(c) > len(s):
            c = c[:len(s)]
        if c == s:
            r = GOOD_BUTTON #max_r * len(c)/len(self.int_code)
        else:
            self.state = []
            r = WRONG_BUTTON
        return r

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
            lim = max(-1, len(self.state)-NB_PREV_STATES*2)
            for i in range(len(self.state)-1, lim, interval):
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
        if N_STATES:
            states = get_states()
        # 2. Construct count vector
        bow = np.array([])
        if BOW:
            bow = get_bow()
        # 3. Construct most-used vector
        mu = np.array([])
        if MOST_USED:
            mu = np.array(most_used())
        # 4. Construct interval vector
        inter = np.array([])
        if INTERVAL:
            inter = get_hist_interval()
        return np.concatenate([self.get_observation(), states, bow, mu, inter])

    # This methods returns what Barry sees in this state
    def get_observation(self):
        return self.buttons - self.barry

    def fill_bucket(self):
        self.bucket += 30

    def reset(self):
        self.state = []
        self.steps = 0
        self.barry = BARRY_START
        self.bucket = 0
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
            textsurface = myfont.render(f"Step: {self.steps}/{self.episode_length}", True, BLACK)
            location = [10, 10]
            self.screen.blit(textsurface, location)

        def draw_bucket():
            height = self.bucket_dim[1]
            width = self.bucket_dim[0]
            ypos = self.dimensions[1] - height
            xpos = self.bucket_pos
            filled = self.bucket
            b_ypos = self.dimensions[1] - filled
            pg.draw.rect(self.screen, BLACK, (xpos, ypos, width, height), 5)
            self.screen.fill(BLUE, (xpos, b_ypos, width, filled))

        def draw_state():
            myfont = pg.font.SysFont("Comic Sans MS", 25)
            textsurface = myfont.render(f"State: {self.state[-len(self.code):]}", True, BLACK)
            location = [150, 10]
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
        draw_bucket()
        draw_info()
        draw_state()
        draw_code()
        pg.display.update()
        self.clock.tick(7) # How many FPS?


    def get_name(self):
        name = POLICY + str(self.episode_length)
        if N_STATES:
            name += f'States:{NB_PREV_STATES}'
        if INTERVAL:
            name += f'Interval:{NB_PREV_STATES}'
        if BOW:
            name += '+BoW'
        if MOST_USED:
            name += '+mu'
        name += '-' + self.code
        return name
