import gym
from gym import spaces
import numpy as np
import pygame as pg

# colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BACKCOLOR = (91, 54, 13)
WHITE = (255, 255, 255)
PINK = (255, 103, 253)
GREEN = (0, 255, 0)
BLUE_ISH = (105, 103, 253)
BLACK = (0, 0, 0)


class BarryWorld(gym.Env):
    def __init__(self):
        ############ Graphical vars ###############
        self.graphics_initialised = False
        self.screen = None
        self.clock = None
        self.btn_dim = [25,10]
        self.barry_dim = [10, 100]
        self.btn_col = [BLUE, RED, GREEN]
        self.barry_col = PINK
        self.has_pressed = False

        self.dimensions = [500, 500]  # horizontal and vertical dimensions of the world
        self.buttons = np.array([100, 200, 300])  # The x-positions of the buttons in the world
        self.barry = 150  # The starting position of Barry
        self.step_length = 10  # 25 # How big are the steps of Barry when going left or right?

        self.bucket = 0
        self.bucket_pos = 400
        self.bucket_dim = [75, 200]
        #############################################
        self.code = '231'  # The code for a water-droplet
        self.int_code = [int(i) for i in list(self.code)]  # list with int representation of the code

        # Keeps track of the buttons pressed
        self.state = []

        self.actions = ['left', 'right', 'press']

        # Toggle certain history reps on and off
        self.add_states = True  # add normal history of length n_prev_states?
        self.add_most_used = False  # add most used action?
        self.add_counts = False  # add a cumulative sum of used letters?
        self.add_interval = False  # add interval of history of n_prev_states with one state skipped?

        self.n_prev_states = 3  # Nb of previous states to remember
        # self.repr_length = len(self.actions) + self.n_prev_states if self.add_counts else self.n_prev_states
        self.repr_length = 3
        if self.add_states:
            self.repr_length += self.n_prev_states
        if self.add_counts:
            self.repr_length += len(self.actions)
        if self.add_most_used:
            self.repr_length += 1
        if self.add_interval:
            self.repr_length += self.n_prev_states

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = self.repr_length

        self.steps = 0

        self.episode_length = 50

    def step(self, action):
        self.has_pressed = False
        reward = 0
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
                if self.code_complete():
                    # Put some water in the bucket
                    self.fill_bucket()
                reward = self.reward_from_move()
            else:
                # Barry, you cant press a button if you're not near one!
                reward = -1
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
        r = -1
        c = self.int_code
        s = self.state
        if len(s) > len(c):
            s = s[-len(c):]
        elif len(c) > len(s):
            c = c[:len(s)]
        if c == s:
            # The max reward is scaled with how long the current, correct sequence is
            r = max_r * len(c)/len(self.int_code)
        return r

    # # -> This function takes a list of ints as input and returns a string.
    # # -> An int corresponds to an character as follows: int: i -> str: self.actions[i-1]
    # # The ints don't correspond directly to an index because '0' is the empty character
    # # and '0' is not a member of self.actions.
    # def _word(self, lst):
    #     return ''.join([self.actions[i - 1] for i in lst])

    # This is the function where you decide what the agent (=neural network) can 'see'.
    # This can also include information about previous states (=history)
    def _get_state_repr(self):
        def get_count():
            count = []
            for i in range(1, len(self.actions)+1):
                count.append(self.state.count(i))
            return np.array(count)

        def most_used():
            count = get_count()
            return np.array([np.argmax(np.array(count))+1])

        def get_hist_interval():
            interval = -2
            hist = []
            lim = max(-1, len(self.state)-self.n_prev_states*2)
            for i in range(len(self.state)-1, lim, interval):
                hist.append(self.state[i])
            result = np.array([0] * (self.n_prev_states - len(hist)) + hist)
            return result

        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        # 1. construct states vector
        states = np.array([])
        if self.add_states:
            if len(self.state) < self.n_prev_states:
                states = np.array([0] * (self.n_prev_states - len(self.state)) + self.state)
            else:
                states = np.array(self.state[-self.n_prev_states:])
        # 2. Construct count vector
        counts = np.array([])
        if self.add_counts:
            counts = get_count()
        # 3. Construct most-used vector
        mu = np.array([])
        if self.add_most_used:
            mu = np.array(most_used())
        # 4. Construct interval vector
        inter = np.array([])
        if self.add_interval:
            inter = get_hist_interval()
        return np.concatenate([self.get_observation(), states, counts, mu, inter])

    # This methods returns what Barry sees in this state
    def get_observation(self):
        return self.buttons - self.barry

    def fill_bucket(self):
        self.bucket += 10

    def reset(self):
        self.state = []
        self.steps = 0
        self.barry = 150
        self.bucket = 0
        return self._get_state_repr()

    def render(self, mode=None):
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
            textsurface = myfont.render(f"Step: {self.steps}/{self.episode_length}", False, BLACK)
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
        pg.display.update()
        self.clock.tick(15)


    def get_name(self):
        name = ''
        if self.add_states:
            name += f'States:{self.n_prev_states}'
        if self.add_interval:
            name += f'Interval:{self.n_prev_states}'
        if self.add_counts:
            name += '+BoW'
        if self.add_most_used:
            name += '+mu'
        name += '-' + self.code
        return name