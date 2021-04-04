import gym
from gym import spaces
import numpy as np


# -> In words world the task is to 'type' a specific word with a select amount of keys (=actions)
# One observation consists of the last typed letter
# -> e.g.: word="helloworld", the agent can select the following actions ['d', 'e', 'h', 'l', 'o', 'r', 'w']
# the internal state of the environment is the current typed word e.g.'dorw'
# the agent 'sees' w or in this case 7 cause we represent letters with their index+1
# -> Words without repeating letters 'should' (knock on wood) be easier to learn than words with repeated
# letters cause one character is always followed by the same character. e.g.: 'blue'

class WordsWorld(gym.Env):
    def __init__(self):
        self.goal = 'abbaabbaba' #'hottentottententententoonstelling'  #'helloworld'  #  # the desired word to learn

        self.state = []

        self.actions = list(set(self.goal))
        self.actions.sort()
        self.nb_goal = [self.actions.index(l) + 1 for l in self.goal]

        # Toggle certain history reps on and off
        self.add_states = False # add normal history of length n_prev_states?
        self.add_most_used = False # add most used action?
        self.add_counts = False # add a cumulative sum of used letters?
        self.add_interval = True # add interval of history of n_prev_states with one state skipped?

        self.n_prev_states = 4 # Nb of previous states to remember
        # self.repr_length = len(self.actions) + self.n_prev_states if self.add_counts else self.n_prev_states
        self.repr_length = 0
        self.construct_repr_length()
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.repr_length)

        self.steps = 0

        self.episode_length = len(self.goal)

    def construct_repr_length(self):
        if self.add_states:
            self.repr_length += self.n_prev_states
        if self.add_counts:
            self.repr_length += len(self.actions)
        if self.add_most_used:
            self.repr_length += 1
        if self.add_interval:
            self.repr_length += self.n_prev_states

    def set_user_parameters(self, params: dict):
        """
        This method sets the parameters to values
        given by the user. The parameters that are set
        are those used by testSuite.py.
        """

        assert params, "params variable can't be None"
        for p, val in params.items():
            setattr(self, p, val)
        self.construct_repr_length()

    def step(self, letter):
        self.state.append(letter + 1)
        if len(self.state) > len(self.goal):
            self.state = self.state[-len(self.goal):]
        self.steps += 1
        done = False if self.steps < self.episode_length else True
        # Was it the correct letter?
        if self.state == self.nb_goal[:len(self.state)]:
            reward = 2
            if len(self.state) == len(self.nb_goal):
                reward = 5
            return self._get_state_repr(), reward, done, {}

        # Correct letter but not in the big picture e.g: goal = 'ababc' and the agent types 'abab->a'
        # then yes 'b' is followed by 'a' somewhere but there 'c' was needed
        if len(self.state) > self.n_prev_states \
                and self._word(self.state[-(self.n_prev_states + 1):]) in self._word(self.nb_goal):
            reward = 1
        else:
            # Bad choice, wrong letter
            reward = -1
        return self._get_state_repr(), reward, done, {}

    # -> This function takes a list of ints as input and returns a string.
    # -> An int corresponds to an character as follows: int: i -> str: self.actions[i-1]
    # The ints don't correspond directly to an index because '0' is the empty character
    # and '0' is not a member of self.actions.
    def _word(self, lst):
        return ''.join([self.actions[i - 1] for i in lst])

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
        return np.concatenate([states, counts, mu, inter])


    def reset(self):
        self.state = []
        self.steps = 0
        return self._get_state_repr()

    def render(self, mode=None):
        state = self._word(self.state)
        print(state[:-1] + '\u001b[1m' + state[-1] + '\u001b[0m')

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
        name += '-' + self.goal
        return name