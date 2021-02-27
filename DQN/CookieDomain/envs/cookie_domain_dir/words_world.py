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
        self.goal = 'helloworld'  # 'hottentottententententoonstelling' # the desired word to learn

        self.state = []

        self.actions = list(set(self.goal))
        self.actions.sort()
        self.nb_goal = [self.actions.index(l) + 1 for l in self.goal]

        self.repr_length = 3

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.repr_length)

        self.steps = 0

        self.episode_length = len(self.goal)

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
        if len(self.state) > self.repr_length \
                and self._word(self.state[-(self.repr_length + 1):]) in self._word(self.nb_goal):
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
        # Current implementation returns a np array of size self.repr_length
        # if the current state is smaller than the repr_length it is
        # filled with extra '0s' (=empty character)
        if len(self.state) < self.repr_length:
            return np.array([0] * (self.repr_length - len(self.state)) + self.state)
        else:
            return np.array(self.state[-self.repr_length:])

    def reset(self):
        self.state = []
        self.steps = 0
        return self._get_state_repr()

    def render(self, mode=None):
        state = self._word(self.state)
        print(state[:-1] + '\u001b[1m' + state[-1] + '\u001b[0m')
