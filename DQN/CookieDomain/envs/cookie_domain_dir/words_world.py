import gym
from gym import spaces
import numpy as np

# -> In words world the task is to 'type' a specific word with a select amount of keys (=actions)
# One observation consists of the last typed letter
# -> e.g.: word="helloworld", the agent can select the following actions ['d', 'e', 'h', 'l', 'o', 'r', 'w']
# the internal state of the environment is the current typed word e.g.'dorw'
# the agent 'sees' w or in this case 7 cause we represent letters with their index
# -> Words whithout repeating letters 'should' (knock on wood) be easyer to learn than words with repeated
# letters cause one character is always followed by the same charachter. e.g.: 'blue'

class WordsWorld(gym.Env):
    def __init__(self):
        self.goal = 'red' # the desired word to learn

        self.state = [0]*len(self.goal) # The initial state is the empty character

        self.actions = list(set(self.goal))
        self.actions.sort()
        self.nb_goal = [self.actions.index(l)+1 for l in self.goal]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(1)

        self.steps = 0

        self.episode_length = 10

    def step(self, letter):
        self.state.append(letter+1)
        if len(self.state) > len(self.goal):
            self.state = [0]*(len(self.goal)-1) + [letter+1] # self.state[-len(self.goal):]
        self.steps += 1
        done = False if self.steps < self.episode_length else True
        # Was it the correct letter?
        if self.state == self.nb_goal[:len(self.state)]:
            reward = 5
            if len(self.state) == len(self.nb_goal):
                reward = 10
            return np.array(self.state[-1]), reward, True, {}

        # Correct letter but not in the big picture e.g: goal = 'ababc' and the agent types 'abab->a'
        # then yes 'b' is followed by 'a' somewhere but there 'c' was needed
        if self._word(self.state[-2:]) in self._word(self.nb_goal) and len(self.state)>1:
            reward = 5
        else:
            # Bad choise, wrong letter
            reward = -5
        return np.array(self.state[-1]), reward, done, {}

    def _word(self, lst):
        return ''.join([self.actions[i-1] for i in lst])


    def reset(self):
        self.state = [0]
        self.steps = 0
        return np.array(0)

    def render(self, mode=None):
        state = self._word(self.state)
        print(state)
