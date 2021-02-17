import numpy as np
import gym
import envs
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from matplotlib import pyplot as plt


class Trainer:
    def __init__(self):
        self.N_EPISODES = 300
        self.STEPS_PER_EPISODE = 2
        self.BATCH_SIZE = 32

        # self.ENV = 'CookieDomain-v0'
        # self.ENV = 'CartPole-v0'
        self.ENV = 'WordsWorld-v0'

        self.env = gym.make(self.ENV)

        self.env.episode_length = self.STEPS_PER_EPISODE

        np.random.seed(123)
        self.env.seed(123)

        self.nb_actions = self.env.action_space.n
        self.state_size = (self.env.observation_space.n,)  # self.env.observation_space.shape  # (self.env.observation_space.n,)

        self.warmup_episodes = self.BATCH_SIZE // self.env.episode_length + 1

        self.dqn = None

        self.episode_reward = None  # A variable containing a list with reward/episode after training

        self.window_length = 1  # Not really sure what this is, but I think it is a way to have combine
        # past n past observations in one state. (A way to make simple memory, it is also used in the examples of the atari games
        # where the window_length is 4 to account for acceleration etc)

    def _build_model(self):
        model = Sequential()
        # Input Layer
        model.add(Flatten(input_shape=self.state_size))
        # Layer 1
        model.add(Dense(10))
        model.add(Activation('relu'))
        # Layer 2
        model.add(Dense(10))
        model.add(Activation('relu'))
        # # Layer 3
        # model.add(Dense(20))
        # model.add(Activation('relu'))
        # Output Layer
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model

    def start(self):
        model = self._build_model()
        memory = SequentialMemory(limit=5000, window_length=self.window_length)
        policy = EpsGreedyQPolicy(eps=0.1)
        # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05,
        #                               nb_steps=self.env.episode_length*(self.N_EPISODES+self.warmup_episodes)*0.5)
        # test_policy = EpsGreedyQPolicy(eps=0.05)
        self.dqn = DQNAgent(model=model, batch_size=self.BATCH_SIZE, enable_double_dqn=True, nb_actions=self.nb_actions,
                       memory=memory, nb_steps_warmup=self.warmup_episodes*self.env.episode_length,
                       target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        out = self.dqn.fit(self.env, nb_steps=(self.N_EPISODES + self.warmup_episodes)*self.env.episode_length,
                      visualize=False, verbose=1)
        self.episode_reward = out.history['episode_reward'][self.warmup_episodes:]
        return self.episode_reward

    def test(self):
        self.dqn.test(self.env, nb_episodes=10, visualize=True)

    # Plot the reward/episode
    def plot(self, save=False):
        if self.episode_reward is None:
            print("You should call plot after training. No data present for reward/episode")
            return
        plt.figure(1)
        plt.title(f'Average Rewards in {self.ENV} on {len(self.episode_reward)} episodes')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.plot(self.episode_reward)
        plt.show()


trainer = Trainer()
trainer.start()
trainer.test()
trainer.plot()


