import numpy as np
import gym
import envs
import os
# import gym_two_rooms.envs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse


class Trainer:
    def __init__(self, env=None):
        self.N_EPISODES = 2000
        # self.STEPS_PER_EPISODE = 3
        self.BATCH_SIZE = 32

        # Pass an environment to DQN_keras_rl.py with the '-e' or '--environment' flag
        if env:
            self.ENV = env
        else:
            # self.ENV = 'CookieDomain-v0'
            # self.ENV = 'CartPole-v0'
            # self.ENV = 'WordsWorld-v0'
            # self.ENV = 'TwoRooms-v0'
            self.ENV = 'BarryWorld-v0'

        self.env = gym.make(self.ENV)

        # self.env.episode_length = self.STEPS_PER_EPISODE

        np.random.seed(123)
        self.env.seed(123)

        self.nb_actions = self.env.action_space.n
        self.state_size = (self.env.observation_space,)  # (self.env.observation_space.n,)  #
        # self.env.observation_space.shape

        self.dqn = None

        self.warmup_episodes = self.BATCH_SIZE // self.env.episode_length + 1

        self.window_length = 1  # Not really sure what this is, but I think it is a way to have combine
        # past n past observations in one state. (A way to make simple memory, it is also used in the examples of the
        # atari games where the window_length is 4 to account for acceleration etc)

        self.episode_reward = None  # A variable containing a list with reward/episode after training

        self.name = f"({self.N_EPISODES}){self.env.get_name()}"  # ask the environment for a string representation of
        # the parameters, also append the number of episodes to the front

        self.total_nb_steps = (self.N_EPISODES + self.warmup_episodes) * self.env.episode_length

    def init_agent(self):
        model = self._build_model()
        memory = SequentialMemory(limit=int(self.total_nb_steps*0.7), window_length=self.window_length)
        policy = EpsGreedyQPolicy(eps=0.2)  # <- When not a lot of exploration is needed (better choice for our envs)
        # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.2, value_test=0.1, nb_steps=self.total_nb_steps)
        test_policy = EpsGreedyQPolicy(eps=0.1)  # do some random actions even when testing
        self.dqn = DQNAgent(model=model, batch_size=self.BATCH_SIZE, enable_double_dqn=True, nb_actions=self.nb_actions,
                            memory=memory, nb_steps_warmup=self.warmup_episodes * self.env.episode_length,
                            target_model_update=1e-2, policy=policy, test_policy=test_policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def _build_model(self):
        model = Sequential()
        # Input Layer
        model.add(Flatten(input_shape=(1,) + self.state_size))
        # Layer 1
        model.add(Dense(50))
        model.add(Activation('relu'))
        # Layer 2
        model.add(Dense(50))
        model.add(Activation('relu'))
        # Layer 3
        model.add(Dense(50))
        model.add(Activation('relu'))
        # Output Layer
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model

    def start(self, save=False):
        self.init_agent()
        out = self.dqn.fit(self.env, nb_steps=self.total_nb_steps, visualize=False, verbose=1)
        self.episode_reward = out.history['episode_reward'][self.warmup_episodes:]
        if save:
            self.save_model()
        return self.episode_reward

    def test(self):
        self.dqn.test(self.env, nb_episodes=1, visualize=True)

    def save_model(self):
        dir = f'models/{self.ENV}/'
        path = f'{dir}{self.name}.h5'
        self.dqn.save_weights(path, overwrite=True)

    def load_model(self, path):
        self.init_agent()
        self.dqn.load_weights(path)

    # Plot the reward/episode
    def plot(self, save=False):
        if self.episode_reward is None:
            print("You should call plot after training. No data present for reward/episode")
            return
        plt.figure(1)
        plt.title(f'Average Rewards in {self.ENV} on {len(self.episode_reward)} episodes')
        plt.xlabel('episode')
        plt.ylabel('reward')
        step = 5
        plt.plot(np.arange(0, len(self.episode_reward), step),
                 [self.episode_reward[i] for i in range(0, len(self.episode_reward), step)])
        if save:
            dir = f'graphs/{self.ENV}/'
            plt.savefig(f'{dir}{self.name}')
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['train', 'test'], default='train')
    parser.add_argument('-e', '--environment', type=str)
    parser.add_argument('-w', '--weights', type=str, default=None)
    args = parser.parse_args()

    if args.environment:
        trainer = Trainer(args.environment)
    else:
        trainer = Trainer()
    if args.mode == "train":
        trainer.start(save=True)
        trainer.plot(save=True)
    elif args.mode == "test":
        filepath = "(500)States:3-231.h5"
        if args.weights:
            filepath = args.weights
        trainer.load_model(f'models/{trainer.ENV}/{filepath}')
        trainer.test()

