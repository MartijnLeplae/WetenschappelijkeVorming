import numpy as np
import gym
import envs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


class Trainer:
    def __init__(self):
        self.N_EPISODES = 1000
        self.STEPS_PER_EPISODE = 200
        self.BATCH_SIZE = 32

        self.ENV = 'CookieDomain-v0'
        # self.ENV = 'CartPole-v0'

        self.env = gym.make(self.ENV)

        self.env.episode_length = self.STEPS_PER_EPISODE

        np.random.seed(123)
        self.env.seed(123)

        self.nb_actions = self.env.action_space.n
        self.state_size = (self.env.state_vector_size,)

        self.warmup_episodes = self.BATCH_SIZE // self.env.episode_length + 1

        self.dqn = None

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.state_size))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model

    def start(self):
        model = self._build_model()
        memory = SequentialMemory(limit=5000, window_length=1)
        policy = EpsGreedyQPolicy(eps=0.1)
        self.dqn = DQNAgent(model=model, batch_size=self.BATCH_SIZE, enable_double_dqn=True, nb_actions=self.nb_actions,
                       memory=memory, nb_steps_warmup=self.warmup_episodes*self.env.episode_length,
                       target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        out = self.dqn.fit(self.env, nb_steps=(self.N_EPISODES + self.warmup_episodes)*self.env.episode_length,
                      visualize=False, verbose=1)
        return out.history['episode_reward'][self.warmup_episodes:]

    def test(self):
        self.dqn.test(self.env, nb_episodes=10, visualize=True)


trainer = Trainer()
trainer.start()
trainer.test()