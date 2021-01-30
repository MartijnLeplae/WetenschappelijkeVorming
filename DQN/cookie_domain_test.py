import gym
import envs

env = gym.make('CookieDomain-v0')
env.reset()
o = env.observation_space
print(o.n)
a = env.step(action=1)
env.reset()
