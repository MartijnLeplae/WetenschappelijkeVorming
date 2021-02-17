import gym
import envs

env = gym.make('OTTWorld-v0')
env.reset()
#o = env.observation_space
#print(o.n)
a = env.step()
env.reset()
