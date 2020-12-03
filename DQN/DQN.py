import gym
import random
import numpy as np
import os

from collections import deque
from tensorflow import keras
from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

environment = 'CartPole-v0'
env = gym.make(environment)


state_size = env.observation_space.shape[0]
acion_size = env.action_space.n

batch_size = 32
n_episodes = 350


output_dir = f'model_output/{environment}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Replay memory and size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95

        ### DOUBLE Q-LEARNING PARAMETERS ###
        self.target_update_method = 0  # 0 for Polyak update; 1 for updating after 'update_freq' memory replays
        self.tau = 0.01  # Parameter of double-q learning: determines the speed of averaging between model and target model
        # Parameters for other double-q learning method
        self.model_t_update_counter = 0
        self.update_freq = 1

        # Percentage of actions to be random
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        # Parameters for model evaluation
        self.epsilon_greedy = 0.05  # Use sometimes random actions in stead of the model output
        self.do_evaluate = True
        self.show_progressbar = False

        self.learning_rate = 0.01

        self.timeout = 250  # Maximum steps in one episode (CartPole is limited to 200 by itself)

        # Build model and target model
        self.model = self._build_model()
        self.model_t = self._build_model()
        self.model_t.set_weights(self.model.get_weights())

        self.name = f'g:{self.gamma}, lr:{self.learning_rate}, dc:{self.epsilon_decay}, dq:{self.target_update_method}'

    # Design of the deep-q neural network
    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # Appends an experience to the replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    # Choose an action based on the current state or a random action with probability epsilon
    def act(self, state, test=False):
        epsilon = self.epsilon
        if test:
            epsilon = self.epsilon_greedy
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        act_value = self.model.predict(state)
        return np.argmax(act_value[0])

    # Trains the model on batch_size experiences picked from self.memory at random
    def replay(self, batch_size):
        # Training is possible if memory has enough experiences
        if len(self.memory) < batch_size:
            return
        # Pick random minibatch from replay memory
        minibatch = random.sample(self.memory, batch_size)
        # Train the model on each element in the minibatch
        for state, action, reward, next_state, done in minibatch:
            # Determine the target reward
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model_t.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            # One backprop step
            self.model.fit(state, target_f, batch_size=1, shuffle=False, verbose=0)

        # Update the target model (double-q learning)
        self.update_target(method=self.target_update_method)

        # Decay the epsilon parameter
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Method to update the target model
    def update_target(self, method=0):
        if method == 0:
            # Methode A: Polyak Averaging
            new_target_weights = []
            for target_weights, weights in zip(self.model_t.get_weights(), self.model.get_weights()):
                new_target_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
            self.model_t.set_weights(np.array(new_target_weights))
        else:
            # Methode B: Updating target weights every self.update_freq
            if agent.model_t_update_counter >= agent.update_freq:
                agent.model_t.set_weights(agent.model.get_weights())
                agent.model_t_update_counter = 0
            agent.model_t_update_counter += 1

    # Method to calculate the average reward for the current model over n trials
    def evaluate(self, n=30):
        avr = []
        for _ in tqdm(range(n), disable=not self.show_progressbar):
            s = env.reset()
            s = np.reshape(s, [1, state_size])
            current_reward = []
            for j in range(self.timeout):
                # env.render()
                a = agent.act(s, test=True)
                ns, r, d, _ = env.step(a)
                r = r if not d else -10
                current_reward.append(r)
                ns = np.reshape(ns, [1, state_size])
                s = ns
                if d:
                    avr.append(sum(current_reward))
                    break
        return sum(avr) / len(avr)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)


def main():
    print(agent.name)
    rewards = []
    for e in range(n_episodes):
        current_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # Play episode (until done or timeout)
        for time in range(agent.timeout):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            current_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # What to do after each episode: evaluate agent if needed and print progress
            if done:
                if agent.do_evaluate:  # or e % 100 == 0: uncomment to evaluate agent every 100 episodes
                    avv_reward = agent.evaluate()
                    rewards.append(avv_reward)
                    print(f"episode {e}/{n_episodes}, score: {time}, average reward: {avv_reward}, e: {agent.epsilon:.2}")
                else:
                    print(f"episode {e}/{n_episodes}, score: {time}, current_reward: {current_reward}, e: {agent.epsilon:.2}")
                break

        # Do memory replay
        agent.replay(batch_size)

        # Save the model weights every 50 episodes
        if e % 50 == 0:
            if not os.path.exists(output_dir + agent.name):
                os.makedirs(output_dir + agent.name)
            agent.save(output_dir + '/' + agent.name + f"/weights_{e:04d}.hdf5")

    # Plot and save average reward per episode if measured
    if agent.do_evaluate:
        plt.figure(1)
        # Set title and window title
        fig = plt.gcf()
        fig.canvas.set_window_title(agent.name)
        plt.title(f'Average Rewards in {environment} on {n_episodes} episodes')
        # Plot rewards
        plt.plot(rewards)
        dir_name = 'graphs/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name + agent.name + '.png')
        plt.show()  # (uncomment to display graph after training, only works when having a screen (not via ssh))


# Method to test an agent for given model (name = path to .hdf5 file)
def test_agent(name):
    agent.load(name)
    print('max achievable reward: 199')
    n = 10
    for j in range(n):
        s = env.reset()
        s = np.reshape(s, [1, state_size])
        for i in range(agent.timeout):
            env.render()
            a = agent.act(s, test=True)
            ns, r, d, _ = env.step(a)
            if d:
                print(f'Score:{i}')
                break
            ns = np.reshape(ns, [1, state_size])
            s = ns


agent = DQNAgent(state_size, acion_size)
if __name__ == '__main__':
    #main() #uncomment to train a model
    test_agent('model_output/CartPole-v0/g:0.95, lr:0.001, dc:0.995, dq:0/weights_0150.hdf5')



