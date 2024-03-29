import gym
import envs
import random
import numpy as np
import os
import itertools

render = False  # Display graphs and render environment? -> not possible on remote machine

from collections import deque # List which allows to add items in the front and to the end
from tensorflow import keras  # To crate neural network to approximate optimal policy
import matplotlib
if not render:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress annoying warnings


# PARAMETERS 

environment = "CookieDomain-v0" # Name of the environment we'll be training in
#environment = "CartPole-v0" # Name of the environment we'll be training in
env = gym.make(environment)


state_size = env.observation_space.n  #shape[0]
#state_size = env.observation_space.shape[0]   #shape[0]
acion_size = env.action_space.n

batch_size = 64  # Should be a power of 2
n_episodes = 1000 # Number of games we want to play

history_size = 7 # 1 + nb of previous states to use

dqn_input_vector_size = state_size*history_size  # what is the size of the inputlayer of the dqn
                                                 # This is the same as the history size*state_size if the representation
                                                 # only contains current and previous states. But it can differ
                                                 # if extra information is added

end_episode_reward = 0.0  # What reward is given at the end of an episode. Can be negative if ending the episode is
                          # considered a bad thing. (e.g in cookiedomain-v0 it is not considered bad because one episode
                          # is always the same length


output_dir = f'model_output/{environment}/'

# Create output directory if it doesn't already exist 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class ReplayMemory:
    def __init__(self, maxlen):
        self.history_size = history_size
        self.maxlen = maxlen
        self.observations = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)
        self.repr = 0  # What representation to use? (In get_repr_at method)

    def add_experience(self, state, action, reward, done):
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_repr_at(self, i):
        # Representation 0 -> return a vector containing history size states
        if self.repr == 0:
            n = self.history_size
            states = list(itertools.islice(self.observations, i - n, i))
            states = np.reshape(np.concatenate(states).ravel(), [1, dqn_input_vector_size])
            return states
        # Representation 1 -> same as repr 0, but an extra element is appended expressing whether
        # an interaction was done in the past states or not.
        # (But is does not mean that that interaction was any good:
        # pressing multiple times on the button, interacting when no button or cookie is present,...)
        elif self.repr == 1:
            n = self.history_size
            states = list(itertools.islice(self.observations, i - n, i))
            #states = np.reshape(np.concatenate(states).ravel(), [1, state_size * n])
            past_actions = list(itertools.islice(self.actions, i - n, i))
            did_interaction = past_actions.__contains__(4)
            to_add = 0
            if did_interaction:
                to_add = 1
            repr = np.append(states, to_add)
            repr = np.reshape(repr, [1, self.history_size*state_size+1])
            return repr


    # Returns the representation of the state at position i of the replay memory
    # The states and new_states variables are thus represented as specified in get_repr_at method
    def get_state(self, i):
        i = int(i)
        # stel i is index in self.observations
        # return [[i-n...i], action in i, reward in i, [i-n+1, i+1], done in i]
        states = self.get_repr_at(i)
        new_states = self.get_repr_at(i + 1)
        # return de state (= lijst met voorlaatste state en de n-1 states daarvoor, de actie die leidt tot de
        # laaste state en de n-1 states daarvoor, en de reward die je voor die actie krijgt)
        return [states, self.actions[i], self.rewards[i], new_states, self.dones[i]]



    # maak een mini batch die bestaat uit batch_size elementen (één element is hetgeen wat de get_state funtie returnt)
    def get_mini_batch(self, batch_size):
        ind_range = list(range(history_size, len(self.observations)-1))
        # Steps to make semi-random batch, which slightly favors positive rewards
        rew = self.rewards
        rew = list(itertools.islice(rew, history_size, len(self)-1))
        reward_at = list(zip(ind_range, rew))
        # Sort the list with (index, reward) tuples on reward
        srtd = sorted(reward_at, key=lambda x: x[1], reverse=True)
        srtd = np.array([tup[0] for tup in srtd])
        # Give each index a specific (decreasing) probability
        p = np.array([0.99**i for i in range(len(srtd))])
        # Make sure the rewards add up to one
        p = p / sum(p)
        # Construct the minibatch of indexes
        somewhat_random_indices = np.random.choice(srtd, size=batch_size, p=p)
        random_indices = random.sample(ind_range, batch_size)
        # return a list of the state representations of the indexes
        return [self.get_state(i) for i in somewhat_random_indices]

    # This function is used to reset empty the replay memory and fill it with zeros.
    # It is needed between two episodes in the testing environment. (Cause it is the same agent
    # and we don't want the agent to base his decisions on previous episodes)
    def reset(self):
        for i in range(self.maxlen):
            self.add_experience(np.array([0]*state_size),0,0,False)

    # The length of a ReplayMemory object is defined as the length of the observations attribute
    def __len__(self):
        return len(self.observations)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Replay memory and size
        self.memory = ReplayMemory(2000) #deque(maxlen=2000) # When deque length exceeds 2000, oldest elements will be removed when adding new ones

        # Discount factor to discount future rewards
        self.gamma = 0.95


        ### DOUBLE Q-LEARNING PARAMETERS ###
        self.target_update_method = 0  # 0 for Polyak update; 1 for updating after 'update_freq' memory replays
        self.tau = 0.01  # Parameter of double-q learning: determines the speed of averaging between model and target model
        # Parameters for other double-q learning method
        self.model_t_update_counter = 0
        self.update_freq = 1

        # Percentage of actions to be random
        self.epsilon = 1.0
        
        # As the agent is learning, decrease the epsilon factor to shift focus from exploration to exploitation
        self.epsilon_decay = 0.995

        # Minimum value for epsilon
        self.epsilon_min = 0.1

        # Parameters for model evaluation
        self.epsilon_greedy = 0.05  # Use sometimes random actions in stead of the model output
        self.do_evaluate = True
        self.show_progressbar = False
        # This is the memory used in a test method
        self.model_memory = ReplayMemory(history_size) #deque(history_size*state_size*[0], maxlen=history_size*state_size)
        self.model_memory.reset()

        self.learning_rate = 0.001

        self.timeout = 250  # Maximum steps in one episode (CartPole is limited to 200 by itself)

        # Build model and target model
        self.internal_layers = [20,30,20]
        self.model = self._build_model()
        self.model_t = self._build_model()
        self.model_t.set_weights(self.model.get_weights())

        self.name = f'g:{self.gamma}, lr:{self.learning_rate}, dc:{self.epsilon_decay}, ' \
                    f'dq:{self.target_update_method}, net:{self.internal_layers},hist:{history_size}, ep:{n_episodes}'

    # Design of the deep-q neural network to approximate optimal policy
    def _build_model(self):
        model = keras.models.Sequential()
        model.add(
                  keras.layers.Dense(self.internal_layers[0],
                  input_dim=dqn_input_vector_size,
                  activation='relu')
                  )

        for i in range(1, len(self.internal_layers)):
            model.add(keras.layers.Dense(self.internal_layers[i], activation='relu'))

        # Add ouput layer, which has a neuron for each action. We choose a linear model s.t. the action astimates are directly correlated to the neurons
        model.add(keras.layers.Dense(self.action_size, activation='linear'))

        # mse (mean squared error) loss turns out to work more accurately than cross entropy in this case
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # Appends an experience to the replay memory
    # Models, given an action and state, what will happen in the next state and the associated reward
    def remember(self, state, action, reward, next_state, done):
        #self.memory.append([state, action, reward, next_state, done])
        self.memory.add_experience(state, action, reward, done)

    # Choose an action based on the current state or a random action with probability epsilon
    def act(self, state, test=False):
        epsilon = self.epsilon
        if test:
            epsilon = self.epsilon_greedy
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)  # exploratory action
        # Ook wanneer je het model test wilt het model een vector als input die ook de history bevat.
        # de agent bevat dus ook een atribuut (=model_memory) dat bijhoudt wat de vorige observaties waren. (zie evaluate functie)
        if test:
            repr = self.model_memory.get_repr_at(len(self.model_memory))
            act_value = self.model.predict(repr)
            return np.argmax(act_value[0])

        act_value = self.model.predict(state)         # exploitative action
        return np.argmax(act_value[0])

    # Trains the model on batch_size experiences picked from self.memory at random
    def replay(self, batch_size):
        # Training is possible if memory has enough experiences
        if len(self.memory.observations) < batch_size+history_size: #len(self.memory) < batch_size:
            return
        # Pick random minibatch from replay memory
        # To train neural network and improve approximation of optimal policy
        minibatch = self.memory.get_mini_batch(batch_size) #random.sample(self.memory, batch_size)
        # Train the model on each element in the minibatch
        for state, action, reward, next_state, done in minibatch:
            # Determine the target reward
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model_t.predict(next_state)[0])
            # Maximize future reward to current reward:
            target_f = self.model.predict(state)
            # Map terget for current state to future state
            target_f[0][action] = target

            # One backpropagation step: one step in fitting model to optimal policy
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
    def evaluate(self, n=15):
        avr = []
        for _ in tqdm(range(n), disable=not self.show_progressbar):
            s = env.reset()
            s = np.reshape(s, [1, state_size])
            current_reward = []
            self.model_memory.reset()
            for j in range(self.timeout):
                if render:
                    env.render()
                a = agent.act(s, test=True)
                ns, r, d, _ = env.step(a)
                r = r if not d else end_episode_reward
                current_reward.append(r)
                #ns = np.reshape(ns, [1, state_size])
                # Add the current observation to the memory so this information is still present in n steps
                self.model_memory.add_experience(ns,a,r,d)
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
    for e in range(n_episodes+1):
        current_reward = 0
        state = env.reset()  # Start each episode at the beginning of an episode:
        state = np.reshape(state, [1, state_size])

        # Play episode (until done or timeout (= maximum game time))
        for time in range(agent.timeout):
            if render:
                env.render()            # Will only work on a local machine

            # als je nog niet voldoende observaties deed doe je gewoon of de agent nu en in het verleden allemaal nullen zag
            # Dit zou nog aangepast kunnen worden naar bvb de observaties aanvullen tot het jusite aantal. Stel dat je bvb nog maar één observatie deed
            # dan kan je deze observatie aanvullen met 0'en zodat je hitory_size observaties in history_rep hebt.
            if len(agent.memory.observations) < history_size:
                history_rep = dqn_input_vector_size*[0]
            else:
                # If the replay memory is large enough, the history representation of the current state is asked
                history_rep = agent.memory.get_repr_at(len(agent.memory))

            action = agent.act(history_rep) #agent.act(state)  # action is either 0 or 1: move left or right respectively
            next_state, reward, done, _ = env.step(action)  # Pass in an action and retrieve next state and reward
            reward = reward if not done else end_episode_reward  # -10 is the penalty applied for poor actions
            current_reward += reward
            # next_state = np.reshape(next_state, [1, state_size])

            agent.remember(next_state, action, reward, next_state, done)
            # state = next_state

            # What to do after each episode: evaluate agent if needed and print progress
            if done:
                if agent.do_evaluate: #or e % 100 == 0:  # uncomment to evaluate agent every 100 episodes
                    avv_reward = agent.evaluate()
                    rewards.append(avv_reward)
                    print(f"episode {e}/{n_episodes}, average reward: {avv_reward}, e: {agent.epsilon:.2}")
                else:
                    print(f"episode {e}/{n_episodes}, current_reward: {current_reward}, e: {agent.epsilon:.2}")
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
        dir_name = f'graphs/{environment}/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(dir_name + agent.name + '.png')
        if render:
            plt.show()  # (uncomment to display graph after training, only works when having a screen (not via ssh))


# Method to test an agent for given model (name = path to .hdf5 file)
# Possible improvement: Some variables have to be te same as in training (network layout, ...) but
# this is not automatically done right now
def test_agent(name):
    agent.load(name)
    print('max achievable reward: 199')
    n = 10
    for j in range(n):
        s = env.reset()
        s = np.reshape(s, [1, state_size])
        for i in range(agent.timeout):
            if render:
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
    main() #uncomment to train a model
    #test_agent('model_output/CartPole-v0/g:0.95, lr:0.001, dc:0.995, dq:0/weights_0150.hdf5')



