import gym
from gym import spaces
import random as rnd
import numpy as np
from collections import deque
# Imports for rendering
import time
import os

"""
Idee: 3 kamers en 1 gang. (Kamers en gang hebben grootte 1 om de omgeving wat te vereenvoudigen) 
In één kamer staat een knop (willekeurig geplaatst aan het begin van een episode)
Als de agent hiermee interageert (op drukt), verschijnt er in één van de andere kamers een koekje.
Wanneer de agent met het koekje interageert (opeten), krijgt hij 1 rewardpunt.
1 episode bestaat uit 100 Actions waarin de agent zo veel mogelijk koekjes probeert te eten.

tekening met kamernummers:
      |2| 
    |1|0|3|

De agent kan enkel dingen observeren in de kamer waar hij is.

(Voor een volgende versie kunnen we evt de kamers groter maken. 
Kan de agent dan alles in één kamer zien of alleen wat op zijn positie staat?)
"""

class CookieDomain(gym.Env):
    def __init__(self):
        '''
        Observation:
            Type: Discrete(2)
            Num     Observation               Range
            0       Room                      [0, n_rooms]
            1       Interactable              None=0, Button=1, Cookie=2 (What object is present in the Room)

        Actions:
            Type: Discrete(5)
            Num   Action
            0     Go Left
            1     Go Right
            2     Go Up
            3     Go Down
            4     (Try to) Interact with an intaractable object (Button or Cookie)
        '''
        self.n_rooms = 3 # momenteel is de rest nog hardcoded op de waarde 3

        # Button = RoomNumber
        self.button = None
        self.cookie = None

        self.n_steps = 0
        self.episode_length = 200 # nb of actions in one episode

        self.action_space = spaces.Discrete(5) # Left, Right, Up, Down, Interact(=> push button, eat cookie if possible)
        self.observation_space = spaces.Discrete(2) # (Room, interactable object)
        self.history_length = 15
        self.state_vector_size = self.observation_space.n*(self.history_length+1)

        self.state = None

        # For rendering purposes
        self.latest_action = None
        self.actions = ['left', 'right', 'up', 'down', 'interact']

        # keep track of the nb cookies eaten for debugging purposes
        self.nb_cookies_eaten = 0

        self.history = deque([(0,0) for i in range(self.history_length)],maxlen=self.history_length)

    def step(self, action):
        # Returns whether an step was taken in the good direction. This is if new_room is closer to the cookie than room.
        def _good_direction(room, new_room):
            if self.cookie == None:
                return False
            if self.cookie == 2:
                if (room == 1 and new_room == 0) or (room == 0 and new_room == 2) or (room == 3 and new_room == 0):
                    return True
            elif self.cookie == 3:
                if (room == 1 and new_room == 0) or (room == 0 and new_room == 3) or (room == 2 and new_room == 0):
                    return True
            return False

        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        self.latest_action = action
        room, _ = self.state
        new_room = room
        reward = 0.0
        if action == 0:
            reward = -1
            if room == 0:
                new_room = 1
            elif room == 3:
                new_room = 0
        elif action == 1:
            reward = -1
            if room == 1:
                new_room = 0
            elif room == 0:
                new_room = 3
        elif action == 2:
            reward = -1
            if room == 0:
                new_room = 2
        elif action == 3:
            reward = -1
            if room == 2:
                new_room = 0
        else:
            reward = -1
            if self.button == room:
                possible = [1,2,3]
                possible.remove(room)
                self.cookie = possible[rnd.randint(0,1)]
            elif self.cookie is not None and room == self.cookie:
                self.button = self._get_new_button_pos()
                self.cookie = None
                reward = 3
                self.nb_cookies_eaten += 1

        # If a step was taken in a good direction, the reward can be different.
        # It has to be seen if it matters how big this intermediate reward is.
        if _good_direction(room, new_room):
            reward = -0.5
        obj = 0
        if new_room == self.button:
            obj = 1
        elif self.cookie is not None and self.cookie == new_room:
            obj = 2

        self.state = new_room, obj
        done = (self.n_steps >= self.episode_length)
        self.end_of_episode = done
        # if done and self.nb_cookies_eaten > 30:
        #     print(f'More than 30 cookies were eaten:{self.nb_cookies_eaten}')
        # if done:
        #    print(f'Cookies eaten in {self.n_steps} steps = {self.nb_cookies_eaten}')

        self.n_steps += 1

        self.current_hist_rep = np.concatenate(list(self.history))

        self.history.append(self.state)
        s = self._state()
        return s, reward, done, {}

    def _state(self):
        return np.concatenate((np.array(self.state), self.current_hist_rep))

    # Function to get the new button location (can be random or a specific room)
    def _get_new_button_pos(self):
        return 1 #rnd.randint(1, self.n_rooms+1)

    def reset(self):
        self.nb_cookies_eaten = 0
        self.button = self._get_new_button_pos()
        self.cookie = None
        self.state = 0,0
        self.n_steps = 0
        self.history = deque([(0, 0) for i in range(self.history_length)], maxlen=self.history_length)
        self.current_hist_rep = np.concatenate(list(self.history))
        return self._state()

    def render(self, mode=None):
        def _clear():
            os.system('clear')
        def _print_world():
            top = [2]
            bottom = [1,0,3]
            agent = self.state[0]
            if agent in top:
                top = ['A']
            elif agent == 1:
                bottom = ['A', 0, 3]
            elif agent == 0:
                bottom = [1, 'A', 3]
            elif agent == 3:
                bottom = [1, 0, 'A']
            if self.cookie is not None:
                if self.cookie in top:
                    top = ['C']
                elif self.cookie in bottom:
                    bottom[2] = 'C'

            print(f'  |{top[0]}|')
            print(f'|{bottom [0]}|{bottom[1]}|{bottom[2]}|')
            #time.sleep(0.05)
            print(f'Action:{self.actions[self.latest_action]}')
            #time.sleep(0.25)
            #_clear()
            print('cookies eaten:', self.nb_cookies_eaten)
        if mode == 'human':
            _print_world()

