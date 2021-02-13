import gym
from gym import spaces
import random as rnd
import numpy as np

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
        self.step_limit = 100 # nb of actions in one episode

        self.action_space = spaces.Discrete(5) # Left, Right, Up, Down, Interact(=> push button, eat cookie if possible)
        self.observation_space = spaces.Discrete(2) # (Room, interactable object)

        self.state = None

        print('Environment initialised succesfully!')

    def step(self, action):
        err_msg = f"{action} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        room, _ = self.state
        new_room = room
        reward = 0.0
        if action == 0:
            if room == 0:
                new_room = 1
            elif room == 3:
                new_room = 0
        elif action == 1:
            if room == 1:
                new_room = 0
            elif room == 0:
                new_room = 3
        elif action == 2:
            if room == 0:
                new_room = 2
        elif action == 3:
            if room == 2:
                new_room = 0
        else:
            if self.button == room:
                possible = [1,2,3]
                possible.remove(room)
                self.cookie = possible[rnd.randint(0,1)]
            elif self.cookie is not None and room == self.cookie:
                self.button = rnd.randint(1, self.n_rooms+1)
                reward = 1.0

        obj = 0
        if new_room == self.button:
            obj = 1
        elif self.cookie is not None and self.cookie == new_room:
            obj = 2

        self.state = new_room, obj
        done = (self.n_steps >= self.step_limit)

        self.n_steps += 1

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.button = rnd.randint(1, self.n_rooms+1)
        self.cookie = None
        self.state = 0,0
        self.n_steps = 0
        return np.array(self.state)

    def render(self):
        pass