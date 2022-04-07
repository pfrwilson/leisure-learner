

from turtle import width
from gym import Env 

from gym.spaces import Tuple, Discrete

import matplotlib.pyplot as plt
import numpy as np

ALL_ACTIONS = {
    'U': 0, 
    'D': 1,
    'C': 2, 
    'L': 4, 
    'R': 5, 
    'UL': 6, 
    'UR': 7, 
    'DL': 8, 
    'DR': 9
}

def move(action_name, x, y):
    
    assert action_name in ALL_ACTIONS.keys()
    
    if 'U' in action_name:
        x -= 1
    if 'D' in action_name: 
        x += 1
    if 'L' in action_name: 
        y -= 1
    if 'R' in action_name:
        y += 1
    
    return x, y

class WindyGridworld(Env):
    
    def __init__(self, height=20, width=10, rewards=[(1, 0, 5)], wind=True, start='random',
                 allowed_actions = ['L', 'C', 'R'], 
                 reward_terminates_episode=False): 
        
        self.start = start
        self.height = height 
        self.width = width 
        self.rewards = rewards
        self.wind = wind 
        
        self.reward_terminates_episode = reward_terminates_episode
        
        assert all(
            map(
                lambda action: action in ALL_ACTIONS.keys(),
                allowed_actions
            )
        )
        self.actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))
        
        self.observation_space = Tuple([
            Discrete(self.height), Discrete(self.width), 
        ])
        
        self.__pos = None
        
    @property
    def pos(self):
        if self.__pos is None:
            raise ValueError('environment reset needed')

        else: return self.__pos
     
    def step(self, action):
        
        x, y = self.pos
        
        assert action in range(self.action_space.n)
        x, y = move(self.actions[action], x, y)
        
        if self.wind:
            x -= 1
    
        reward = 0
        for reward_spec in self.rewards:
            value, x_, y_ = reward_spec
            if (x, y) == (x_, y_): 
                reward += value
                
        done = self.check_terminal_state(x, y)
        if self.reward_terminates_episode and reward > 0:
            done = True
        
        self.__pos = (x, y) if not done else None 
        
        return self.__pos, reward, done, {}
    
    def check_terminal_state(self, x, y):
        if x < 0: return True
        if y < 0: return True 
        if x >= self.height: return True 
        if y >= self.width: return True 
        
        return False
    
    def reset(self): 
           
        if self.start == 'random': 
            self.__pos = (self.height - 1, np.random.randint(0, self.width))
        
        else: 
            try: 
                x, y = self.start
                self.__pos = x, y
            except:
                raise KeyError()
                f'start parameter {self.start} not accepted'
        
        return self.pos
        
    def render(self, ax=None): 
        
        canvas = np.zeros((self.height, self.width))
        for value, x, y in self.rewards:
            canvas[x, y] = value
            
        canvas[self.pos] = -1
        
        if ax:
            return ax.imshow(canvas), 
        
        return plt.imshow(canvas), 
        
        