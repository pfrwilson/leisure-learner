

from turtle import width
from gym import Env 

from gym.spaces import Tuple, Discrete

import matplotlib.pyplot as plt
import numpy as np

class WindyGridworld(Env):
    
    def __init__(self, height, width, rewards, wind, start='random', 
                 reward_terminates_episode=False): 
        
        self.start = start
        self.height = height 
        self.width = width 
        self.rewards = rewards
        self.wind = wind 
        
        self.reward_terminates_episode = reward_terminates_episode
        
        self.action_space = Discrete(3)
        self.actions = {
            'left': 0, 
            'center': 1, 
            'right': 2
        }
        
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
        
        wind = self.get_wind(x, y)
        
        # move
        if action == self.actions['left']:
            y -= 1
        elif action == self.actions['right']: 
            y += 1
        elif action == self.actions['center']:
            y = y 
        else: 
            raise KeyError(f'action {action} unknown')
        
        x -= wind[0]
        y -= wind[1]
    
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
        
    def get_wind(self, x, y):
        # wind only goes in 1 direction for now
        
        return self.wind[y], 0 
    
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
        
        