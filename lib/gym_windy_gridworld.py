

from turtle import width
from gym import Env 

from gym.spaces import Tuple, Discrete

import matplotlib.pyplot as plt
import numpy as np

class WindyGridworld(Env):
    
    def __init__(self, height=20, width=10, rewards=[(1, 0, 5)], wind=True, start='random',
                 allowed_actions = ['left', 'center', 'right'], 
                 reward_terminates_episode=False): 
        
        self.start = start
        self.height = height 
        self.width = width 
        self.rewards = rewards
        self.wind = wind 
        
        self.reward_terminates_episode = reward_terminates_episode
        
        assert all(
            map(
                lambda action: action in ['left', 'center', 'right', 'up', 'down'],
                allowed_actions
            )
        )
        self.actions = allowed_actions
        self.action_space = Discrete(len(allowed_actions))
        
        #self.actions = {
        #    'left': 0, 
        #    'center': 1, 
        #    'right': 2, 
        #    'up': 3, 
        #    'down': 4,
        #}
        
        self.observation_space = Tuple([
            Discrete(self.height), Discrete(self.width), 
        ])
        
        self.__pos = None
        
    @property
    def pos(self):
        if self.__pos is None:
            raise ValueError('environment reset needed')

        else: return self.__pos
        
    def move(self, action, x, y):
        
        if action not in range(len(self.actions)): 
            raise ValueError(f'action {action} not supported.')
        
        if self.actions[action] == 'up':
            x -= 1
            y = y
        
        if self.actions[action] == 'center':
            x = x 
            y = y
        
        if self.actions[action] == 'left':
            y -= 1
            x = x
            
        if self.actions[action] == 'right':
            y += 1 
            x = x
            
        if self.actions[action] == 'down':
            x += 1 
            y = y
        
        return x, y
     
    def step(self, action):
        
        x, y = self.pos
        
        x, y = self.move(action, x, y)
        
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
        
        