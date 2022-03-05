
from gym import Env
from typing import List, Tuple, TypeVar
from gym import spaces
import matplotlib
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, close
from matplotlib.patches import Rectangle, Circle

ObsType = TypeVar("ObsType")

class GridWorld(Env):
    
    def __init__(self, height: int, width: int, 
                 rewards: List[Tuple[float, int, int]], 
                 randstart = True):
        
        self.height = height
        self.width = width 
        self.rewards = rewards
        self.randstart = randstart
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.height * self.width)
        
        self.__reward_array = np.zeros(
            (self.height, self.width)
        )
        for value, x, y in rewards:
            self.__reward_array[x, y] = value
        
        self.__pos = None
        self.__init_pos()
        self.__done = False
        
    def __init_pos(self):
        start_x = self.height - 1
        start_y = self.width // 2 if self.randstart else np.random.randint(0, self.width)
        self.__pos = self.xy_to_obs(start_x, start_y)
        
    def obs_to_xy(self, idx):
        return idx // self.height, idx % self.height
    
    def xy_to_obs(self, x, y):
        return x * self.height + y 
    
    def step(self, action: int) -> Tuple[ObsType, float, bool, dict]:
        
        if not action in self.action_space:
            raise ValueError(f'action {action} not in {self.action_space}')
        if self.__done: 
            raise ValueError('In terminal state. Try reset()')
            
        current_x, current_y = self.obs_to_xy(self.__pos)
        
        ## terminal
        if current_x == 0: 
            self.__done = True
            return (
                self.xy_to_obs(current_x, current_y), 
                0., 
                True, 
                {}
            )
        
        if action == 0: 
            next_y = max(current_y - 1, 0)
            
        elif action == 1: 
            next_y = current_y
        
        elif action == 2: 
            next_y = min(current_y + 1, self.width - 1)
            
        next_x = current_x - 1
    
        reward = self.__reward_array[next_x, next_y]
        self.__pos = self.xy_to_obs(next_x, next_y)
        return self.__pos, reward, False, {'xy': (current_x, current_y)}
                
    def render(self, background=None):
        
        if background is not None:
            canvas = background
        else: 
            canvas = np.zeros((self.height, self.width)) 
            
        fig, axis = plt.subplots(1, 1)
        axis.set_axis_off()
        axis.imshow(canvas)

        reward_colormap = get_cmap('plasma')
        
        for value, x, y in self.rewards:
            reward = Rectangle((y-np.sqrt(2)/4, x-np.sqrt(2)/4), np.sqrt(2)/2, np.sqrt(2)/2, color=reward_colormap(value))
            axis.add_patch(reward)
    
        current_x, current_y = self.obs_to_xy(self.__pos)
        color = 'green' if self.__reward_array[current_x, current_y] != 0 else 'blue'
        axis.add_patch(Circle((current_y, current_x), .5, color=color))

    def reset(self):
        
        self.__done = False
        self.__init_pos        
