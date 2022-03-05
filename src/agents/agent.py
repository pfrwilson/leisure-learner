
from abc import ABC, abstractmethod
from gym import Env
from ..logging.logger_base import Logger
from typing import Optional
import time

class Agent(ABC):
    
    def __init__(self, env: Env, logger: Logger):
        self.env = env
        self.logger = logger
        self.reset()
        self.current_episode = 0
        self.global_step = 0
        
    def reset(self):
        self.observation = self.env.reset()
        self.done = False
    
    def play(self, action):
        if self.done: 
            raise ValueError('In terminal state')
        obs, value, done, info = self.env.step(action)
        self.observation = obs
        self.done = done
        return obs, value, done, info

    def learn_episode(self, step_limit: int = 1e6):
        step = 0
        while step < step_limit and not self.done:
            self.learn_step()                                      
            self.global_step +=1
        self.current_episode += 1
        self.reset()
            
    @abstractmethod
    def policy(self):
        """
        return an action given the current observation 
        """
        pass
        
    @abstractmethod
    def learn_step(self):
        """
        perform a learning step. 
        """    



        
    