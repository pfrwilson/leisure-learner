
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

    def learn_episode(self, step_limit: int = 1e6):
        step = 0
        while step < step_limit: 
            done = self.learn_step()                                      
            self.global_step +=1
            if done: break
            
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



        
    