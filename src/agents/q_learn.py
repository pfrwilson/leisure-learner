
from dataclasses import dataclass
from .agent import Agent
from gym import Env
import numpy as np
from gym.spaces import Discrete
from ..logging.logger_base import Logger

@dataclass
class QLearningConfig:
    q_initialization: str
    seed: int
    epsilon: float          # epsilon greedy
    gamma: float            # discount factor
    alpha: float            # learning rate


class QLearningAgent(Agent):
    
    def __init__(self, env: Env, config: QLearningConfig, logger: Logger):
        super().__init__(env, logger)
        assert type(env.observation_space) == Discrete
        assert type(env.action_space) == Discrete
        
        self.config = config
        self.initialize_q_table()
        
    def policy(self):
        if self.done: 
            raise ValueError('in terminal state')
        if np.random.rand() <= self.config.epsilon:
            return self.env.action_space.sample()
        else: 
            return np.argmax(self.Q[self.observation])
        
    def learn_step(self):
        old_state = self.observation
        action = self.policy()
        obs, reward, done, info = self.play(action)
        new_state = self.observation
        target = reward + self.config.gamma * np.max(self.Q[new_state])
        update = target - self.Q[old_state, action]
        self.Q[old_state, action] = self.Q[old_state, action] + self.config.alpha * update
        self.logger.log_dict({
            'reward': reward, 
            'info': info, 
            'update': update, 
        })
    
    def initialize_q_table(self):
        shape = self.env.observation_space.n, self.env.action_space.n
        if self.config.q_initialization == 'optimistic':
            self.Q = np.ones(shape)
        elif self.config.q_initialization == 'pessimistic':
            self.Q = np.zeros(shape)
        elif self.config.q_initialization == 'random':
            np.random.seed(self.config.seed)
            self.Q = np.random.rand(*shape)
        else: raise ValueError(f'intitialization {self.config.q_initialization} not supported')
        