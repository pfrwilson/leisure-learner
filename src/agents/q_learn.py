
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


@dataclass
class FreetimeQLearningConfig:
    f_initialization: str
    q_initialization: str
    seed: int
    epsilon: float          # epsilon greedy
    gamma: float            # discount factor
    alpha: float            # learning rate



def initialize_table(initialization, num_states, num_actions, seed):
        shape = num_states, num_actions
        if initialization == 'optimistic':
            return np.ones(shape)
        elif initialization == 'pessimistic':
            return np.zeros(shape)
        elif initialization == 'random':
            np.random.seed(seed)
            return np.random.rand(*shape)
        else: raise ValueError(f'intitialization {initialization} not supported')
        

class QLearningAgent(Agent):
    
    def __init__(self, env: Env, config: QLearningConfig, logger: Logger):
        super().__init__(env, logger)
        assert type(env.observation_space) == Discrete
        assert type(env.action_space) == Discrete
        
        self.config = config
        self.Q = initialize_table(
            self.config.q_initialization, 
            self.env.observation_space.n, 
            self.env.action_space.n, 
            self.config.seed
        )
        
    def policy(self):
        if np.random.rand() <= self.config.epsilon:
            return self.env.action_space.sample()
        else: 
            return np.argmax(self.Q[self.observation])
        
    def learn_step(self) -> bool:
        old_state = self.observation
        action = self.policy()
        new_state, reward, done, info = self.env.step(action) 
        self.observation = new_state
        
        if not done:
            target = reward + self.config.gamma * np.max(self.Q[new_state])
        else:
            target = reward
        
        update = target - self.Q[old_state, action]
        self.Q[old_state, action] = self.Q[old_state, action] + self.config.alpha * update
        
        self.logger.log_dict({
            'reward': reward, 
            'info': info, 
            'update': update, 
            'best_Q': np.max(self.Q),
        })
        
        return done
    
    
class FreetimeQLearningAgent(Agent):
    
    def __init__(self, env: Env, config: FreetimeQLearningConfig, logger: Logger):
        super().__init__(env, logger)
        assert type(self.env.action_space) == Discrete and type(self.env.observation_space) == Discrete
    
        self.config = config
        self.Q = initialize_table(
            self.config.q_initialization, 
            self.env.observation_space.n,
            self.env.action_space.n, 
            self.config.seed
        )
        self.F = initialize_table(
            self.config.f_initialization, 
            self.env.observation_space.n,
            self.env.action_space.n, 
            self.config.seed + 1
        )
        
    def policy(self):
        """
        Freetime policy implementation: 
        choose randomly when free time is available, 
        else play epsilon-greedy
        """
        
        actions_with_freetime = np.where(self.F[self.observation] >= 1)[0]
        if np.random.rand() <= self.config.epsilon:
            return self.env.action_space.sample()
        elif len(actions_with_freetime) > 0:
            return np.random.choice(actions_with_freetime)
        else: 
            return np.argmax(self.Q[self.observation])
        
    def learn_step(self):
        
        old_state = self.observation
        action = self.policy()
        new_state, reward, done, info = self.env.step(action)
        self.observation = new_state
        
        # f_reward: did the best q_value increase by less than reciprocal of discount
        if done: 
            f_reward = 0
        else:
            ratio = np.max(self.Q[old_state]) / (self.config.gamma * np.max(self.Q[new_state]) + 0.00001)
            f_reward = int(ratio < 1)
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.config.gamma * np.max(self.Q[new_state])
        q_delta = q_target - self.Q[old_state, action] 
        self.Q[old_state, action] = self.Q[old_state, action] + self.config.alpha * q_delta
        
        if done:
            f_target = f_reward
        else:
            f_target = f_reward + f_reward * np.max(self.F[new_state]) 
        f_delta = f_target - self.F[old_state, action]
        self.F[old_state, action] = self.F[old_state, action] + self.config.alpha * f_delta
        
        self.logger.log_dict({
            'q_delta': q_delta, 
            'f_delta': f_delta, 
            'reward': reward, 
            'f_reward': f_reward,
            'best_Q': np.max(self.Q), 
            'best_f': np.max(self.F),
            'info': info
        })
        
        return done