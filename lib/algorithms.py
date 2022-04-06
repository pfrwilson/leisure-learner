from os import terminal_size
from typing import Literal, Tuple, Union
from matplotlib.pyplot import hist
import numpy as np
from .gym_windy_gridworld import WindyGridworld
from tqdm import tqdm

def build_q_table(state_shape: Tuple[int], num_actions: Tuple[int], 
                  initialization: Literal['random', 'pessimistic', 'optimistic'] = 'pessimistic'):
    
    shape = (*state_shape, num_actions)
    
    if initialization == 'random': 
        return np.random.rand(*shape)

    elif initialization == 'pessimistic': 
        return np.zeros(shape)
    
    elif initialization == 'optimistic': 
        return np.ones(shape)


def select_greedy(Q: np.ndarray, state: Union[int, Tuple[int, int]]): 
    
    x = Q[state]
    a = np.random.choice(np.where(x == x.max())[0])
    
    return a

    
def select_epsilon_greedy(Q: np.ndarray, state: Union[int, Tuple[int]], epsilon: float):
    
    if np.random.rand() < epsilon:
        actions = Q[state]   
        return np.random.choice(np.arange(actions.size))
    
    else: return select_greedy(Q, state)


def select_epsilon_greedy_freetime(Q: np.ndarray, F, state, epsilon: float):
    
    if np.random.rand() < epsilon:
        actions = Q[state]
        return np.random.choice(np.arange(actions.size))
    
    freetime_actions = np.where(F[state] >= 1)[0]
    if len(freetime_actions) > 0:
        return np.random.choice(freetime_actions)
    
    else: 
        return select_greedy(Q, state)


def Q_learn(env, Q, num_steps, epsilon, discount, alpha):
    
    rewards = np.zeros(num_steps)
    
    state = env.reset()
    
    for step in tqdm(range(num_steps)):
        
        action = select_epsilon_greedy(Q, state, epsilon)
        
        new_state, reward, terminal_state, info = env.step(action)
        
        if terminal_state: 
            target = reward
        else: 
            target = reward + discount * Q[new_state].max()
        
        Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
        
        rewards[step] = reward
        
        if terminal_state:
            state = env.reset()
        else:
            state = new_state
            
        step += 1

    return Q, np.cumsum(rewards)
    
    
def Q_learn_freetime(env, Q, num_steps, epsilon, discount, alpha, alpha_f):
    
    F = np.ones_like(Q)
    
    rewards = np.zeros(num_steps)
    freetime_rewards = np.zeros(num_steps)
    
    state = env.reset()
    
    for step in tqdm(range(num_steps)):
        
        action = select_epsilon_greedy_freetime(Q, F, state, epsilon)
        
        new_state, reward, terminal_state, info = env.step(action)
        
        if terminal_state:
            q_target = reward
        else: 
            q_target = reward + discount * Q[new_state].max()
        
        if terminal_state:
            freetime_reward = 0
            freetime_target = 0
        else:
            new_Q = Q[new_state].max() * discount
            old_Q = Q[state].max()
            freetime_reward = 1 if new_Q + 0.01 >= old_Q else 0
            freetime_target = freetime_reward + F[new_state].max() if freetime_reward == 1 else 0
        
        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * q_target
        F[state][action] = (1 - alpha_f) * F[state][action] + alpha_f * freetime_target
        
        rewards[step] = reward
        freetime_rewards[step] = freetime_reward
        
        if terminal_state:
            state = env.reset()
        else: 
            state = new_state
            
    return Q, F, np.cumsum(rewards), np.cumsum(freetime_rewards)
        
        