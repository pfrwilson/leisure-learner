raise DeprecationWarning()

from typing import List
import numpy as np
from gym import Env
from tqdm import tqdm
from ..logging.logger_base import Logger

def initialize_q_table(initialization: str, n_states, n_actions, seed=0):
    
    shape = n_states, n_actions
    
    if initialization == 'optimistic':
        return np.ones(shape)
    elif initialization == 'pessimistic':
        return np.zeros(shape)
    elif initialization == 'random':
        np.random.seed(seed)
        return np.random.rand(*shape)
    else: raise ValueError(f'intitialization {initialization} not supported')

def e_greedy(Q, s, epsilon):
    
    if np.random.rand() <= epsilon:
        a = np.random.randint(0, Q.shape[1])
    else: 
        a = np.argmax(Q[s])    
    
    return a

def q_learn_step(Q, s, a, r, gamma, next_s, alpha):
    
    target = r + gamma * Q[next_s, :].max()

    delta = Q[s, a] - target

    Q[s, a] = Q[s, a] - alpha*delta

    return delta

def q_learn(Q, env: Env, gamma, alpha, epsilon, n_steps, logger: Logger):
    
    s = env.reset()
    episode = 0
    
    with tqdm(range(n_steps), desc='Step #') as pbar:
        for step in pbar:
            
            a = e_greedy(Q, s, epsilon)
            
            obs, r, done, info = env.step(a)
            
            if done:       # terminal state
                Q[s, a] = Q[s, a] + (r - Q[s, a])
                episode += 1 
                s = env.reset()
                continue      
            
            s_next = obs
            
            delta = q_learn_step(Q, s, a, r, gamma, s_next, alpha)
            
            s = s_next
            
            logger.log_dict({
                'reward': r, 
                'delta': delta, 
                'episode': episode
            })
            
            if step % 100 == 0:
                pbar.set_postfix({
                    'reward': r, 
                    'episode': episode
                })
