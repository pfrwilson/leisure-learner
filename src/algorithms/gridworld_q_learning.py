

raise DeprecationWarning()

from typing import List
import numpy as np
from gym import Env
from tqdm import tqdm

def initialize_q_table(initialization: str, shape, seed=0):
    
    if initialization == 'optimistic':
        return np.ones(shape)
    elif initialization == 'pessimistic':
        return np.zeros(shape)
    elif initialization == 'random':
        np.random.seed(seed)
        rng = np.random.rand(**shape)
    else: raise ValueError(f'intitialization {initialization} not supported')


def q_learn_step(Q, s, a, r, gamma, next_s, alpha):
    
    target = r + gamma * Q[next_s, :].max()

    delta = Q[s, a] - target
    qdrop = Q[s1[0], s1[1], action] - disct * Q[s2[0], s2[1], :].max()
    ratio = Q[s1[0], s1[1], :].max() / ( disct* Q[s2[0],s2[1],:].max() + 0.00001)
    
    if rew == 1:
        ratio = 0

    Q[s, a] = Q[s, a] - alpha*delta

    return Q, delta, qdrop, ratio


def q_learn(Q, env: Env, gamma, alpha, n_steps):
    
    s = env.reset()
    
    with tqdm(range(n_steps), desc='Step #') as pbar:
        for step in pbar:
            
            
    