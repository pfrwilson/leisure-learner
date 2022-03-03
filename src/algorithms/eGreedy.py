import numpy as np



def eGreedy(Q, state, eps):
    r = np.random.random()
    if r < eps:
        action_space = Q.shape[3]
        a = np.random.randint(action_space)
    else:
        x = Q[state[0],state[1],0,:]
        a = np.random.choice(np.where(x == x.max())[0])
    
    return a
