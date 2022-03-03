from GridWorld import *
import numpy as np
from eGreedy import eGreedy
from Qlearn import Qlearn


def play_baseline(steps, wraparound, randstart, Q, alpha, eps):

    X = 0
    Y = 1
    Z = 2

    epicount = 0
    epirew = 0
    episteps = 0
    score = np.zeros(steps)
    episode_steps = np.zeros(int(steps//10))
    episode_rewards = np.zeros(int(steps//10))

    grid_wrold = GridWorld(wraparound,randstart)

    i=0
    while i<steps:
        s1 = grid_wrold.get_state()
        action = eGreedy(Q, s1, eps)
        rew = grid_wrold.take_action(action)
        s2 = grid_wrold.get_state()



        disct = grid_wrold.get_disct()
        Q, delta, qdrop, ratio = Qlearn(Q,s1,action,rew,disct,s2,alpha)
        epirew += rew
        episteps += 1
        if disct == 0:
            
            episode_steps[epicount] = episteps
            episode_rewards[epicount] = epirew
            epicount += 1

            # Print end of episode info 
            if epicount%100==0:
                print('Episode:', epicount, '| reward =',epirew,' steps =', episteps)

            epirew = 0
            episteps = 0
            grid_wrold.reset()

        score[i] = rew
        
        i+=1
    return (np.cumsum(score), Q)

    