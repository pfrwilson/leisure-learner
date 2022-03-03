from ..environments.environment_base import EnvironmentBase
import numpy as np
from .eGreedy import eGreedy
from .Qlearn import Qlearn


def play_freetime(env: EnvironmentBase, steps, Q, alpha, eps, discount, alpha2):

    epicount = 0
    epirew = 0
    episteps = 0
    epifree = 0
    score = np.zeros(steps)
    episode_steps = np.zeros(int(steps//10))
    episode_rewards = np.zeros(int(steps//10))
    episode_time = np.zeros(int(steps//10))

    F = np.ones((env.rows,env.col,1,3))
    #F =  np.zeros((env.rows,env.col,1,3))

    i=0
    while i<steps:
        s1 = env.get_state()

        if s1 in [[1,5],[1,6],[1,4]]:
            c = 2

        if s1[0] == 1:
            debug=True

        # Check if an action has free time
        state_actions = F[s1[0],s1[1],0,:]
        time = np.where(state_actions>=1)[0]
        
        if len(time) >= 1:
            # play random on free time 
            # *Only amoung those actions with Freetime
            #action_space = Q.shape[3]
            #action = np.random.randint(action_space)
            action = np.random.choice(time)
        else:
            # Play egreedy
            action = eGreedy(Q, s1, eps)
        # play our action, get new state and disct,
        rew = env.take_action(action)
        s2 = env.get_state()
        disct = env.get_disct()

        # Learn Q
        Q, delta, qdrop, ratio = Qlearn(Q,s1,action,rew,disct,s2,alpha)
        # Learn F 
        Frew = int(ratio < 1)
        Fdisct = int(Frew)
        F = Qlearn(F, s1, action, Frew, Fdisct, s2, alpha2)[0]
        epirew = epirew + rew
        epifree = epifree + Frew
        episteps = episteps + 1
        if disct == 0:

            episode_steps[epicount] = episteps
            episode_rewards[epicount] = epirew
            episode_time[epicount] = epifree
            epicount += 1

            # Print end of episode info 
            # Print end of episode info 
            if epicount%100==0:
                print('Episode:', epicount, '| reward =',epirew,' steps =', episteps, ' freetime =',epifree)

            epirew = 0
            episteps = 0
            epifree = 0 

            env.reset()

        score[i] = rew

        i += 1
    
    return np.cumsum(score), Q, F