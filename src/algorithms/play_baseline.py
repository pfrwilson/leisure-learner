from ..environments.environment_base import EnvironmentBase
import numpy as np
from .eGreedy import eGreedy
from .gridworld_q_learning import q_learn


def play_baseline(env: EnvironmentBase, steps, Q, alpha, eps):

    X = 0
    Y = 1
    Z = 2

    epicount = 0
    epirew = 0
    episteps = 0
    score = np.zeros(steps)
    episode_steps = np.zeros(int(steps//10))
    episode_rewards = np.zeros(int(steps//10))

    i=0
    while i<steps:
        s1 = env.get_state()
        action = eGreedy(Q, s1, eps)
        rew = env.take_action(action)
        s2 = env.get_state()



        disct = env.get_disct()
        Q, delta, qdrop, ratio = q_learn(Q,s1,action,rew,disct,s2,alpha)
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
            env.reset()

        score[i] = rew
        
        i+=1
    return (np.cumsum(score), Q)

    