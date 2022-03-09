import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.play_baseline import play_baseline
from src.environments.GridWorld import GridWorld
import os

def main():

    eps = 0.05
    alpha = 0.01
    wraparound = False
    randstart = True
    steps = 100000

    dim = (20,10,1,3)

    Q1 = np.ones(dim)
    world_dim = [20,10]
    rewards = [[1,0,0],[1,5,9]]
    env = GridWorld( world_dim, rewards, randstart = True)
    score_opt, Q1 = play_baseline(env, steps, Q1, alpha, eps)

    Q2 = np.random.rand(20,10,1,3)
    env = GridWorld(wraparound, randstart)
    score_rand, Q2 = play_baseline(env, steps, Q2, alpha, eps)

    Q3 = np.zeros(dim)
    env = GridWorld(wraparound, randstart)
    score_pess, Q3 = play_baseline(env, steps, Q3, alpha, eps)

    plt.plot(score_opt, label= "Optimistic Init" )
    plt.plot(score_rand, label= "Random Init" )
    plt.plot(score_pess, label= "Pessimistic Init" )
    plt.title("Cumulated score over steps")
    plt.legend()
    plt.show()

    max_opt_val = np.max(Q1,axis=3)
    print(max_opt_val[:,:,0].shape)
    plt.imshow(max_opt_val[:,:,0], vmin=0,vmax=1,cmap="jet")
    plt.colorbar()
    plt.title("Opt init Q-Table")
    plt.show()

    max_rand_val = np.max(Q2,axis=3)
    print(max_opt_val[:,:,0].shape)
    plt.imshow(max_rand_val[:,:,0], vmin=0,vmax=1 ,cmap="jet")
    plt.colorbar()
    plt.title("random init Q-Table")
    plt.show()

    max_pess_val = np.max(Q3,axis=3)
    print(max_opt_val[:,:,0].shape)
    plt.imshow(max_pess_val[:,:,0], vmin=0,vmax=1, cmap="jet")
    plt.colorbar()
    plt.title("Pess init Q-Table")
    plt.show()

    action_opt = np.argmax(Q1,axis=3)
    plt.imshow(action_opt[:,:,0])
    plt.colorbar()
    plt.title("Opt init Action map")
    plt.show()

    action_rand = np.argmax(Q2,axis=3)
    plt.imshow(action_rand[:,:,0])
    plt.colorbar()
    plt.title("Random init action map")
    plt.show()

    action_pess = np.argmax(Q3,axis=3)
    plt.imshow(action_pess[:,:,0])
    plt.colorbar()
    plt.title("Pess init Q-Table")
    plt.show()
    

if __name__ == '__main__':
    main()