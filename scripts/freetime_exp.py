import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.play_freetime import play_freetime
from src.environments.GridWorld import GridWorld

eps = 0.05
alpha = 0.01
wraparound = False
randstart = True
steps = 300000
discount = 0.98

dim = (20,10,1,3)

Q1 = np.ones(dim)
world_dim = [20,10]
rewards = [[1,0,7],[1,4,5]]
env = GridWorld(world_dim, rewards, randstart = True)
score_opt, Q1, F1 = play_freetime(env, steps, Q1, alpha, eps, discount, alpha)

Q2 = np.random.rand(20,10,1,3)
world_dim = [20,10]
rewards = [[1,0,7],[1,4,5]]
env = GridWorld(world_dim, rewards, randstart = True)
score_rand, Q2, F2 = play_freetime(env, steps, Q2, alpha, eps, discount, alpha)

Q3 = np.zeros(dim)
world_dim = [20,10]
rewards = [[1,0,7],[1,4,5]]
env = GridWorld(world_dim, rewards, randstart = True)
score_pess, Q3, F3 = play_freetime(env, steps, Q3, alpha, eps, discount, alpha)


plt.plot(score_opt, label= "Optimistic Init" )
plt.plot(score_rand, label= "Random Init" )
plt.plot(score_pess, label= "Pessimistic Init" )
plt.title("Freetime cumulative score over steps")
plt.legend()
plt.show()


max_opt_val = np.max(Q1,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet", vmax= 2, vmin= 0)
plt.title("Freetime Opt Q")
plt.colorbar()
plt.show()

max_opt_val = np.max(Q2,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet", vmax= 2, vmin= 0)
plt.title("Freetime Rand Q")
plt.colorbar()
plt.show()

max_opt_val = np.max(Q3,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet", vmax= 2, vmin= 0)
plt.title("Freetime Pess Q")
plt.colorbar()
plt.show()

############################ PLOT F

max_opt_val = np.max(F1,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet")
plt.title("Freetime Opt F")
plt.colorbar()
plt.show()

max_opt_val = np.max(F2,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet")
plt.title("Freetime Rand F")
plt.colorbar()
plt.show()

max_opt_val = np.max(F3,axis=3)
print(max_opt_val[:,:,0].shape)
plt.imshow(max_opt_val[:,:,0], cmap="jet")
plt.title("Freetime Pess F")
plt.colorbar()
plt.show()

###################

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