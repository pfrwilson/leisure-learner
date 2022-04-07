
from gym import Env
import seaborn as sns
import matplotlib.pyplot as plt
from .algorithms import select_greedy
import numpy as np
from typing import Literal
import pandas as pd
from itertools import product

#from .algorithms import (
#    q_learn, 
#    q_learn_eligibility_traces, 
#    sarsa, 
#    sarsa_eligibility_traces,
#    select_epsilon_greedy,
#    select_greedy
#)

#ALGORITHMS = {
#    'q_learn': q_learn, 
#    'sarsa': sarsa, 
#    'q_learn_eligibility_traces': q_learn_eligibility_traces, 
#    'sarsa_eligibility_traces': sarsa_eligibility_traces
#}


#def detect_convergence(values, patience, delta):
#    """ 
#    Detects convergence (determined by a change of at most delta in the values for 
#    at least "patience" consecutive indices) and returns the index and value where convergence has 
#    occurred, otherwise returns None. 
#    """
#    
#    counter = 0
#    prev_value = 0
#    
#    for i, value in enumerate(values):
#        if np.abs(value - prev_value) < delta:
#            counter += 1
#        else: 
#            counter = 0 
#        if counter >= patience: 
#            return i, value
#        prev_value = value
#
#
#def find_best_eps_alpha(
#    env: Env, 
#    algorithm: Literal['q_learn', 'sarsa', 'q_learn_eligibility_traces', 'sarsa_eligibility_traces'], 
#    resolution: int, 
#    num_steps: int, 
#    num_runs: int = 1
#):
#    """
#    Performs a grid search to determine the performance of all choices for
#    epsilon and alpha. Tries all epsilons and alphas in a between 0 and 1 with a certain
#    spacing based on the resolution parameter. 
#    s
#    @params:
#        env: the environment within which we are operating (one of the gridworld
#        environments)
#        algorithm: which algorithm to use
#        resolution: how many alpha and epsilon values to try - higher means exponentially 
#                longer run time
#        num_step: the number of steps to run each set of parameters. 
#        num_runs: number of runs to average results over
#        
#    returns:
#        a table of scores for each pair of epsilon and alpha. The score is the 
#        number of episodes completed after reaching the step limit. 
#    """
#
#    alphas = np.linspace(1/resolution, 1, resolution)
#    epsilons = np.linspace(1/resolution, 1, resolution)
#
#    tables = [] 
#    best_params_list = []
#
#    table = pd.DataFrame()
#    table.index.name = 'alpha'
#    table.columns.name = 'epsilon'
#
#    best_score = 0
#    best_params = (-1, -1)
#
#    for i in range(num_runs):
#        for alpha, epsilon in product(alphas, epsilons): 
#            
#            Q, history = ALGORITHMS[algorithm](
#                env, 10000, 1, epsilon, 'random', alpha, step_limit=num_steps,
#            )
#
#            final_score = history['episode_by_timestep'][-1]        
#            table.loc[f'{alpha:.2f}', f'{epsilon:.2f}'] = final_score
#            if final_score > best_score:
#                best_score = final_score
#                best_params = alpha, epsilon
#
#        tables.append(table)
#        best_params_list.append(best_params)
#    
#    best_params = ( sum([alpha for alpha, epsilon in best_params_list]) / len([alpha for alpha, epsilon in best_params_list]), 
#                    sum([epsilon for alpha, epsilon in best_params_list]) / len([epsilon for alpha, epsilon in best_params_list]) ) 
#    
#    table = sum(tables) / len(tables)
#    
#    return table, best_params
#
#
#def make_grid_search_plots(env, resolution, num_steps, num_runs=1):
#    """
#    Creates plots showing the heatmaps of the scores for various 
#    choices of epsilon and alpha for the different algorithms.
#    """
#
#    tables = {}
#    best_params = {}
#
#    for algorithm in ALGORITHMS.keys():
#        fname = f'{algorithm}_grid_search_res_{resolution}_nsteps_{num_steps}.csv'
#        table, params = find_best_eps_alpha(env, algorithm, resolution, num_steps, num_runs=num_runs)
#        tables[algorithm] = table
#        best_params[algorithm] = params 
#
#    vmax = max([np.max(table.values) for table in tables.values()])    
#    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#    for (key, table), ax in zip(tables.items(), axs.flatten()):
#        sns.heatmap(table, annot=resolution <= 8, fmt='g', cmap='viridis', ax=ax, vmax=vmax)
#        ax.set_title(key)
#        
#    fig.suptitle(f'Episodes completed after {num_steps} steps')
#    fig.tight_layout()
#    
#    return fig, tables, best_params


def make_arrow(ax, start, stop):
    
    y, x = start 
    y_1, x_1 = stop
    dy = y_1 - y
    dx = x_1 - x
    
    ax.arrow(x, y, dx, dy, width=0.2, length_includes_head=True, facecolor='white', edgecolor='black')
    
    
def make_trajectory_map(Q, env: Env, max_steps=100, show_multiple=True, title=None, num_plots=2):
    """
    Makes a plot which shows the Q table as a heatmap and the optimal trajectory
    given the heatmap
    """
    
    if show_multiple: 
        fig, axs = plt.subplots(num_plots, num_plots)
        axs = axs.flatten()
    else: 
        fig, ax = plt.subplots(1, 1)
        axs = (ax, )
    
    for ax in axs:
        
        ax.imshow(np.max(Q, axis=-1))
        steps = 0
        state = env.reset()
        done = False
        while not done and steps < max_steps:
            
            action = select_greedy(Q, state)
            new_state, reward, done, info = env.step(action)
            
            if not done:
                make_arrow(ax, state, new_state) 
            
            state = new_state
            steps += 1
        
    if title:
        fig.suptitle(title)    
    
    return fig

