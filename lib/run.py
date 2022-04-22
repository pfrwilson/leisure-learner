
import matplotlib.pyplot as plt
from .utils import make_trajectory_map
import numpy as np


def plot_Q(Q, title, vmin, vmax):
    
    plt.figure()
    plt.imshow(Q.max(axis=-1), vmin=vmin, vmax=vmax, cmap='jet')
    plt.title(title)
    plt.colorbar()


def plot_F(F, title, vmin, vmax, action='max'): 
    plt.figure()
    if action == 'max':
        plt.imshow(F.max(axis=-1), vmin=vmin, vmax=vmax)
    if action == 'min':
        plt.imshow(F.min(axis=-1), vmin=vmin, vmax=vmax)
    else:
        plt.imshow(F[..., action], vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar    
  
    
def plot_errorbars(values, label):
    values = np.stack(values, axis=0)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    plt.plot(mean, label=label)
    plt.fill_between(np.arange(mean.shape[0]), mean+std, mean-std, alpha=0.5)
    plt.xlabel('number of timesteps')
    plt.ylabel('cumulative rewards')
    
    
def run(config):
    
    from .gym_windy_gridworld import WindyGridworld
    from .algorithms import Q_learn, Q_learn_freetime, build_q_table
    
    INITIALIZATIONS = config.initializations
    
    print('Configuring environment...')
    env = WindyGridworld(
        height=config.env.height, 
        width=config.env.width, 
        rewards=list(config.env.rewards), 
        wind=config.env.wind, 
        start=config.env.start, 
        allowed_actions=list(config.env.allowed_actions), 
        reward_terminates_episode=config.env.reward_terminates_episode
    )
    
    # BASELINE EXPERIMENT
    
    print('Running baseline...')
    
    results_baseline = {
        initialization: [] for initialization in INITIALIZATIONS
    }
    
    Q_tables = {}
    
    for initialization in INITIALIZATIONS:
        
        for run in range(config.num_runs):
            Q = build_q_table(
                (env.height, env.width),                
                env.action_space.n, 
                initialization = initialization,
                seed = config.random_initialization_seed # type: ignore
            )
            
            Q, rewards = Q_learn(
                env, 
                Q, 
                config.baseline.num_steps, 
                config.baseline.epsilon, 
                config.baseline.discount, 
                config.baseline.alpha
            )
            
            results_baseline[initialization].append(rewards)
            Q_tables[initialization] = Q
            
    if config.baseline.show_trajectory:
        for initialization, Q in Q_tables.items():
            make_trajectory_map(Q, env, title=f'Init {initialization} baseline trajectory', 
                                num_plots=config.trajectory_maps.num_plots)
    
    if config.baseline.show_rewards:
        plt.figure()
        for title, rewards in results_baseline.items():
            plot_errorbars(rewards, label=title)
            plt.legend()
            plt.title('baseline results')
    
    if config.baseline.show_q: 
        for title, Q in Q_tables.items():
            plot_Q(Q, f'{title} init q-table baseline', config.q_plots.vmin, config.q_plots.vmax)
 
    print('Running freetime')
    
    results_freetime = {
        initialization: [] for initialization in INITIALIZATIONS
    }
    Q_tables = {}
    F_tables = {}
    
    for initialization in INITIALIZATIONS:
        
        for run in range(config.num_runs):
            Q = build_q_table(
                (env.height, env.width),                
                env.action_space.n, 
                initialization = initialization, 
                seed = config.random_initialization_seed # type: ignore
            )
            
            Q, F, rewards, _ = Q_learn_freetime(
                env, 
                Q, 
                config.freetime.num_steps, 
                config.freetime.epsilon, 
                config.freetime.discount, 
                config.freetime.alpha, 
                config.freetime.alpha_f, 
                config.freetime.tolerance
            )
            
            results_freetime[initialization].append(rewards)
            Q_tables[initialization] = Q
            F_tables[initialization] = F
        
    if config.freetime.show_trajectory: 
        for initialization, Q in Q_tables.items():
            make_trajectory_map(Q, env, title=f'Init {initialization} freetime trajectory', 
                                num_plots=config.trajectory_maps.num_plots)
    
    if config.freetime.show_rewards: 
        plt.figure()
        for title, rewards in results_freetime.items():
            plot_errorbars(rewards, label=title)
            plt.legend()
            plt.title('freetime results_freetime')
        
    if config.freetime.show_q:
        for title, Q in Q_tables.items():
            plot_Q(Q, f'{title} init q-table freetime', config.q_plots.vmin, config.q_plots.vmax)
        
    if config.freetime.show_f:
        for action in config.freetime.show_f_actions:
            for title, F in F_tables.items():
                plot_F(F, f'{title} init f-table freetime action {action}', config.f_plots.vmin, config.f_plots.vmax, 
                       action)
        
    if config.plot_freetime_vs_baseline_same_table:
        for initialization in INITIALIZATIONS:
            plt.figure()
            plot_errorbars(results_baseline[initialization], label='baseline')
            plot_errorbars(results_freetime[initialization], label='freetime')
            plt.title(f'{initialization} initialization')
            plt.legend()
            
    plt.show()
    