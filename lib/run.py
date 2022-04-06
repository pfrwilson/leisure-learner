
import matplotlib.pyplot as plt

INITIALIZATIONS = ['pessimistic', 'optimistic', 'random']

def plot_Q(Q, title, vmin, vmax):
    
    plt.figure()
    plt.imshow(Q.max(axis=-1), vmin=vmin, vmax=vmax, cmap='jet')
    plt.title(title)
    plt.colorbar()


def run(config):
    
    from .gym_windy_gridworld import WindyGridworld
    from .algorithms import Q_learn, Q_learn_freetime, build_q_table
    
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
    
    results = {}
    Q_tables = {}
    
    for initialization in INITIALIZATIONS:
        Q = build_q_table(
            (env.height, env.width),                
            env.action_space.n, 
            initialization = initialization         # type: ignore
        )
        
        Q, rewards = Q_learn(
            env, 
            Q, 
            config.baseline.num_steps, 
            config.baseline.epsilon, 
            config.baseline.discount, 
            config.baseline.alpha
        )
        
        results[initialization] = rewards
        Q_tables[initialization] = Q
    
    plt.figure()
    for title, rewards in results.items():
        plt.plot(rewards, label=title)
        plt.legend()
    
    for title, Q in Q_tables.items():
        plot_Q(Q, title, config.q_plots.vmin, config.q_plots.vmax)
        
    plt.show()
    
    print('Running freetime')
    
    results = {}
    Q_tables = {}
    F_tables = {}
    
    for initialization in INITIALIZATIONS:
        Q = build_q_table(
            (env.height, env.width),                
            env.action_space.n, 
            initialization = initialization         # type: ignore
        )
        
        Q, F, rewards, _ = Q_learn_freetime(
            env, 
            Q, 
            config.freetime.num_steps, 
            config.freetime.epsilon, 
            config.freetime.discount, 
            config.freetime.alpha, 
            config.freetime.alpha_f
        )
        
        results[initialization] = rewards
        Q_tables[initialization] = Q
        F_tables[initialization] = F
    
    plt.figure()
    for title, rewards in results.items():
        plt.plot(rewards, label='title')
        plt.legend()
    
    for title, Q in Q_tables.items():
        plot_Q(Q, title, config.q_plots.vmin, config.q_plots.vmax)
        
    for title, F in F_tables.items():
        plot_Q(F, f'{title}_freetime', config.f_plots.vmin, config.f_plots.vmax)
        
    plt.show()
    