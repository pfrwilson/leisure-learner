from .agent import Agent

def build_agent(name, env, logger, config):
    
    if name == 'q_learning':
        from .q_learn import QLearningAgent, QLearningConfig
        return QLearningAgent(
            env, QLearningConfig(**config), logger
        )
        
    elif name == 'q_learning_freetime': 
        from .q_learn import FreetimeQLearningAgent, FreetimeQLearningConfig
        return FreetimeQLearningAgent(
            env, 
            FreetimeQLearningConfig(**config), 
            logger
        )
        