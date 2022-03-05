
import imp
import hydra
from src.agents.q_learn import QLearningConfig
import numpy as np
from einops import rearrange

from src.environments import build_env

@hydra.main(config_path='configs/config.yaml')
def main(config):

    env = build_env(config.environment.name, config.environment.config)
    
    from src.logging.simple_logger import ListLogger
    logger = ListLogger()
    
    # TODO factory method here instead of constructor
    from src.agents.q_learn import QLearningAgent
    agent = QLearningAgent(env, QLearningConfig(**config.agent), logger)

    from src.training.trainer import Trainer
    trainer = Trainer(max_steps=config.training.n_steps)
    trainer.train(agent)

    create_report(agent, env, logger)

# TODO move to a reporting module
def create_report(agent, env, logger):
    rewards = logger.records['reward']
    rewards = np.array(rewards)
    cum_rewards = np.cumsum(rewards)
    import matplotlib.pyplot as plt
    plt.plot(cum_rewards)
    
    Q = agent.Q
    max_Q = np.max(Q, axis=1)
    max_Q = rearrange(max_Q, '( w h ) -> h w', h=env.height, w=env.width)
    plt.figure()
    plt.imshow(max_Q)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
    
    