
from tqdm import tqdm
from ..agents.agent import Agent
from time import sleep
from dataclasses import dataclass
from ..logging.logger_base import Logger

class Trainer:

    def __init__(self, max_episodes=None):
        self.max_episodes = int(max_episodes)

    def train(self, agent: Agent):
    
        pbar = tqdm(range(self.max_episodes), desc='EPISODE')
        for episode in pbar:
            agent.learn_episode()
            