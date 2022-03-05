
from tqdm import tqdm
from ..agents.agent import Agent
from time import sleep
from dataclasses import dataclass
from ..logging.logger_base import Logger

class Trainer:

    def __init__(self, max_steps=None):
        self.max_steps = max_steps

    def train(self, agent: Agent):
    
        pbar = tqdm()
        pbar.set_description
        
        while agent.global_step <= self.max_steps:
            agent.learn_episode()
            pbar.update()