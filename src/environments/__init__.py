
from tkinter import Grid
from omegaconf import DictConfig
from typing import Optional
from .experimental.gridworld import GridWorld

PREMADES = {
    'gridworld_default': GridWorld(20, 10, [[1, 0, 5]])
}

def build_env(name: str, config: Optional[DictConfig] = None):
    if name == 'gridworld':
        return build_gridworld(config)
    elif name in PREMADES.keys():
        return PREMADES[name]
    
def build_gridworld(config: DictConfig):
    return GridWorld(**config)
    