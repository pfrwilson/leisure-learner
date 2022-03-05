
from abc import ABC, abstractmethod
from numpy import ndarray


class Logger(ABC):
    
    @abstractmethod
    def log_dict(self, d: dict):
        pass
    
    @abstractmethod
    def log(self, name, value):
        pass
    