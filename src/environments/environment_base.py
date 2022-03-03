from abc import ABC, abstractclassmethod, abstractmethod

class EnvironmentBase(ABC):
    
    @abstractmethod
    def take_action(self, action):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_prev_state(self):
        pass
    
    @abstractmethod
    def get_disct(self):
        pass
    