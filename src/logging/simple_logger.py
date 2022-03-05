from .logger_base import Logger

class ListLogger(Logger):
    
    def __init__(self):
        self.records = {}
    
    def log_dict(self, d: dict):
        for name, value in d.items():
            self.log(name, value)
        
    def log(self, name: str, obj):
        if name in self.records.keys():
            self.records[name].append(obj)
        else:
            self.records[name] = [obj]
        
