from abc import ABC, abstractmethod

class ExternalConnectionService(ABC):

    @abstractmethod 
    def start(self):
        pass
    
    @abstractmethod 
    def stop(self):
        pass
    
    @abstractmethod 
    def find_federation(self):
        pass