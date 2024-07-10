from abc import ABC, abstractmethod

class ModelHandler(ABC):
    
    @abstractmethod 
    def set_config(self, config):
        pass
    
    @abstractmethod 
    def accept_model(self, model):
        pass
    
    @abstractmethod 
    def get_model(self, model):
        pass

    @abstractmethod
    def pre_process_model(self):
        pass
    
def factory_ModelHandler(model_handler):
    from nebula.core.neighbormanagement.modelhandlers.stdmodelhandler import STDModelHandler
    from nebula.core.neighbormanagement.modelhandlers.aggmodelhandler import AGGModelHandler
    
    options = {
        'std': STDModelHandler,
        "aggregator": AGGModelHandler
    } 
    
    cs = options.get(model_handler)
    return cs() 