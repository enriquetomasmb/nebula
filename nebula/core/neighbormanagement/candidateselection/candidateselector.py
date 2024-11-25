from abc import ABC, abstractmethod
from typing import Type

class CandidateSelector(ABC):
    
    @abstractmethod 
    def set_config(self, config):
        pass
    
    @abstractmethod 
    def add_candidate(self, candidate):
        pass
    
    @abstractmethod 
    def select_candidates(self):
        pass
    
    @abstractmethod 
    def remove_candidates(self):
        pass
    
    @abstractmethod 
    def any_candidate(self):
        pass
    
def factory_CandidateSelector(topology) -> CandidateSelector:
    from nebula.core.neighbormanagement.candidateselection.fccandidateselector import FCCandidateSelector
    from nebula.core.neighbormanagement.candidateselection.hetcandidateselector import HETCandidateSelector
    from nebula.core.neighbormanagement.candidateselection.ringcandidateselector import RINGCandidateSelector
    
    options = {
        "ring": RINGCandidateSelector,
        "fully": FCCandidateSelector,
        "random": HETCandidateSelector,  
    } 
    
    cs = options.get(topology, FCCandidateSelector)
    return cs() 