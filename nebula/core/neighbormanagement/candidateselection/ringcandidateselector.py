from nebula.core.neighbormanagement.candidateselection.candidateselector import CandidateSelector
from nebula.core.utils.locker import Locker

class RINGCandidateSelector(CandidateSelector):

    def __init__(self):
        self.candidates = []
        self.candidates_lock = Locker(name="candidates_lock")
        
    def set_config(self, config):
        pass    
    
    def add_candidate(self, candidate):
        """
            To avoid topology problems select 1st candidate found
        """
        self.candidates_lock.acquire()
        if len(self.candidates) == 0:
            self.candidates.append(candidate)
        self.candidates_lock.release()
      
    def select_candidates(self):
        self.candidates_lock.acquire()
        cdts = self.candidates.copy()
        self.candidates_lock.release()
        return cdts
    
    def remove_candidates(self):
        self.candidates_lock.acquire()
        self.candidates = []
        self.candidates_lock.release()

    def any_candidate(self):
        self.candidates_lock.acquire()
        any = True if len(self.candidates) > 0 else False
        self.candidates_lock.release()
        return any