from nebula.core.neighbormanagement.candidateselection.candidateselector import CandidateSelector
from nebula.core.utils.locker import Locker

class HETCandidateSelector(CandidateSelector):
    
    def __init__(self):
        self.candidates = []
        self.loss, self.weight_distance, self.weight_hetereogeneity = (0, 0.2, 0.8)
        self.candidates_lock = Locker(name="candidates_lock")
        
    def set_config(self, config):
        """
        Args:
            config contains values to evaluate suitability of candidades
        """
        self.loss, self.weight_distance, self.weight_hetereogeneity = config    
    
    def add_candidate(self, candidate):
        """
        Args:
            candidate is compound of three data:
                - candidate.addr
                - candidate number of neighbors
                - candidate current model loss
        """
        addr, n_neighbors, loss = candidate
        hv = self.__calculate_hetereogeneity(loss)
        self.candidates_lock.acquire()
        self.candidates.append((addr, n_neighbors, hv))
        self.candidates_lock.release()
      
    def select_candidates(self):
        """
            Calculate suitability of candidates and sort them, then return
            the average number of candidates calculated using info from federation nodes

        Returns:
            best 'n' candidates
        """
        self.candidates_lock.acquire()
        bc = self.__suitability_function()
        n = self.__calculate_ideal_neighbors()
        self.candidates_lock.release()
        n = n if n > 0 else len(bc)
        return [addr for addr, _ in bc[:n]]
    
    def remove_candidates(self):
        self.candidates_lock.acquire()
        self.candidates = []
        self.candidates_lock.release()
    
    def any_candidate(self):
        self.candidates_lock.acquire()
        any = True if len(self.candidates) > 0 else False
        self.candidates_lock.release()
        return any
    
    #TODO hay q descontar los vecinos propios ya establecidos
    def __calculate_ideal_neighbors(self):
        """
        Returns:
            Average number of neighbors in candidate nodes
        """
        average_neighbors = 0
        if(len(self.candidates)):
            n_neighbors = [ pn[1] for pn in self.candidates]
            average_neighbors = sum(n_neighbors) / len(n_neighbors) if n_neighbors else 0
        return average_neighbors
        
    def __calculate_hetereogeneity(self, loss):
        """
            Calculate dataset heterogeneity between self.dataset and candidate.dataset using current loss value,
            assuming the models are close enough to show good results
        """
        if self.loss < 0 or loss < 0:
            return 0
        else:
            return abs((self.loss-loss))

    def __suitability_function(self):
        """
            Calculate suitability using hetereogeneity value and position on candidate.list. The reason to use that position is
            because we assume that better candidates in terms of distance/quality would be in first positions of the list, and slower
            or worse connection ones would be at the end
        """
    
        best_candidates = []
        total_positions = len(self.candidates)
        
        # lower positions in list represents higher value of Distance/Quality of connection
        def calculate_position_weight(position):
            return (total_positions - position - 1) / (total_positions - 1)
        
        # MAX and MIN value of hetereogeneity to normalize
        min_hv = min(self.candidates, key=lambda x: x[2])[2]
        max_hv = max(self.candidates, key=lambda x: x[2])[2]
        
        # Smaller values of HET represents higher suitability values
        def normalize_hv(hv):
            return (max_hv - hv) / (max_hv - min_hv) if max_hv != min_hv else 0.5
        
        for position, (addr, n, hv) in enumerate(self.candidates):
            position_weight = calculate_position_weight(position)
            normalized_hv = normalize_hv(hv) 
            suitability = self.weight_distance * position_weight + self.weight_hetereogeneity * normalized_hv  # suitability of the node
            best_candidates.append((addr, suitability))
        
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return best_candidates