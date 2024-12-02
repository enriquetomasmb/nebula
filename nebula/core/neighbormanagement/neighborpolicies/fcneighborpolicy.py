from nebula.core.neighbormanagement.neighborpolicies.neighborpolicy import NeighborPolicy
from nebula.core.utils.locker import Locker

class FCNeighborPolicy(NeighborPolicy):
    
    def __init__(self):
        self.max_neighbors = None
        self.nodes_known = set()
        self.neighbors = set()
        self.neighbors_lock = Locker(name="neighbors_lock")
        self.nodes_known_lock = Locker(name="nodes_known_lock")
        
    def need_more_neighbors(self):
        """
            Fully connected network requires to be connected to all devices, therefore,
            if there are more nodes known that self.neighbors, more neighbors are required
        """
        self.neighbors_lock.acquire()
        need_more = (len(self.neighbors) < len(self.nodes_known))
        self.neighbors_lock.release()
        return need_more
    
    def set_config(self, config):
        """
        Args:
            config[0] -> list of self neighbors
            config[1] -> list of nodes known on federation
        """
        self.neighbors_lock.acquire()
        self.neighbors = config[0] 
        self.neighbors_lock.release()
        for addr in config[1]:
                self.nodes_known.add(addr)
            
    def accept_connection(self, source, joining=False):
        """
            return true if connection is accepted
        """
        self.neighbors_lock.acquire()
        ac = not source in self.neighbors
        self.neighbors_lock.release()
        return ac
    
    def meet_node(self, node):
        """
            Update the list of nodes known on federation
        """
        self.nodes_known_lock.acquire()
        self.nodes_known.add(node)
        self.nodes_known_lock.release()
        
    def get_nodes_known(self, neighbors_too=False):
        self.nodes_known_lock.acquire()
        nk = self.nodes_known.copy()
        if not neighbors_too:
            self.neighbors_lock.acquire()
            nk = self.nodes_known - self.neighbors
            self.neighbors_lock.release()
        self.nodes_known_lock.release()
        return nk     
    
    def forget_nodes(self, node, forget_all=False):
        self.nodes_known_lock.acquire()
        if forget_all:
            self.nodes_known.clear()
        else:
            self.nodes_known.discard(node)
        self.nodes_known_lock.release()
        
    def get_actions(self): 
        """
            return list of actions to do in response to connection
                - First list represents addrs argument to LinkMessage to connect to
                - Second one represents the same but for disconnect from LinkMessage
        """ 
        return [self._connect_to(), self._disconnect_from()]
          
    
    def _disconnect_from(self):
        return ""
    
    def _connect_to(self):
        ct = ""
        self.neighbors_lock.acquire()
        ct = " ".join(self.neighbors)
        self.neighbors_lock.release()
        return ct
    
    def update_neighbors(self, node, remove=False):
        self.neighbors_lock.acquire()
        if remove:
            self.neighbors.remove(node)
        else:
            self.neighbors.add(node)
        self.neighbors_lock.release() 