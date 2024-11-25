from nebula.core.neighbormanagement.neighborpolicies.neighborpolicy import NeighborPolicy
from nebula.core.utils.locker import Locker
import random

class RINGNeighborPolicy(NeighborPolicy):
        
    def __init__(self):
        self.max_neighbors = 2
        self.nodes_known = set()
        self.neighbors = set()
        self.neighbors_lock = Locker(name="neighbors_lock")
        self.nodes_known_lock = Locker(name="nodes_known_lock")
        self.addr = ""
        
    def need_more_neighbors(self):
        self.neighbors_lock.acquire()
        need_more = len(self.neighbors) < self.max_neighbors
        self.neighbors_lock.release()
        return need_more
    
    def set_config(self, config):
        """
        Args:
            config[0] -> list of self neighbors
            config[1] -> list of nodes known on federation
            config[2] -> self.addr
        """
        self.neighbors_lock.acquire()
        self.neighbors = config[0]
        self.neighbors_lock.release()
        for addr in config[1]:
                self.nodes_known.add(addr)
        self.addr = config[2]
            
    def accept_connection(self, source, joining=False):
        """
            return true if connection is accepted
        """
        ac = False
        self.neighbors_lock.acquire()
        if not joining:    
            ac = not source in self.neighbors
        else:
            ac = not len(self.neighbors) == self.max_neighbors
        self.neighbors_lock.release()
        return ac
    
    def meet_node(self, node):
        self.nodes_known_lock.acquire()
        self.nodes_known.add(node)
        self.nodes_known_lock.release()
        
    def forget_nodes(self, node, forget_all=False):
        self.nodes_known_lock.acquire()
        if forget_all:
            self.nodes_known.clear()
        else:
            self.nodes_known.discard(node)
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
        
    def get_actions(self): 
        """
            return list of actions to do in response to connection
                - First list represents addrs argument to LinkMessage to connect to
                - Second one represents the same but for disconnect from LinkMessage
        """ 
        self.neighbors_lock.acquire()
        ct_actions = []
        df_actions = []
        if len(self.neighbors) < self.max_neighbors:
            list_neighbors = list(self.neighbors)
            index = random.randint(0, len(list_neighbors)-1)
            node = list_neighbors[index]
            ct_actions.append(node)                            # connect to
            df_actions.append(self.addr)                       # disconnect from
        self.neighbors_lock.release()
        return [ct_actions, df_actions]
    
    def update_neighbors(self, node, remove=False):
        self.neighbors_lock.acquire()
        if remove:
            self.neighbors.remove(node)
        else:
            self.neighbors.add(node)
        self.neighbors_lock.release() 