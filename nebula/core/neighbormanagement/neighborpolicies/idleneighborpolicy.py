from nebula.core.neighbormanagement.neighborpolicies.neighborpolicy import NeighborPolicy

class IDLENeighborPolicy(NeighborPolicy):

    def __init__(self):
        pass

    def set_config(self, config):
        pass
    
    def need_more_neighbors(self):
        return False

    def accept_connection(self, source, joining=False):
        return False
    
    def get_actions(self):
        return [[],[]]

    def meet_node(self, node):
        pass
    
    def forget_nodes(self, node, forget_all=False):
        pass
    
    def get_nodes_known(self, neighbors_too=False):
        return Set()
    
    def update_neighbors(self, node, remove=False):
        pass