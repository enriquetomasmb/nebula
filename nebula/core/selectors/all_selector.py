import logging

from nebula.core.selectors.selector import Selector


class AllSelector(Selector):
    """
    Selects all neighbors
    """
    def __init__(self, config = None):
        super().__init__(config)
        self.config = config
        logging.info("[AllSelector] Initialized")

    def node_selection(self, node):
        neighbors = self.neighbors_list.copy()
        logging.info(f"[AllSelector] available neighbors: {neighbors}")
        if len(neighbors) == 0:
            logging.error(
                "[AllSelector] Trying to select neighbors when there are no neighbors"
            )
            return None
        selected_nodes = neighbors
        logging.info(f"[AllSelector] selection finished - selected_nodes: {selected_nodes}")
        return selected_nodes
