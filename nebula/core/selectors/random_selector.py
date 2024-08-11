import logging
import math

import numpy as np

from nebula.core.selectors.selector import Selector


class RandomSelector(Selector):
    """
    Selects neighbors randomly
    MIN_AMOUNT_OF_SELECTED_NEIGHBORS: Defines the minimum amount of nodes that
        needs to be selected for proper functioning
    MAX_PERCENT_SELECTABLE_NEIGHBORS: Defines the maximum amount of nodes that
        can be selected as percentage of the total amount of neighbors
    """
    MIN_AMOUNT_OF_SELECTED_NEIGHBORS = 1
    MAX_PERCENT_SELECTABLE_NEIGHBORS = 0.8

    def __init__(self, config = None):
        super().__init__(config)
        self.config = config
        logging.info("[RandomSelector] Initialized")

    def node_selection(self, node):
        neighbors = self.neighbors_list.copy()
        if len(neighbors) == 0:
            logging.error(
                "[RandomSelector] Trying to select neighbors when there are no neighbors - aggregating itself only"
            )
            self.selected_nodes = [node.addr]
            return self.selected_nodes
        logging.info(f"[RandomSelector] available neighbors: {neighbors}")
        num_selected = max(
            self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS,
            math.floor(len(neighbors) * self.MAX_PERCENT_SELECTABLE_NEIGHBORS)
        )
        selected_nodes = np.random.choice(neighbors, num_selected, replace = False).tolist()
        self.selected_nodes = selected_nodes + [node.addr]
        logging.info(f"[RandomSelector] selection finished, selected_nodes: {self.selected_nodes}")
        return self.selected_nodes
