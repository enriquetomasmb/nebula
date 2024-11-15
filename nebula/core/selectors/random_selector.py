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
    MAX_PERCENT_SELECTABLE_NEIGHBORS = 0.7

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        logging.info("[RandomSelector] Initialized")

    async def node_selection(self, node):
        neighbors = self.neighbors_list.copy()
        if len(neighbors) == 0:
            logging.error(
                "[RandomSelector] Trying to select neighbors when there are no neighbors - aggregating itself only"
            )
            self.selected_nodes = [node.addr]
            return self.selected_nodes
        logging.info(f"[RandomSelector] available neighbors: {neighbors}")

        # needed to remove fixed seeds
        # import time
        # np.random.seed(int(str(time.time_ns())[-8:]))

        # Calculation of the amount of selected Neighbors according to thesis:
        # num_selected = max(
        #    self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS,
        #    math.floor(len(neighbors) * self.MAX_PERCENT_SELECTABLE_NEIGHBORS)
        # )
        # Improved way to calculate the amount of selected nodes (randomly distributed, the original implementation
        # would always select the maximum possible amount of nodes)
        max_selectable = math.floor(len(neighbors) * self.MAX_PERCENT_SELECTABLE_NEIGHBORS)
        num_selected = np.random.randint(
            self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS, max(max_selectable, self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS) + 1
        )

        selected_nodes = np.random.choice(neighbors, num_selected, replace=False).tolist()
        self.selected_nodes = selected_nodes + [node.addr]
        logging.info(f"[RandomSelector] selection finished, selected_nodes: {self.selected_nodes}")
        return self.selected_nodes
