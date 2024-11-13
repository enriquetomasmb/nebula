import logging
import math

import numpy as np

from nebula.core.selectors.selector import Selector
from nebula.core.utils.helper import cosine_metric

class DistanceSelector(Selector):
    """
    Selects neighbors based on model distance
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
        logging.info("[DistanceSelector] Initialized")

    def node_selection(self, node):
        #if self.selected_nodes != []:
        #    return self.selected_nodes
        
        threshold = float(node.node_selection_strategy_parameter)
        neighbors = self.neighbors_list.copy()
        
        if len(neighbors) == 0:
            logging.error(
                "[DistanceSelector] Trying to select neighbors when there are no neighbors - aggregating itself only"
            )
            self.selected_nodes = [node.addr]
            return self.selected_nodes
        logging.info(f"[DistanceSelector] available neighbors: {neighbors}")

        distances = {}

        pending_models = node.aggregator.get_pending_models_to_aggregate()

        local_model = pending_models[node.addr][0]

        for device in pending_models:
            if device!=node.addr:
                neighbor_model=pending_models[device][0]
                neighbor_distance = cosine_metric(local_model, neighbor_model, similarity=True)
                distances[device]=neighbor_distance

        for neighbor in distances:
            #logging.info(f"[DistanceSelector] processed_node: {neighbor}, distance: {distances[neighbor]}")
            if distances[neighbor] >= threshold:
                logging.info(f"[DistanceSelector] selection, selected_node: {neighbor}, distance: {distances[neighbor]}")
                self.selected_nodes.append(neighbor)
            
        """
        max_selectable = math.floor(len(neighbors) * self.MAX_PERCENT_SELECTABLE_NEIGHBORS)
        num_selected = np.random.randint(
            self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS, max(max_selectable, self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS) + 1
        )

        selected_nodes = np.random.choice(neighbors, num_selected, replace=False).tolist()"""

        self.selected_nodes = self.selected_nodes + [node.addr]
        logging.info(f"[DistanceSelector] selection finished, selected_nodes: {self.selected_nodes}")
        return self.selected_nodes
