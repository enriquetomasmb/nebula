import logging
import math
from statistics import mean, stdev

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
        self.stop_training=False
        self.already_activated=False
        self.final_list=False
        logging.info("[DistanceSelector] Initialized")

    def node_selection(self, node):

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

        distance_values = distances.values()
        avg_distance = mean(distance_values)
        std_dev_distance = stdev(distance_values)

        logging.info(f"[DistanceSelector] average: {avg_distance}, stddev: {std_dev_distance}")

        lower_bound = avg_distance - threshold*std_dev_distance
        upper_bound = avg_distance + threshold*std_dev_distance

        if mean(distances.values()) < 0.95 and node.round < int(node.total_rounds*0.2):
            self.selected_nodes=self.neighbors_list + [node.addr]
        
        elif not self.already_activated:
            logging.info(f"[DistanceSelector] DetectorSelector stop training activated")
            self.stop_training=True
            self.selected_nodes=self.neighbors_list + [node.addr]
        
        elif not self.final_list:
            self.selected_nodes=[]
            for neighbor in distances:
                if avg_distance <= distances[neighbor]:
                    logging.info(f"[DistanceSelector] selected_node: {neighbor}, distance: {distances[neighbor]}")
                    self.selected_nodes.append(neighbor)
                else:
                    logging.info(f"[DistanceSelector] NOT selected_node: {neighbor}, distance: {distances[neighbor]}")
            self.selected_nodes = self.selected_nodes + [node.addr]
            self.final_list=True

        logging.info(f"[DistanceSelector] selection finished, selected_nodes: {self.selected_nodes}")
        return self.selected_nodes
