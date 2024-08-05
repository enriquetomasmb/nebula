import logging
import math
from collections import namedtuple

import numpy as np

from nebula.core.selectors.selector import Selector
from sklearn.preprocessing import normalize


class PrioritySelector(Selector):
    """
    PrioritySelector evaluates nodes based on a set of features
    (CPU Usage, data size, bytes sent/received, loss and latency)
    These features are calculated into a score which then defines
    the probability of this node being selected for aggregation.
    MIN_AMOUNT_OF_SELECTED_NEIGHBORS: Defines the minimum amount of nodes that
        needs to be selected for proper functioning
    MAX_PERCENT_SELECTABLE_NEIGHBORS: Defines the maximum amount of nodes that
        can be selected as percentage of the total amount of neighbors
    FEATURE_WEIGHTS: Defines the weight of the features in the following order:
        [loss, cpu_percent, data_size, bytes_received, bytes_sent, latency, age]
    """
    MIN_AMOUNT_OF_SELECTED_NEIGHBORS = 1
    MAX_PERCENT_SELECTABLE_NEIGHBORS = 0.8
    FEATURE_WEIGHTS = [10.0, 1.0, 1.0, 0.5, 0.5, 10.0, 3.0]

    def __init__(self, config = None):
        super().__init__(config)
        self.config = config
        FeatureWeights = namedtuple(
            'FeatureWeights',
            ['loss', 'cpu_percent', 'data_size', 'bytes_received', 'bytes_sent', 'latency', 'age']
        )
        self.feature_weights = FeatureWeights(*self.FEATURE_WEIGHTS)
        logging.info("[PrioritySelector] Initialized")

    def node_selection(self, node):
        neighbors = self.neighbors_list.copy()

        if len(neighbors) == 0:
            logging.error(
                "[PrioritySelector] Trying to select neighbors when there are no neighbors"
            )
            return node

        num_selected = max(
            self.MIN_AMOUNT_OF_SELECTED_NEIGHBORS,
            math.floor(len(neighbors) * self.MAX_PERCENT_SELECTABLE_NEIGHBORS)
        )

        availability = []
        feature_array = np.empty((7, 0))

        for node in neighbors:
            feature_list = list((self.features[node]["loss"],
                                 self.features[node]["cpu_percent"],
                                 self.features[node]["data_size"],
                                 self.features[node]["bytes_received"],
                                 self.features[node]["bytes_sent"],
                                 self.features[node]["latency"],
                                 self.ages[node]))

            # Set loss to 100 if loss metric is unavailable
            if feature_list[0] == -1:
                feature_list[0] = 100

            logging.info(f"[PrioritySelector] Features for node {node}: {feature_list}")

            availability.append(self.features[node]["availability"])

            feature = np.array(feature_list).reshape(-1, 1).astype(np.float64)
            feature_array = np.append(feature_array, feature, axis = 1)

        logging.info(f"[PrioritySelector] Features: {feature_array}")

        # Prevent 0 denominator:
        if self.feature_weights.cpu_percent == 0:
            self.feature_weights.cpu_percent = 0.01
        if self.feature_weights.latency == 0:
            self.feature_weights.latency = 0.01

        # Invert the weights of cpu_percent & latency
        self.feature_weights.cpu_percent = 1 / self.feature_weights.cpu_percent
        self.feature_weights.latency = 1 / self.feature_weights.latency

        # Normalized features
        feature_array_normed = normalize(feature_array, axis = 1, norm = 'l1')

        # Add weight to features
        weight = np.array(self.FEATURE_WEIGHTS).reshape(-1, 1)
        feature_array_weighted = np.multiply(feature_array_normed, weight)

        logging.info(f"[PrioritySelector] Features weighted: {feature_array_weighted}")

        # Before availability
        scores = np.sum(feature_array_weighted, axis = 0)

        # Add availability
        final_scores = np.multiply(scores, np.array(availability))

        # Probability selection
        p = normalize([final_scores], axis = 1, norm = 'l1')

        selected_nodes = np.random.choice(
            neighbors, num_selected, replace = False, p = p[0]).tolist()

        # Update ages
        for node in neighbors:
            if node not in selected_nodes:
                self.ages[node] = self.ages[node] + 2

        logging.info(f"[PrioritySelector] selection finished, selected_nodes: {selected_nodes}")

        return selected_nodes
