import logging
from collections import namedtuple

import numpy as np
from sklearn.preprocessing import normalize

from nebula.addons.functions import print_msg_box
from nebula.core.selectors.selector import Selector


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
    # Original Feature Weights provided in Report / Thesis
    # please change the weights if you want emphsize the impantance of the features
    FEATURE_WEIGHTS = [70.0, 5.0, 1.0, 0.0, 80.0, 2.0, 1.0, 10.0]
    # Feature Weights for Testing (Latency can be changed reliably by virtual constraints)
    # FEATURE_WEIGHTS = [0, 0, 0, 0, 0, 100, 0]

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        FeatureWeights = namedtuple(
            "FeatureWeights",
            ["loss", "cpu_percent", "data_size", "bytes_received", "bytes_sent", "latency", "age", "sustainability"],
        )
        self.feature_weights = FeatureWeights(*self.FEATURE_WEIGHTS)
        logging.info("[PrioritySelector] Initialized")

    async def node_selection(self, node):
        neighbors = self.neighbors_list.copy()

        if len(neighbors) == 0:
            logging.error(
                "[PrioritySelector] Trying to select neighbors when there are no neighbors - aggregating itself only"
            )
            self.selected_nodes = [node.addr]
            return self.selected_nodes

        availability = []
        # add the sustainability metric
        feature_array = np.empty((8, 0))

        for neighbor in neighbors:
            if neighbor not in self.ages.keys():
                self.ages[neighbor] = 1

            # Invert CPU Percent/Latency, 0.000001 is added to avoid division by zero
            feature_list = list((
                self.features[neighbor]["loss"],
                1 / (self.features[neighbor]["cpu_percent"] + 0.000001),
                self.features[neighbor]["data_size"],
                self.features[neighbor]["bytes_received"],
                self.features[neighbor]["bytes_sent"],
                1 / (self.features[neighbor]["latency"] + 0.000001),
                self.ages[neighbor],
                self.features[neighbor]["sustainability"],
            ))

            # Set loss to 100 if loss metric is unavailable
            if feature_list[0] == -1:
                feature_list[0] = 100

            logging.info(f"[PrioritySelector] Features for node {neighbor}: {feature_list}")

            availability.append(self.features[neighbor]["availability"])

            feature = np.array(feature_list).reshape(-1, 1).astype(np.float64)
            feature_array = np.append(feature_array, feature, axis=1)

        # Normalized features
        feature_array_normed = normalize(feature_array, axis=1, norm="l1")

        # Add weight to features
        weight = np.array(self.FEATURE_WEIGHTS).reshape(-1, 1)
        feature_array_weighted = np.multiply(feature_array_normed, weight)

        # Compute scores
        scores = np.sum(feature_array_weighted, axis=0)

        print_msg_box(msg=f"Scores: {dict(zip(neighbors, scores, strict=False))}", title="Final NSS Scores")

        # Calculate mean and standard deviation
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # # Identify non-outlier nodes
        # non_outlier_indices = [i for i, score in enumerate(scores) if abs(score - mean_score) <= std_score]
        # logging.info("[PrioritySelector] Non-outlier nodes: {}".format([neighbors[i] for i in non_outlier_indices]))

        # # Ensure at least num_selected nodes are selected
        # if len(non_outlier_indices) < num_selected:
        #     # Sort indices by absolute difference from mean
        #     sorted_indices = np.argsort([abs(scores[i] - mean_score) for i in range(len(scores))])
        #     selected_indices = sorted_indices[:num_selected]
        # else:
        #     selected_indices = non_outlier_indices[:num_selected]

        # Identify non-outlier nodes
        non_outlier_indices = [i for i, score in enumerate(scores) if abs(score - mean_score) <= std_score]
        selected_nodes = [neighbors[i] for i in non_outlier_indices]

        # Update ages
        for neighbor in neighbors:
            if neighbor not in selected_nodes:
                self.ages[neighbor] = self.ages[neighbor] + 2

        # Add own node
        self.selected_nodes = selected_nodes + [node.addr]

        logging.info(f"[PrioritySelector] selection finished, selected_nodes: {self.selected_nodes}")

        return self.selected_nodes
