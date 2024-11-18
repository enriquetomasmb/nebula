import logging
import random
from statistics import mean, stdev

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

    def __init__(self, config=None, voting=False, threshold=0):
        super().__init__(config)
        self.config = config
        self.stop_training = False
        self.already_activated = False
        self.final_list = False
        self.number_votes = 100000
        self.threshold = threshold
        self.rounds_without_training = 0
        self.voting_enabled = voting
        logging.info("[DistanceSelector] Initialized")

    def should_train(self):
        if not self.voting_enabled:
            return True

        logging.info(f"[DistanceSelector] Rounds without training: {self.rounds_without_training}")

        # Increase probability by 10% for each round without training
        probability = 0.1 + 0.1 * self.rounds_without_training
        round_train = random.random() < probability

        vote_train = self.number_votes > len(self.neighbors_list) * (0.3 - 0.1 * self.threshold)

        logging.info(f"[DistanceSelector] Train Vote : {vote_train}, Spontaneous: {round_train}")

        train = vote_train or round_train

        if train:
            self.rounds_without_training = 0
        else:
            self.rounds_without_training = self.rounds_without_training + 1

        return train

    def reset_votes(self):
        logging.info(f"[DistanceSelector] Reseting Votes {self.number_votes}")
        self.number_votes = 0

    def add_vote(self):
        self.number_votes = self.number_votes + 1

    async def node_selection(self, node):
        if self.final_list:
            # mandar voto
            message = node.cm.mm.generate_vote_message()
            await node.cm.send_message_to_neighbors(message, neighbors=self.selected_nodes)
            logging.info(f"[DistanceSelector] Sending Votes {self.selected_nodes}")
            return self.selected_nodes

        neighbors = self.neighbors_list.copy()  # node.cm.get_all_addrs_current_connections(only_direct=True)

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
            if device != node.addr:
                neighbor_model = pending_models[device][0]
                neighbor_distance = cosine_metric(local_model, neighbor_model, similarity=True)
                distances[device] = neighbor_distance

        distance_values = distances.values()
        avg_distance = mean(distance_values)
        std_dev_distance = stdev(distance_values)

        logging.info(f"[DistanceSelector] average: {avg_distance}, stddev: {std_dev_distance}")

        limit = avg_distance + self.threshold * std_dev_distance

        if mean(distances.values()) < 0.95 and node.round < int(node.total_rounds * 0.2) and not self.already_activated:
            self.selected_nodes = self.neighbors_list + [node.addr]

        elif not self.already_activated:
            logging.info("[DistanceSelector] DetectorSelector stop training activated")
            self.stop_training = True
            self.selected_nodes = self.neighbors_list + [node.addr]

        elif not self.final_list:
            self.number_votes = 0
            self.selected_nodes = []
            for neighbor in distances:
                if limit <= distances[neighbor]:
                    logging.info(f"[DistanceSelector] selected_node: {neighbor}, distance: {distances[neighbor]}")
                    self.selected_nodes.append(neighbor)
                else:
                    logging.info(f"[DistanceSelector] NOT selected_node: {neighbor}, distance: {distances[neighbor]}")
            self.selected_nodes = self.selected_nodes + [node.addr]
            self.final_list = True

        # mandar voto

        message = node.cm.mm.generate_vote_message()
        await node.cm.send_message_to_neighbors(message, neighbors=self.selected_nodes)

        logging.info(f"[DistanceSelector] selection finished, selected_nodes: {self.selected_nodes}")
        return self.selected_nodes
