import logging

from nebula.addons.functions import print_msg_box


class Selector:
    """
    Base class for all different Node Selection Strategy Selectors
    For more information see:
    https://files.ifi.uzh.ch/CSG/staff/feng/external/theses/MA-Chenfei-Ma.pdf
    Attributes:
        node_name (str): Name of the node
        config (dict): Configuration of the node
        neighbors_list (list): List of neighbors
        features (dict): Features of the neighbors
        ages (dict): Age of the neighbors
    """

    def __init__(self, config=None):
        self.config = config
        self.neighbors_list = []
        self.selected_nodes = []
        self.features = {}
        self.ages = {}

    def add_node_features(self, node, features):
        self.features[node] = features
        self.features[node]["availability"] = 1
        selector_received_nss_msg = (
            "Node: {}\n"
            "CPU Usage (%): {}%\n"
            "Bytes Sent: {}\n"
            "Bytes Received: {}\n"
            "Loss: {}\n"
            "Data Size: {}\n"
            "Latency (ms): {}\n"
            "Availability: {}\n"
            "Sustainability: {}"
        )
        print_msg_box(
            selector_received_nss_msg.format(
                node,
                round(features["cpu_percent"], 2),
                features["bytes_sent"],
                features["bytes_received"],
                features["loss"],
                features["data_size"],
                round(features["latency"], 2),
                features["availability"],
                features["sustainability"],
            ),
            indent=2,
            title="Selector: Received NSS Features",
        )

    def get_neighbors(self):
        return self.neighbors_list

    def add_neighbor(self, neighbor):
        logging.info(f"[Selector] Adding Neighbor: {neighbor}")
        if neighbor not in self.neighbors_list:
            self.neighbors_list.append(neighbor)

    def reset_neighbors(self):
        self.neighbors_list = []

    async def node_selection(self, node):
        """To be overridden by the subclasses (selectors)"""
        pass

    def clear_selector_features(self):
        self.features = {}

    def init_age(self):
        for i in self.neighbors_list:
            self.ages[i] = 1
