import os
import csv
import logging
import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.engine import Engine

def save_data(scenario, type_data, source_ip, addr, round=None, time=None, data_contribution=None):
    """
    Save communication data between nodes and aggregated models.

    Args:
        source_ip (str): Source IP address.
        addr (str): Destination IP address.
        round (int): Round number.
        time (float): Time taken to process the data. 
    """

    source_ip = source_ip.split(":")[0]
    addr = addr.split(":")[0]

    try:
        combined_data = {}

        if type_data == 'communication':
            combined_data["communication"] = {
                "time": time,
                "round": round,
            }
        elif type_data == 'aggregated_models':
            combined_data["aggregated_models"] = {
                "time": time,
                "round": round,
            }
        elif type_data == 'node_participation':
            combined_data["node_participation"] = {
                "round": round,
            }
        elif type_data == 'data_contribution':
            combined_data["data_contribution"] = {
                "data_contribution": data_contribution,
                "round": round,
            }
        """elif type_data == 'last_activity':
            combined_data["last_activity"] = {
                "time": time,
                "round": round,
            }"""

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = f"{addr}_storing_{source_ip}_info.json"
        full_file_path = os.path.join(script_dir, scenario, file_name)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        all_metrics = []
        if os.path.exists(full_file_path):
            with open(full_file_path, 'r') as existing_file:
                try:
                    all_metrics = json.load(existing_file)
                except json.JSONDecodeError:
                    logging.error(f"JSON decode error in file: {full_file_path}")
                    all_metrics = []

        all_metrics.append(combined_data)

        with open(full_file_path, 'w') as json_file:
                json.dump(all_metrics, json_file, indent=4)

    except Exception as e:
        logging.error(f"Error saving data: {e}")


class Reputation:
    """
    Class to define the reputation of a participant.
    """
    reputation_history = {}
    neighbor_reputation_history = {}

    def __init__(self, engine: "Engine"):
        self._engine = engine

    @property
    def engine(self):
        return self._engine

    def calculate_reputation(self, scenario, log_dir, id_node, addr, nei, current_round=None):
        """
        Calculate the reputation of each participant based on the data stored.

        Args:
            scenario (str): Scenario name.
        """
        logging.info(f"id_node: {id_node}, addr: {addr}, nei: {nei}")
        addr = addr.split(":")[0].strip()
        nei = nei.split(":")[0].strip()

        array_communication = []
        array_aggregated_models = []
        count_node_participation = 0
        array_data_contribution = []

        communication_time_normalized = 0
        aggregated_models_time_avg = 0
        data_contribution_normalized = 0
        node_participation = 0

        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_name = f"{addr}_storing_{nei}_info.json" # Read the file to calculate the reputation
            full_file_path = os.path.join(script_dir, scenario, file_name)
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

            if os.path.exists(full_file_path) and os.path.getsize(full_file_path) > 0:
                with open(full_file_path, 'r') as json_file:
                    all_metrics = json.load(json_file)
                    for metric in all_metrics:
                        if "communication" in metric:
                            round = metric["communication"]["round"]    
                            if round == current_round:
                                array_communication.append(metric["communication"]["time"])
                        if "aggregated_models" in metric:
                            round = metric["aggregated_models"]["round"]
                            if round == current_round:
                                array_aggregated_models.append(metric["aggregated_models"]["time"])
                        if "node_participation" in metric:
                            round = metric["node_participation"]["round"]
                            if round == current_round:
                                count_node_participation += 1
                        if "data_contribution" in metric:
                            round = metric["data_contribution"]["round"]
                            if round == current_round:
                                array_data_contribution.append(metric["data_contribution"]["data_contribution"])
                                
                        """if "last_activity" in metric:
                            round = metric["last_activity"]["round"]
                            if round == current_round:
                                last_activity_time = metric["last_activity"]["time"]
                                if isinstance(last_activity_time, float):
                                    last_activity_time = datetime.fromtimestamp(last_activity_time).strftime("%Y-%m-%d %H:%M:%S")
                                last_activity_times.append(last_activity_time)"""

                    # Data similitude
                    similarity_file = os.path.join(log_dir, f"participant_{id_node}_similarity.csv")
                    similarity_reputation = Reputation.read_similarity_file(similarity_file, nei)
                    logging.info(f"Similarity reputation: {similarity_reputation}")

                    if len(array_communication) > 0:
                        communication_time_normalized = Reputation.callback_normalized_value(array_communication)
                    else: 
                        communication_time_normalized = 0

                    if len(array_aggregated_models) > 0:
                        aggregated_models_time_avg = sum(array_aggregated_models) / len(array_aggregated_models)
                        if aggregated_models_time_avg >= 1:
                            aggregated_models_time_avg = Reputation.normalize(aggregated_models_time_avg, 0.25, aggregated_models_time_avg+1)
                    else:
                        aggregated_models_time_avg = 0

                    if count_node_participation > 0:
                        node_participation = count_node_participation
                    else: 
                        node_participation = 0

                    if len(array_data_contribution) > 0:
                        data_contribution_normalized = Reputation.callback_normalized_value(array_data_contribution)
                    else: 
                        data_contribution_normalized = 0
                    
                    # Calculate average last_activity
                    """if len(last_activity_times) > 0:
                        last_activity_datetimes = [datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S") for time_str in last_activity_times]

                        avg_last_activity_datetime = sum([(dt - datetime(1970, 1, 1)).total_seconds() for dt in last_activity_datetimes]) / len(last_activity_datetimes)
                        avg_last_activity = datetime(1970, 1, 1) + timedelta(seconds=avg_last_activity_datetime)

                        current_time = datetime.now()
                        time_diff = (current_time - avg_last_activity).total_seconds()

                        intervals = [
                            (10, 1),
                            (15, 0.9),
                            (20, 0.8),
                            (30, 0.7),
                            (40, 0.6),
                            (50, 0.5),
                            (60, 0.4),
                            (70, 0.3),
                            (80, 0.2),
                            (90, 0.1),
                        ]
                        avg_last_activity_normalized = next((score for threshold, score in intervals if time_diff <= threshold), 0.1)
                        logging.info(f"Avg last activity times: {avg_last_activity_normalized}")

                    else:
                        avg_last_activity_normalized = 0 # No last activity """

                    # Weights for each metric
                    weight_to_communication = 0.2
                    weight_to_aggregated_models = 0.1
                    weight_to_data_contribution = 0.1
                    weight_to_similarity = 0.4
                    weight_to_node_participation = 0.2
                    #weight_to_last_activity = 0.2

                    logging.info(f"Before calculate reputation")
                    logging.info(f"Communication: {communication_time_normalized}")
                    logging.info(f"Aggregated models: {aggregated_models_time_avg}")
                    logging.info(f"Data contribution: {data_contribution_normalized}")
                    logging.info(f"Node participation: {node_participation}")
                    logging.info(f"Similarity: {similarity_reputation}")
                    
                    logging.info(f"Width weigh*metric")
                    logging.info(f"Communication: {weight_to_communication * communication_time_normalized}")
                    logging.info(f"Aggregated models: {weight_to_aggregated_models * aggregated_models_time_avg}")
                    logging.info(f"Data contribution: {weight_to_data_contribution * data_contribution_normalized}")
                    logging.info(f"Node participation: {weight_to_node_participation * node_participation}")
                    logging.info(f"Similarity: {weight_to_similarity * similarity_reputation}")
                    # Reputation calculation
                    reputation = ( weight_to_communication * communication_time_normalized 
                                + weight_to_aggregated_models * aggregated_models_time_avg
                                + weight_to_data_contribution * data_contribution_normalized
                                + weight_to_node_participation * node_participation
                                + weight_to_similarity * similarity_reputation )

                    # Create graphics to metrics
                    self.create_graphics_to_metrics(communication_time_normalized, aggregated_models_time_avg, data_contribution_normalized, node_participation, similarity_reputation, addr, nei, current_round, self.engine.total_rounds)

                    # Save history reputation
                    average_reputation = Reputation.save_reputation_history_in_memory(addr, nei, reputation, current_round)
                    
                    logging.info(f"Average reputation to node {nei}: {average_reputation}")
                    return average_reputation
        except Exception as e:
                logging.error(f"Error calculating reputation: {e}")

    def create_graphics_to_metrics(self, com_time, agg_time, data_contribution, node_participation, similarity, addr, nei, current_round, total_rounds):
        if current_round is not None and current_round < total_rounds:
            communication_time_dict = {
                f"Reputation_communication_time/{addr}": {
                    nei: com_time
                }
            }

            aggregated_models_time_dict = {
                f"Reputation_aggregated_models_time/{addr}": {
                    nei: agg_time
                }
            }

            data_contribution_dict = {
                f"Reputation_data_contribution/{addr}": {
                    nei: data_contribution
                }
            }

            node_participation_dict = {
                f"Reputation_node_participation/{addr}": {
                    nei: node_participation
                }
            }

            similarity_dict = {
                f"Reputation_similarity/{addr}": {
                    nei: similarity
                }
            }

            if communication_time_dict is not None:
                self.engine.trainer._logger.log_data(communication_time_dict, step=current_round)

            if aggregated_models_time_dict is not None:
                self.engine.trainer._logger.log_data(aggregated_models_time_dict, step=current_round)

            if data_contribution_dict is not None:
                self.engine.trainer._logger.log_data(data_contribution_dict, step=current_round)

            if node_participation_dict is not None:
                self.engine.trainer._logger.log_data(node_participation_dict, step=current_round)

            if similarity_dict is not None:
                self.engine.trainer._logger.log_data(similarity_dict, step=current_round)

    @staticmethod
    def save_reputation_history_in_memory(addr, nei, reputation, current_round):
        """
        Save the reputation history of a participant (addr) regarding its neighbor (nei) in memory 
        and calculate the average reputation.

        Args:
            addr (str): The identifier of the node whose reputation is being saved.
            nei (str): The neighboring node involved.
            reputation (float): The reputation value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative reputation including the current round.
        """

        key = (addr, nei) 

        if key not in Reputation.reputation_history:
            Reputation.reputation_history[key] = {}

        logging.info(f"[Before Update] Reputation history for {key}: {Reputation.reputation_history[key]}")
        Reputation.reputation_history[key][current_round] = reputation

        total_reputation = 0
        total_weights = 0
        rounds = sorted(Reputation.reputation_history[key].keys(), reverse=True)

        logging.info(f"Rounds being processed for {key}: {rounds}")

        for i, round in enumerate(rounds):
            rep = Reputation.reputation_history[key][round]
            decay_factor = Reputation.calculate_decay_rate(rep) ** i
            total_reputation += rep * decay_factor
            total_weights += decay_factor
            logging.info(f"Round: {round}, Reputation: {rep}, Decay: {decay_factor}, Total reputation: {total_reputation}")

        if total_weights > 0:
            normalized_reputation = total_reputation / total_weights
        else:
            normalized_reputation = 0

        logging.info(f"[After Update] Reputation history for {key}: {normalized_reputation}")
        return normalized_reputation
    
    @staticmethod
    def calculate_decay_rate(reputation):
        """
        Calculate the decay rate for a reputation value.

        Args:
            reputation (float): Reputation value.

        Returns:
            float: Decay rate.
        """

        if reputation > 0.8:
            return reputation * 0.95 # Muy bajo decaimiento
        elif reputation > 0.6:
            return reputation * 0.9  # Bajo decaimiento
        elif reputation > 0.4:
            return reputation * 0.8  # Moderado decaimiento
        else:
            return reputation * 0.5  # Alto decaimiento
        
    @staticmethod
    def read_similarity_file(file_path, nei):
        """
        Read a similarity file and extract relevant data for each IP.

        Args:
            file_path (str): Path to the similarity file.

        Returns:
            dict: A dictionary containing relevant data for each IP extracted from the file.
                Each IP will have a dictionary containing cosine, euclidean, minkowski,
                manhattan, pearson_correlation, and jaccard values.
        """
        nei = nei.split(":")[0].strip()
        similarity = 0.0
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                source_ip = row['source_ip'].split(":")[0].strip()
                if source_ip == nei:
                    # Design weights for each similarity metric
                    weight_cosine = 0.25
                    weight_euclidean = 0.25
                    weight_manhattan = 0.25
                    weight_pearson = 0.25
                    # Calculate similarity
                    similarity = ( float(row['cosine']) * weight_cosine + 
                                  float(row['euclidean']) * weight_euclidean + 
                                  float(row['manhattan']) * weight_manhattan + 
                                  float(row['pearson_correlation']) * weight_pearson)
        return similarity

    @staticmethod
    def callback_normalized_value (array):
        """
        Calculate the reputation of a node based on the normalized values.

        Args:
            array (list): List of values to normalize.
        
        Returns:
            float: Reputation value (or None if array is empty).
        """

        if not array:
            return None
    
        if len(array) == 1:
            if array[0] >= 1:
                return Reputation.normalize(array[0], 0.25, array[0]+1)
            else:
                return Reputation.normalize(array[0], 0.25, array[0])
        
        min_value = min(array)
        max_value = max(array)
        
        # See values from array
        normalized_values = [Reputation.normalize(value, min_value, max_value) for value in array]
        reputation_normalized = sum(normalized_values) / len(normalized_values)

        return reputation_normalized
    
    @staticmethod
    def normalize(value, min_value, max_value):
        """
        Normalize value within a given range.

        Args:
            value (float): Reputation value.
            min_reputation (float): Minimum reputation value.
            max_reputation (float): Maximum reputation value.

        Returns:
            float: Normalized reputation value within the range [1, 10].
        """

        if max_value == min_value:
            return 0.25
        
        normalized_value = (value - min_value) / (max_value - min_value)

        return normalized_value
    
    # METHODS TO BE USED IN THE FUTURE
    """
            @staticmethod
    def get_existing_reputation(current, nei, round):
        
        Get the existing reputation for a node with respect to a specific neighbor for a specific round.

        Args:
            addr (str): The identifier of the node whose reputation is being retrieved.
            nei (str): The identifier of the neighbor.
            round (int): Round number.

        Returns:
            float: Existing reputation score, or None if not found.
        
        key = (current, nei, round)
        
        # Comprobar si la clave existe en el historial de reputación
        if key in Reputation.reputation_history:
            logging.info(f"History key: {key}, reputation: {Reputation.reputation_history[key]}")
            history_list = Reputation.reputation_history[key]

            # Si la estructura es una lista simple, devolvemos el último valor
            return history_list[-1] if history_list else None

        return None

    def combine_reputation_with_neighbour(self, current_node, source, node_ip, score, round):
        
        Combine the reputation of a node with its neighbour.

        Args:
            current_node (str): The current node's IP address.
            source (str): Source IP address (the node sending the reputation).
            node_ip (str): The node being evaluated.
            score (float): Reputation score from neighbour.
            round (int): Round number.

        Returns:
            float: Combined reputation score or None if the calculation has already been done for this round.
        

        current_node = current_node.split(":")[0].strip()
        source = source.split(":")[0].strip()
        node_ip = node_ip.split(":")[0].strip()
        logging.info(f"Combining reputation - Current node: {current_node}, Source: {source}, Neighbour: {node_ip}, Score: {score}, Round: {round}")

        if current_node == node_ip:
            logging.info(f"Node {current_node} ignoring score about itself ({node_ip}).")
            return None

        # Definir la clave con current_node, node_ip y round
        key = (current_node, node_ip, round)

        # Proceder con el cálculo de reputación si no existe previamente
        if key not in Reputation.neighbor_reputation_history:
            Reputation.neighbor_reputation_history[key] = []

        # Guardar la nueva puntuación enviada por el vecino
        Reputation.neighbor_reputation_history[key].append(score)

        # Calcular la reputación promedio del vecino
        total_reputation = sum(Reputation.neighbor_reputation_history[key])
        average_reputation = total_reputation / len(Reputation.neighbor_reputation_history[key])

        # Obtener la reputación existente del historial
        existing_reputation = Reputation.get_existing_reputation(current_node, node_ip, round)
        if existing_reputation is None:
            logging.warning(f"No existing reputation found for node {current_node} with neighbor {node_ip} in round {round}.")
            return None

        # Definir los pesos para combinar reputaciones
        weight_existing = 0.5
        weight_new = 0.5

        # Combinar la reputación existente con la nueva reputación del vecino
        combined_reputation = (weight_existing * existing_reputation) + (weight_new * average_reputation)
        logging.info(f"Combined reputation for node {current_node} with neighbor {source}: {combined_reputation}")

        return combined_reputation
    """