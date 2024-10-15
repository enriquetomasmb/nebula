import os
import csv
import logging
import json
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.engine import Engine

def save_data(scenario, type_data, source_ip, addr, round=None, time=None, data_contribution=None, type_message=None, current_round=None, rate_of_change=None):
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
                "current_round": current_round,
                "round": round,
                "type_message": type_message,
            }
        elif type_data == 'time_message':
            combined_data["time_message"] = {
                "time": time,
                "type_message": type_message,
                "round": round,
                "current_round": current_round,
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
        elif type_data == 'rate_of_change':
            combined_data["rate_of_change"] = {
                "rate_of_change": rate_of_change,
                "round": round,
            }
        # elif type_data == 'last_activity':
        #     combined_data["last_activity"] = {
        #         "time": time,
        #         "round": round,
        #     }

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
    communication_history = {}
    frequency_history = {}
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
        array_messages_frequency = []
        count_node_participation = 0
        array_data_contribution = []
        rate_of_change = 0

        communication_time_normalized = 0
        messages_frequency_normalized = 0
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
                            type_message = metric["communication"]["type_message"]
                            round = metric["communication"]["round"] 
                            current_round_comm = metric["communication"]["current_round"]
                            time = metric["communication"]["time"]
                            if current_round == current_round_comm:
                                array_communication.append({"time": time, "type_message": type_message, "round": round, "current_round": current_round_comm})
                        if "time_message" in metric:
                            round = metric["time_message"]["round"]
                            current_round_mess = metric["time_message"]["current_round"]
                            time = metric["time_message"]["time"]
                            type_message = metric["time_message"]["type_message"]
                            if current_round == current_round_mess:
                                array_messages_frequency.append({"time": time, "type_message": type_message, "round": round, "current_round": current_round_mess})
                        if "rate_of_change" in metric:
                            round = metric["rate_of_change"]["round"]
                            if round == current_round:
                                rate_of_change = metric["rate_of_change"]["rate_of_change"]
                        # if "node_participation" in metric:
                        #     round = metric["node_participation"]["round"]
                        #     if round == current_round:
                        #         count_node_participation += 1
                        # if "data_contribution" in metric:
                        #     round = metric["data_contribution"]["round"]
                        #     if round == current_round:
                        #         array_data_contribution.append(metric["data_contribution"]["data_contribution"])

                    # Data similitude
                    similarity_file = os.path.join(log_dir, f"participant_{id_node}_similarity.csv")
                    similarity_reputation = Reputation.read_similarity_file(similarity_file, nei)
                    #logging.info(f"Similarity reputation: {similarity_reputation}")

                    if len(array_communication) > 0:
                        communication_time_normalized = Reputation.manage_metric_communication(array_communication, current_round)
                        avg_communication_time_normalized = Reputation.save_communication_history(addr, nei, communication_time_normalized, current_round)
                    else: 
                        communication_time_normalized = 0

                    if len(array_messages_frequency) > 0:
                        messages_frequency_normalized, messages_frequency_count = Reputation.manage_metric_frequency(array_messages_frequency, current_round)
                        avg_messages_frequency_normalized = Reputation.save_frequency_history(addr, nei, messages_frequency_normalized, current_round)
                    else:
                        messages_frequency_normalized = 0

                    if rate_of_change:
                        if rate_of_change < 5:
                            rate_of_change_normalized = max(0.0, 1 - rate_of_change / 5)
                        else:
                            rate_of_change_normalized = 0.0

                    # if count_node_participation > 0:
                    #     node_participation = count_node_participation
                    # else: 
                    #     node_participation = 0

                    # if len(array_data_contribution) > 0:
                    #     data_contribution_normalized = Reputation.callback_normalized_value(array_data_contribution)
                    # else: 
                    #     data_contribution_normalized = 0

                    # Weights for each metric
                    weight_to_communication = 0.3
                    weight_to_message_frequency = 0.2
                    weight_to_similarity = 0.4
                    weight_to_rate_of_change = 0.1
                    # weight_to_node_participation = 0.0
                    # weight_to_data_contribution = 0.0
                    #weight_to_last_activity = 0.2
                    
                    # Reputation calculation
                    reputation = ( weight_to_communication * avg_communication_time_normalized 
                                + weight_to_message_frequency * avg_messages_frequency_normalized
                                + weight_to_similarity * similarity_reputation 
                                + weight_to_rate_of_change * rate_of_change_normalized)	

                    # Create graphics to metrics
                    self.create_graphics_to_metrics(communication_time_normalized, messages_frequency_count, avg_messages_frequency_normalized, similarity_reputation, rate_of_change_normalized, addr, nei, current_round, self.engine.total_rounds)

                    # Save history reputation
                    average_reputation = Reputation.save_reputation_history_in_memory(addr, nei, reputation, current_round)
                    
                    logging.info(f"Average reputation to node {nei}: {average_reputation}")
                    return average_reputation
        except Exception as e:
                logging.error(f"Error calculating reputation: {e}")

    def create_graphics_to_metrics(self, com_time, mess_fre_count, mess_fre_norm, similarity, rate_of_change, addr, nei, current_round, total_rounds):
        """
        Create graphics to metrics.
        """
        
        if current_round is not None and current_round < total_rounds:
            communication_time_dict = {
                f"Reputation_communication_time/{addr}": {
                    nei: com_time
                }
            }

            messages_frequency_count_dict = {
                f"Reputation_messages_frequency_count/{addr}": {
                    nei: mess_fre_count
                }
            }

            messages_frequency_norm_dict = {
                f"Reputation_messages_frequency_normalized/{addr}": {
                    nei: mess_fre_norm
                }
            }

            # data_contribution_dict = {
            #     f"Reputation_data_contribution/{addr}": {
            #         nei: data_contribution
            #     }
            # }

            # node_participation_dict = {
            #     f"Reputation_node_participation/{addr}": {
            #         nei: node_participation
            #     }
            # }

            similarity_dict = {
                f"Reputation_similarity/{addr}": {
                    nei: similarity
                }
            }

            rate_of_change_dict = {
                f"Reputation_rate_of_change/{addr}": {
                    nei: rate_of_change
                }
            }

            if communication_time_dict is not None:
                self.engine.trainer._logger.log_data(communication_time_dict, step=current_round)

            if messages_frequency_count_dict is not None:
                self.engine.trainer._logger.log_data(messages_frequency_count_dict, step=current_round)

            if messages_frequency_norm_dict is not None:
                self.engine.trainer._logger.log_data(messages_frequency_norm_dict, step=current_round)

            # if data_contribution_dict is not None:
            #     self.engine.trainer._logger.log_data(data_contribution_dict, step=current_round)

            # if node_participation_dict is not None:
            #     self.engine.trainer._logger.log_data(node_participation_dict, step=current_round)

            if similarity_dict is not None:
                self.engine.trainer._logger.log_data(similarity_dict, step=current_round)

            if rate_of_change_dict is not None:
                self.engine.trainer._logger.log_data(rate_of_change_dict, step=current_round)

    @staticmethod
    def save_frequency_history(addr, nei, messages_frequency_normalized, current_round):
        """
        Save the frequency history of a participant (addr) regarding its neighbor (nei) in memory.

        Args:
            addr (str): The identifier of the node whose frequency history is being saved.
            nei (str): The neighboring node involved.
            messages_frequency_normalized (float): The frequency value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative frequency including the current round.
        """

        key = (addr, nei)

        if key not in Reputation.frequency_history:
            Reputation.frequency_history[key] = {}

        Reputation.frequency_history[key][current_round] = messages_frequency_normalized

        rounds = Reputation.frequency_history[key]
        recent_rounds = sorted(rounds.keys(), reverse=True)[:2]
        total_frequency = 0
        count_frequency = 0

        for round_num in recent_rounds:
            freq = rounds[round_num]
            total_frequency += freq
            count_frequency += 1

        avg_frequency = total_frequency / count_frequency if count_frequency > 0 else 0

        return avg_frequency

    @staticmethod
    def manage_metric_frequency(array_messages_frequency, current_round=None):
        """
        Manage the frequency metric. Count messages received in the last 60 seconds.

        Args:
            array_messages_frequency (list): List of message frequencies.
            current_round (int): Current round number.

        Returns:
            frequency_score (float): Frequency score.
        """

        start_time = time.time()
        interval = 60
        max_messages = 5

        filtered_messages = []
        for msg in array_messages_frequency:
            if "time" in msg:
                if msg["time"] >= start_time - interval:
                    filtered_messages.append(msg)

        messages_count = len(filtered_messages)

        if messages_count >= max_messages:
            # Penalize when the number of messages exceeds the maximum
            messages_count = max_messages + (max_messages - 1) * (messages_count - max_messages) / (messages_count + 1)

        normalized_messages = 1 - (messages_count / max_messages)

        return normalized_messages, messages_count
    
    @staticmethod
    def manage_metric_communication(array_communication, current_round=None):
        """
        Manage the communication metric.

        Args:
            array_communication (list): List of communication times.
            current_round (int): Current round number.

        Returns:
            communication_score (float): Communication score.
        """

        communication_scores = {
            "model": 0.0,
            "federation": 0.0,
            "reputation": 0.0
        }

        message_counts = {
            "model": 0,
            "federation": 0,
            "reputation": 0
        }
        thresold = 5.0

        for communication in array_communication:
            time_comm = communication["time"]
            type_message = communication["type_message"]
            current_round_comm = communication["current_round"]

            message_counts[type_message] += 1

            if time_comm <= thresold:   
                time_score = 0.5 if current_round_comm == current_round else 0.25
                if type_message == "model":
                    communication_scores["model"] += time_score + (0.5 if current_round_comm == current_round else 0.2)
                elif type_message == "federation":
                    communication_scores["federation"] += time_score + (0.3 if current_round_comm == current_round else 0.15)
                elif type_message == "reputation":
                    communication_scores["reputation"] += time_score + (0.2 if current_round_comm == current_round else 0.1)
            else:
                time_score = 0.2 if current_round_comm == current_round else 0.1  
                if type_message == "model":
                    communication_scores["model"] += time_score + (0.1 if current_round_comm == current_round else 0.05)
                elif type_message == "federation":
                    communication_scores["federation"] += time_score + (0.05 if current_round_comm == current_round else 0.025)
                elif type_message == "reputation":
                    communication_scores["reputation"] += time_score + (0.05 if current_round_comm == current_round else 0.025)

        penalty = 0.0
        if message_counts["model"] == 0:
            penalty += 0.1
        if message_counts["federation"] == 0:
            penalty += 0.05
        if message_counts["reputation"] == 0:
            penalty += 0.05
            
        total_score = sum(communication_scores.values()) - penalty
        total_messages = sum(message_counts.values())
        max_possible_score = total_messages if total_messages > 0 else 1
        communication_score = Reputation.normalize(total_score, 0, max_possible_score)

        return communication_score

    @staticmethod
    def save_communication_history(addr, nei, communication, current_round):
        """
        Save the communication history of a participant (addr) regarding its neighbor (nei) in memory.

        Args:
            addr (str): The identifier of the node whose communication history is being saved.
            nei (str): The neighboring node involved.
            communication (float): The communication value to be saved.
            current_round (int): The current round number.

        Returns:
            float: The cumulative communication including the current round.
        """

        key = (addr, nei)

        if key not in Reputation.communication_history:
            Reputation.communication_history[key] = {}

        Reputation.communication_history[key][current_round] = communication

        rounds = Reputation.communication_history[key]
        recent_rounds = sorted(rounds.keys(), reverse=True)[:2]
        total_communication = 0
        count_communication = 0

        for round_num in recent_rounds:
            comm = rounds[round_num]
            total_communication += comm
            count_communication += 1

        avg_communication = total_communication / count_communication if count_communication > 0 else 0

        return avg_communication

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

        Reputation.reputation_history[key][current_round] = reputation

        total_reputation = 0
        total_weights = 0
        rounds = sorted(Reputation.reputation_history[key].keys(), reverse=True)[:3]

        for i, round in enumerate(rounds, start=1):
            rep = Reputation.reputation_history[key][round]
            decay_factor = Reputation.calculate_decay_rate(rep) ** (i * 2) # Aument the decay factor * 2
            total_reputation += rep * decay_factor
            total_weights += decay_factor
            logging.info(f"Round: {round}, Reputation: {rep}, Decay: {decay_factor}, Total reputation: {total_reputation}")

        if total_weights > 0:
            avg_reputation = total_reputation / total_weights
        else:
            avg_reputation = 0

        return avg_reputation
    
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
            return 0.9 # Muy bajo decaimiento
        elif reputation > 0.6:
            return 0.7  # Bajo decaimiento
        elif reputation > 0.4:
            return 0.5  # Moderado decaimiento
        else:
            return 0.2  # Alto decaimiento
        
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
                    try: 
                        # Design weights for each similarity metric
                        weight_cosine = 0.4
                        weight_euclidean = 0.2
                        weight_manhattan = 0.2
                        weight_pearson = 0.2

                        # Retrieve and normalize metrics if necessary
                        cosine = float(row['cosine'])
                        euclidean = float(row['euclidean'])
                        manhattan = float(row['manhattan'])
                        pearson_correlation = float(row['pearson_correlation'])

                        # Calculate similarity
                        similarity = ( weight_cosine * cosine
                                     + weight_euclidean * euclidean
                                     + weight_manhattan * manhattan
                                     + weight_pearson * pearson_correlation)
                    except Exception as e:
                        logging.error(f"Error reading similarity file: {e}")
        return similarity

    @staticmethod
    def callback_normalized_value (array, metric=None, key=None):
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

        return max(0.0, min(1.0, normalized_value))
    
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