import os
import csv
import logging
import json

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

    def __init__(self):
        pass

    @staticmethod
    def calculate_reputation(scenario, log_dir, id_node, addr, nei=None, current_round=None):
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
        aggregated_models_time_normalized = 0
        data_contribution_normalized = 0
        node_participation_normalized = 0

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

                    # Data similitude
                    similarity_file = os.path.join(log_dir, f"participant_{id_node}_similarity.csv")
                    similarity_reputation = Reputation.read_similarity_file(similarity_file, nei)
                    logging.info(f"Similarity reputation: {similarity_reputation}")

                    if len(array_communication) > 0:
                        #logging.info(f"Array communication: {array_communication}")
                        communication_time_normalized = Reputation.callback_normalized_value(array_communication)
                        logging.info(f"Communication time normalized: {communication_time_normalized}")
                    if len(array_aggregated_models) > 0:
                        #logging.info(f"Array aggregated models: {array_aggregated_models}")
                        aggregated_models_time_normalized = Reputation.callback_normalized_value(array_aggregated_models)
                        logging.info(f"Aggregated models time normalized: {aggregated_models_time_normalized}")
                    if count_node_participation > 0:
                        #logging.info(f"Array node participation: {count_node_participation}")
                        node_participation_normalized = Reputation.callback_normalized_value([count_node_participation])
                        logging.info(f"Node participation normalized: {node_participation_normalized}")
                    if len(array_data_contribution) > 0:
                        #logging.info(f"Array data contribution: {array_data_contribution}")
                        data_contribution_normalized = Reputation.callback_normalized_value(array_data_contribution)
                        logging.info(f"Data contribution normalized: {data_contribution_normalized}")
                    
                    # Weights for each metric
                    weight_to_communication = 0.1
                    weight_to_aggregated_models = 0.2
                    weight_to_data_contribution = 0.1
                    weight_to_similarity = 0.4
                    weight_to_node_participation = 0.2

                    # Reputation calculation
                    reputation = ( weight_to_communication * communication_time_normalized 
                                + weight_to_aggregated_models * aggregated_models_time_normalized
                                + weight_to_data_contribution * data_contribution_normalized
                                + weight_to_node_participation * node_participation_normalized
                                + weight_to_similarity * similarity_reputation )
                    
                    logging.info(f"Reputation finally to node {nei}: {reputation}")
                    return reputation
        except Exception as e:
                logging.error(f"Error calculating reputation: {e}")

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
        reputation = 0.0
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
                    reputation = ( float(row['cosine']) * weight_cosine + 
                                  float(row['euclidean']) * weight_euclidean + 
                                  float(row['manhattan']) * weight_manhattan + 
                                  float(row['pearson_correlation']) * weight_pearson)
        return reputation

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