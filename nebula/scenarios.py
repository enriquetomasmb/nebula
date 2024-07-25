import glob
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import datetime

import docker

from nebula.addons.blockchain.blockchain_deployer import BlockchainDeployer
from nebula.addons.topologymanager import TopologyManager
from nebula.config.config import Config
from nebula.core.utils.certificate import generate_ca_certificate, generate_certificate


# Definition of a scenario
class Scenario:
    def __init__(
        self,
        scenario_title,
        scenario_description,
        simulation,
        federation,
        topology,
        nodes,
        nodes_graph,
        n_nodes,
        matrix,
        dataset,
        iid,
        partition_selection,
        partition_parameter,
        model,
        agg_algorithm,
        rounds,
        logginglevel,
        accelerator,
        network_subnet,
        network_gateway,
        epochs,
        attacks,
        poisoned_node_percent,
        poisoned_sample_percent,
        poisoned_noise_percent,
        with_reputation,
        is_dynamic_topology,
        is_dynamic_aggregation,
        target_aggregation,
        random_geo,
        latitude,
        longitude,
        mobility,
        mobility_type,
        radius_federation,
        scheme_mobility,
        round_frequency,
        mobile_participants_percent,
        additional_participants,
        schema_additional_participants,
    ):
        self.scenario_title = scenario_title
        self.scenario_description = scenario_description
        self.simulation = simulation
        self.federation = federation
        self.topology = topology
        self.nodes = nodes
        self.nodes_graph = nodes_graph
        self.n_nodes = n_nodes
        self.matrix = matrix
        self.dataset = dataset
        self.iid = iid
        self.partition_selection = partition_selection
        self.partition_parameter = partition_parameter
        self.model = model
        self.agg_algorithm = agg_algorithm
        self.rounds = rounds
        self.logginglevel = logginglevel
        self.accelerator = accelerator
        self.network_subnet = network_subnet
        self.network_gateway = network_gateway
        self.epochs = epochs
        self.attacks = attacks
        self.poisoned_node_percent = poisoned_node_percent
        self.poisoned_sample_percent = poisoned_sample_percent
        self.poisoned_noise_percent = poisoned_noise_percent
        self.with_reputation = with_reputation
        self.is_dynamic_topology = is_dynamic_topology
        self.is_dynamic_aggregation = is_dynamic_aggregation
        self.target_aggregation = target_aggregation
        self.random_geo = random_geo
        self.latitude = latitude
        self.longitude = longitude
        self.mobility = mobility
        self.mobility_type = mobility_type
        self.radius_federation = radius_federation
        self.scheme_mobility = scheme_mobility
        self.round_frequency = round_frequency
        self.mobile_participants_percent = mobile_participants_percent
        self.additional_participants = additional_participants
        self.schema_additional_participants = schema_additional_participants

    def attack_node_assign(
        self,
        nodes,
        federation,
        attack,
        poisoned_node_percent,
        poisoned_sample_percent,
        poisoned_noise_percent,
    ):
        """Identify which nodes will be attacked"""
        import random
        import math

        nodes_index = []
        # Get the nodes index
        if federation == "DFL":
            nodes_index = list(nodes.keys())
        else:
            for node in nodes:
                if nodes[node]["role"] != "server":
                    nodes_index.append(node)

        n_nodes = len(nodes_index)
        # Number of attacked nodes, round up
        num_attacked = int(math.ceil(poisoned_node_percent / 100 * n_nodes))
        if num_attacked > n_nodes:
            num_attacked = n_nodes

        # Get the index of attacked nodes
        attacked_nodes = random.sample(nodes_index, num_attacked)

        # Assign the role of each node
        for node in nodes:
            node_att = "No Attack"
            attack_sample_percent = 0
            poisoned_ratio = 0
            if (node in attacked_nodes) or (nodes[node]["malicious"]):
                node_att = attack
                attack_sample_percent = poisoned_sample_percent / 100
                poisoned_ratio = poisoned_noise_percent / 100
            nodes[node]["attacks"] = node_att
            nodes[node]["poisoned_sample_percent"] = attack_sample_percent
            nodes[node]["poisoned_ratio"] = poisoned_ratio
        return nodes

    def mobility_assign(self, nodes, mobile_participants_percent):
        """Assign mobility to nodes"""
        import random

        # Number of mobile nodes, round down
        num_mobile = math.floor(mobile_participants_percent / 100 * len(nodes))
        if num_mobile > len(nodes):
            num_mobile = len(nodes)

        # Get the index of mobile nodes
        mobile_nodes = random.sample(list(nodes.keys()), num_mobile)

        # Assign the role of each node
        for node in nodes:
            node_mob = False
            if node in mobile_nodes:
                node_mob = True
            nodes[node]["mobility"] = node_mob
        return nodes

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


# Class to manage the current scenario
class ScenarioManagement:
    def __init__(self, scenario, controller):
        # Current scenario
        self.scenario = Scenario.from_dict(scenario)

        # Scenario management settings
        self.start_date_scenario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.scenario_name = f'nebula_{self.scenario.federation}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        self.root_path = os.environ.get("NEBULA_ROOT_HOST")
        self.config_dir = os.path.join(os.environ.get("NEBULA_CONFIG_DIR"), self.scenario_name)
        self.log_dir = os.environ.get("NEBULA_LOGS_DIR")
        self.cert_dir = os.environ.get("NEBULA_CERTS_DIR")
        self.advanced_analytics = os.environ.get("NEBULA_ADVANCED_ANALYTICS", "False") == "True"
        self.config = Config(entity="scenarioManagement")
        self.controller = controller
        self.topologymanager = None
        self.env_path = None
        self.use_blockchain = self.scenario.agg_algorithm == "BlockchainReputation"

        # Create Scenario management dirs
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, self.scenario_name), exist_ok=True)
        os.makedirs(self.cert_dir, exist_ok=True)

        # Save the scenario configuration
        scenario_file = os.path.join(self.config_dir, "scenario.json")
        with open(scenario_file, "w") as f:
            json.dump(scenario, f, sort_keys=False, indent=2)

        # Save management settings
        settings = {
            "scenario_name": self.scenario_name,
            "root_path": self.root_path,
            "config_dir": self.config_dir,
            "log_dir": self.log_dir,
            "cert_dir": self.cert_dir,
            "env": None,
            "use_blockchain": self.use_blockchain,
        }

        settings_file = os.path.join(self.config_dir, "settings.json")
        with open(settings_file, "w") as f:
            json.dump(settings, f, sort_keys=False, indent=2)

        self.scenario.nodes = self.scenario.attack_node_assign(
            self.scenario.nodes,
            self.scenario.federation,
            self.scenario.attacks,
            int(self.scenario.poisoned_node_percent),
            int(self.scenario.poisoned_sample_percent),
            int(self.scenario.poisoned_noise_percent),
        )

        if self.scenario.mobility:
            mobile_participants_percent = int(self.scenario.mobile_participants_percent)
            self.scenario.nodes = self.scenario.mobility_assign(self.scenario.nodes, mobile_participants_percent)
        else:
            self.scenario.nodes = self.scenario.mobility_assign(self.scenario.nodes, 0)

        # Save node settings
        for node in self.scenario.nodes:
            node_config = self.scenario.nodes[node]
            participant_file = os.path.join(self.config_dir, f'participant_{node_config["id"]}.json')
            os.makedirs(os.path.dirname(participant_file), exist_ok=True)
            shutil.copy(
                os.path.join(os.path.dirname(__file__), "./frontend/config/participant.json.example"),
                participant_file,
            )
            with open(participant_file) as f:
                participant_config = json.load(f)

            participant_config["network_args"]["ip"] = node_config["ip"]
            participant_config["network_args"]["port"] = int(node_config["port"])
            participant_config["device_args"]["idx"] = node_config["id"]
            participant_config["device_args"]["start"] = node_config["start"]
            participant_config["device_args"]["role"] = node_config["role"]
            participant_config["device_args"]["proxy"] = node_config["proxy"]
            participant_config["device_args"]["malicious"] = node_config["malicious"]
            participant_config["scenario_args"]["rounds"] = int(self.scenario.rounds)
            participant_config["data_args"]["dataset"] = self.scenario.dataset
            participant_config["data_args"]["iid"] = self.scenario.iid
            participant_config["data_args"]["partition_selection"] = self.scenario.partition_selection
            participant_config["data_args"]["partition_parameter"] = self.scenario.partition_parameter
            participant_config["model_args"]["model"] = self.scenario.model
            participant_config["training_args"]["epochs"] = int(self.scenario.epochs)
            participant_config["device_args"]["accelerator"] = self.scenario.accelerator
            participant_config["device_args"]["logging"] = self.scenario.logginglevel
            participant_config["aggregator_args"]["algorithm"] = self.scenario.agg_algorithm
            participant_config["adversarial_args"]["attacks"] = node_config["attacks"]
            participant_config["adversarial_args"]["poisoned_sample_percent"] = node_config["poisoned_sample_percent"]
            participant_config["adversarial_args"]["poisoned_ratio"] = node_config["poisoned_ratio"]
            participant_config["defense_args"]["with_reputation"] = self.scenario.with_reputation
            participant_config["defense_args"]["is_dynamic_topology"] = self.scenario.is_dynamic_topology
            participant_config["defense_args"]["is_dynamic_aggregation"] = self.scenario.is_dynamic_aggregation
            participant_config["defense_args"]["target_aggregation"] = self.scenario.target_aggregation
            participant_config["mobility_args"]["random_geo"] = self.scenario.random_geo
            participant_config["mobility_args"]["latitude"] = self.scenario.latitude
            participant_config["mobility_args"]["longitude"] = self.scenario.longitude
            participant_config["mobility_args"]["mobility"] = node_config["mobility"]
            participant_config["mobility_args"]["mobility_type"] = self.scenario.mobility_type
            participant_config["mobility_args"]["radius_federation"] = self.scenario.radius_federation
            participant_config["mobility_args"]["scheme_mobility"] = self.scenario.scheme_mobility
            participant_config["mobility_args"]["round_frequency"] = self.scenario.round_frequency

            with open(participant_file, "w") as f:
                json.dump(participant_config, f, sort_keys=False, indent=2)

    @staticmethod
    def stop_blockchain():
        if sys.platform == "win32":
            try:
                # Comando adaptado para PowerShell en Windows
                command = "docker ps -a --filter 'label=com.docker.compose.project=blockchain' --format '{{.ID}}' | ForEach-Object { docker rm --force --volumes $_ } | Out-Null"
                os.system(f'powershell.exe -Command "{command}"')
            except Exception as e:
                logging.error("Error while killing docker containers: {}".format(e))
        else:
            try:
                process = subprocess.Popen("docker ps -a --filter 'label=com.docker.compose.project=blockchain' --format '{{.ID}}' | xargs -n 1 docker rm --force --volumes  >/dev/null 2>&1", shell=True)
                process.wait()
            except subprocess.CalledProcessError:
                logging.error("Docker Compose failed to stop blockchain or blockchain already exited.")
            
    @staticmethod
    def stop_participants():
        if sys.platform == "win32":
            try:
                # kill all the docker containers which contain the word "nebula-core"
                commands = [
                    """docker kill $(docker ps -q --filter ancestor=nebula-core) | Out-Null""",
                    """docker rm $(docker ps -a -q --filter ancestor=nebula-core) | Out-Null""",
                    """docker network rm $(docker network ls | Where-Object { ($_ -split '\s+')[1] -like 'nebula-net-scenario' } | ForEach-Object { ($_ -split '\s+')[0] }) | Out-Null""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(f'powershell.exe -Command "{command}"')
                    # logging.info(f"Windows Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))
        else:
            try:
                commands = [
                    """docker kill $(docker ps -q --filter ancestor=nebula-core) > /dev/null 2>&1""",
                    """docker rm $(docker ps -a -q --filter ancestor=nebula-core) > /dev/null 2>&1""",
                    """docker network rm $(docker network ls | grep nebula-net-scenario | awk '{print $1}') > /dev/null 2>&1""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(command)
                    # logging.info(f"Linux Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))

    @staticmethod
    def stop_nodes():
        logging.info("Closing NEBULA nodes... Please wait")
        ScenarioManagement.stop_participants()
        ScenarioManagement.stop_blockchain()

    def load_configurations_and_start_nodes(self, additional_participants=None, schema_additional_participants=None):
        logging.info("Generating the scenario {} at {}".format(self.scenario_name, self.start_date_scenario))

        # Generate CA certificate
        generate_ca_certificate(dir_path=self.cert_dir)

        # Get participants configurations
        participant_files = glob.glob("{}/participant_*.json".format(self.config_dir))
        participant_files.sort()
        if len(participant_files) == 0:
            raise ValueError("No participant files found in config folder")

        self.config.set_participants_config(participant_files)
        self.n_nodes = len(participant_files)
        logging.info("Number of nodes: {}".format(self.n_nodes))

        self.topologymanager = self.create_topology(matrix=self.scenario.matrix) if self.scenario.matrix else self.create_topology()

        # Update participants configuration
        is_start_node = False
        config_participants = []
        for i in range(self.n_nodes):
            with open(f"{self.config_dir}/participant_" + str(i) + ".json") as f:
                participant_config = json.load(f)
            participant_config["scenario_args"]["federation"] = self.scenario.federation
            participant_config["scenario_args"]["n_nodes"] = self.n_nodes
            participant_config["network_args"]["neighbors"] = self.topologymanager.get_neighbors_string(i)
            participant_config["scenario_args"]["name"] = self.scenario_name
            participant_config["scenario_args"]["start_time"] = self.start_date_scenario
            participant_config["device_args"]["idx"] = i
            participant_config["device_args"]["uid"] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"]) + str(self.scenario_name)).encode()).hexdigest()
            if participant_config["mobility_args"]["random_geo"]:
                (
                    participant_config["mobility_args"]["latitude"],
                    participant_config["mobility_args"]["longitude"],
                ) = TopologyManager.get_coordinates(random_geo=True)
            # If not, use the given coordinates in the frontend
            participant_config["tracking_args"]["local_tracking"] = "advanced" if self.advanced_analytics else "basic"
            participant_config["tracking_args"]["log_dir"] = self.log_dir
            participant_config["tracking_args"]["config_dir"] = self.config_dir

            # Generate node certificate
            keyfile_path, certificate_path = generate_certificate(dir_path=self.cert_dir, node_id=f"participant_{i}", ip=participant_config["network_args"]["ip"])

            participant_config["security_args"]["certfile"] = certificate_path
            participant_config["security_args"]["keyfile"] = keyfile_path

            if participant_config["device_args"]["start"]:
                if not is_start_node:
                    is_start_node = True
                else:
                    raise ValueError("Only one node can be start node")
            with open(f"{self.config_dir}/participant_" + str(i) + ".json", "w") as f:
                json.dump(participant_config, f, sort_keys=False, indent=2)

            config_participants.append((participant_config["network_args"]["ip"], participant_config["network_args"]["port"], participant_config["device_args"]["role"]))
        if not is_start_node:
            raise ValueError("No start node found")
        self.config.set_participants_config(participant_files)

        # Add role to the topology (visualization purposes)
        self.topologymanager.update_nodes(config_participants)
        self.topologymanager.draw_graph(path=f"{self.log_dir}/{self.scenario_name}/topology.png", plot=False)

        # Include additional participants (if any) as copies of the last participant
        additional_participants_files = []
        if additional_participants:
            last_participant_file = participant_files[-1]
            last_participant_index = len(participant_files)

            for i, additional_participant in enumerate(additional_participants):
                additional_participant_file = f"{self.config_dir}/participant_{last_participant_index + i}.json"
                shutil.copy(last_participant_file, additional_participant_file)

                with open(additional_participant_file) as f:
                    participant_config = json.load(f)

                participant_config["scenario_args"]["n_nodes"] = self.n_nodes + i + 1
                participant_config["device_args"]["idx"] = last_participant_index + i
                participant_config["network_args"]["neighbors"] = ""
                participant_config["network_args"]["ip"] = participant_config["network_args"]["ip"].rsplit(".", 1)[0] + "." + str(int(participant_config["network_args"]["ip"].rsplit(".", 1)[1]) + 1)
                participant_config["device_args"]["uid"] = hashlib.sha1((str(participant_config["network_args"]["ip"]) + str(participant_config["network_args"]["port"]) + str(self.scenario_name)).encode()).hexdigest()
                participant_config["mobility_args"]["additional_node"]["status"] = True
                participant_config["mobility_args"]["additional_node"]["round_start"] = additional_participant["round"]

                with open(additional_participant_file, "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)

                additional_participants_files.append(additional_participant_file)

        if additional_participants_files:
            self.config.add_participants_config(additional_participants_files)

        if self.scenario.simulation:
            if self.use_blockchain:
                self.start_blockchain()
            self.start_nodes_docker()
        else:
            logging.info("Simulation mode is disabled, waiting for nodes to start...")

    def create_topology(self, matrix=None):
        import numpy as np

        if matrix is not None:
            if self.n_nodes > 2:
                topologymanager = TopologyManager(
                    topology=np.array(matrix),
                    scenario_name=self.scenario_name,
                    n_nodes=self.n_nodes,
                    b_symmetric=True,
                    undirected_neighbor_num=self.n_nodes - 1,
                )
            else:
                topologymanager = TopologyManager(
                    topology=np.array(matrix),
                    scenario_name=self.scenario_name,
                    n_nodes=self.n_nodes,
                    b_symmetric=True,
                    undirected_neighbor_num=2,
                )
        elif self.scenario.topology == "fully":
            # Create a fully connected network
            topologymanager = TopologyManager(
                scenario_name=self.scenario_name,
                n_nodes=self.n_nodes,
                b_symmetric=True,
                undirected_neighbor_num=self.n_nodes - 1,
            )
            topologymanager.generate_topology()
        elif self.scenario.topology == "ring":
            # Create a partially connected network (ring-structured network)
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_ring_topology(increase_convergence=True)
        elif self.scenario.topology == "random":
            # Create network topology using topology manager (random)
            topologymanager = TopologyManager(
                scenario_name=self.scenario_name,
                n_nodes=self.n_nodes,
                b_symmetric=True,
                undirected_neighbor_num=3,
            )
            topologymanager.generate_topology()
        elif self.scenario.topology == "star" and self.scenario.federation == "CFL":
            # Create a centralized network
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_server_topology()
        else:
            raise ValueError("Unknown topology type: {}".format(self.scenario.topology))

        # Assign nodes to topology
        nodes_ip_port = []
        self.config.participants.sort(key=lambda x: x["device_args"]["idx"])
        for i, node in enumerate(self.config.participants):
            nodes_ip_port.append(
                (
                    node["network_args"]["ip"],
                    node["network_args"]["port"],
                    "undefined",
                )
            )

        topologymanager.add_nodes(nodes_ip_port)
        return topologymanager

    def start_blockchain(self):
        BlockchainDeployer(config_dir=f"{self.config_dir}/blockchain", input_dir="/nebula/nebula/addons/blockchain")
        try:
            logging.info("Blockchain is being deployed")
            subprocess.check_call(["docker", "compose", "-f", f"{self.config_dir}/blockchain/blockchain-docker-compose.yml", "up", "--remove-orphans", "--force-recreate", "-d", "--build"])
        except subprocess.CalledProcessError as e:
            logging.error("Docker Compose failed to start Blockchain, please check if Docker Compose is installed (https://docs.docker.com/compose/install/) and Docker Engine is running.")
            raise e

    def start_nodes_docker(self):
        import subprocess

        try:
            # First, get the list of IDs of exited containers
            result_ps = subprocess.run("docker ps -aq -f status=exited --filter 'name=nebula'", shell=True, check=True, capture_output=True, text=True)

            # Get the container IDs
            container_ids = result_ps.stdout.strip()

            if container_ids:
                # Run the command to remove the containers
                result_rm = subprocess.run(f"docker rm $(docker ps -aq -f status=exited --filter 'name=nebula')", shell=True, check=True, capture_output=True, text=True)
                print(f"Dangling containers removed successfully: {result_rm.stdout.strip()}.")
            else:
                print("No dangling containers to remove.")
        except subprocess.CalledProcessError as e:
            print(f"Error removing stopped containers: {e}")
            print(f"Error output: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        logging.info("Starting nodes using Docker Compose...")
        logging.info("env path: {}".format(self.env_path))

        docker_compose_template = textwrap.dedent(
            """
            services:
            {}
        """
        )

        participant_template = textwrap.dedent(
            """
            participant{}:
                image: nebula-core
                restart: no
                volumes:
                    - {}:/nebula
                    - /var/run/docker.sock:/var/run/docker.sock
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 -m memray run -o {} /nebula/nebula/node.py {}
                networks:
                    nebula-net-scenario:
                        ipv4_address: {}
                    nebula-net-base:
                    {}
        """
        )
        participant_template = textwrap.indent(participant_template, " " * 4)

        participant_gpu_template = textwrap.dedent(
            """
            participant{}:
                image: nebula-core
                environment:
                    - NVIDIA_DISABLE_REQUIRE=true
                restart: no
                volumes:
                    - {}:/nebula
                    - /var/run/docker.sock:/var/run/docker.sock
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 -m memray run -o {} /nebula/nebula/node.py {}
                deploy:
                    resources:
                        reservations:
                            devices:
                                - driver: nvidia
                                  count: all
                                  capabilities: [gpu]
                networks:
                    nebula-net-scenario:
                        ipv4_address: {}
                    nebula-net-base:
                    {}
        """
        )
        participant_gpu_template = textwrap.indent(participant_gpu_template, " " * 4)

        network_template = textwrap.dedent(
            """
            networks:
                nebula-net-scenario:
                    name: nebula-net-scenario
                    driver: bridge
                    ipam:
                        config:
                            - subnet: {}
                              gateway: {}     
                nebula-net-base:
                    name: nebula-net-base
                    external: true
                {}
                    {}
                    {}
        """
        )

        # Generate the Docker Compose file dynamically
        services = ""
        self.config.participants.sort(key=lambda x: x["device_args"]["idx"])
        for node in self.config.participants:
            idx = node["device_args"]["idx"]
            path = f"/nebula/app/config/{self.scenario_name}/participant_{idx}.json"
            mem_log = f"/nebula/app/logs/{self.scenario_name}/mem_participant_{idx}.bin"
            logging.info("Starting node {} with configuration {}".format(idx, path))
            logging.info("Node {} is listening on ip {}".format(idx, node["network_args"]["ip"]))
            # Add one service for each participant
            if node["device_args"]["accelerator"] == "gpu":
                logging.info("Node {} is using GPU".format(idx))
                services += participant_gpu_template.format(
                    idx,
                    self.root_path,
                    self.scenario.network_gateway,
                    mem_log,
                    path,
                    node["network_args"]["ip"],
                    "proxy:" if self.scenario.simulation and self.use_blockchain else "",
                )
            else:
                logging.info("Node {} is using CPU".format(idx))
                services += participant_template.format(
                    idx,
                    self.root_path,
                    self.scenario.network_gateway,
                    path,
                    node["network_args"]["ip"],
                    "proxy:" if self.scenario.simulation and self.use_blockchain else "",
                )
        docker_compose_file = docker_compose_template.format(services)
        docker_compose_file += network_template.format(
            self.scenario.network_subnet, self.scenario.network_gateway, "proxy:" if self.scenario.simulation and self.use_blockchain else "", "name: chainnet" if self.scenario.simulation and self.use_blockchain else "", "external: true" if self.scenario.simulation and self.use_blockchain else ""
        )
        # Write the Docker Compose file in config directory
        with open(f"{self.config_dir}/docker-compose.yml", "w") as f:
            f.write(docker_compose_file)

        # Start the Docker Compose file, catch error if any
        try:
            subprocess.check_call(
                [
                    "docker",
                    "compose",
                    "-f",
                    f"{self.config_dir}/docker-compose.yml",
                    "up",
                    "--build",
                    "-d",
                ]
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Docker Compose failed to start, please check if Docker Compose is installed (https://docs.docker.com/compose/install/) and Docker Engine is running.")

        container_ids = None

        # Obtain container IDs
        try:
            # Obtain docker ids
            result = subprocess.run(["docker", "compose", "-f", f"{self.config_dir}/docker-compose.yml", "ps", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise Exception(f"Error obtaining docker IDs: {result.stderr}")

            container_ids = result.stdout.strip().split("\n")

        except subprocess.CalledProcessError as e:
            raise Exception("Docker Compose failed to start, please check if Docker Compose is installed " "(https://docs.docker.com/compose/install/) and Docker Engine is running.")

        if not container_ids or len(container_ids) != len(self.config.participants):
            raise Exception("The number of container IDs does not match the number of participants.")

        # Change log and config directory in dockers to /nebula/app, and change controller endpoint
        for idx, node in enumerate(self.config.participants):
            # Assign docker ID to node
            node["device_args"]["docker_id"] = container_ids[idx]
            # Print the configuration of the node
            node["tracking_args"]["log_dir"] = "/nebula/app/logs"
            node["tracking_args"]["config_dir"] = f"/nebula/app/config/{self.scenario_name}"
            node["scenario_args"]["controller"] = self.controller
            node["security_args"]["certfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_cert.pem"
            node["security_args"]["keyfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_key.pem"
            node["security_args"]["cafile"] = f"/nebula/app/certs/ca_cert.pem"

            # Write the config file in config directory
            with open(f"{self.config_dir}/participant_{node['device_args']['idx']}.json", "w") as f:
                json.dump(node, f, indent=4)

    @classmethod
    def remove_files_by_scenario(cls, scenario_name):
        try:
            shutil.rmtree(os.path.join(os.environ["NEBULA_CONFIG_DIR"], scenario_name))
        except FileNotFoundError:
            logging.warning("Files not found, nothing to remove")
        except Exception as e:
            logging.error("Unknown error while removing files")
            logging.error(e)
            raise e
        try:
            shutil.rmtree(os.path.join(os.environ["NEBULA_LOGS_DIR"], scenario_name))
        except PermissionError:
            # Avoid error if the user does not have enough permissions to remove the tf.events files
            logging.warning("Not enough permissions to remove the files, moving them to tmp folder")
            os.makedirs(
                os.path.join(os.environ["NEBULA_ROOT"], "app", "tmp", scenario_name),
                exist_ok=True,
            )
            shutil.move(
                os.path.join(os.environ["NEBULA_LOGS_DIR"], scenario_name),
                os.path.join(os.environ["NEBULA_ROOT"], "app", "tmp", scenario_name),
            )
        except FileNotFoundError:
            logging.warning("Files not found, nothing to remove")
        except Exception as e:
            logging.error("Unknown error while removing files")
            logging.error(e)
            raise e

    def scenario_finished(self, timeout_seconds):
        client = docker.from_env()
        all_containers = client.containers.list(all=True)
        containers = [container for container in all_containers if self.scenario_name.lower() in container.name.lower()]

        start_time = datetime.now()
        while True:
            all_containers_finished = True
            for container in containers:
                container.reload()
                if container.status != "exited":
                    all_containers_finished = False
                    break
            if all_containers_finished:
                return True

            current_time = datetime.now()
            elapsed_time = current_time - start_time
            if elapsed_time.total_seconds() >= timeout_seconds:
                for container in containers:
                    container.stop()
                return False

            time.sleep(5)
