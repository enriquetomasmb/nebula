import glob
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv

from nebula.addons.env import check_environment
from nebula.config.config import Config
from nebula.config.mender import Mender
from nebula.addons.topologymanager import TopologyManager
from nebula.core.utils.certificate import generate_certificate, generate_ca_certificate
from nebula.addons.blockchain.blockchain_deployer import BlockchainDeployer
from nebula import __version__


# Setup controller logger
class TermEscapeCodeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        escape_re = re.compile(r"\x1b\[[0-9;]*m")
        record.msg = re.sub(escape_re, "", str(record.msg))
        return super().format(record)


log_console_format = "[%(levelname)s] - %(asctime)s - Controller - %(message)s"
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter(log_console_format))
console_handler.setFormatter(TermEscapeCodeFormatter(log_console_format))
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        console_handler,
    ],
)


# Detect ctrl+c and run killports
def signal_handler(sig, frame):
    Controller.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Controller:
    def __init__(self, args):
        self.scenario_name = args.scenario_name if hasattr(args, "scenario_name") else None
        self.start_date_scenario = None
        self.federation = args.federation if hasattr(args, "federation") else None
        self.topology = args.topology if hasattr(args, "topology") else None
        self.waf_port = args.wafport if hasattr(args, "wafport") else 6000
        self.frontend_port = args.webport if hasattr(args, "webport") else 6060
        self.grafana_port = args.grafanaport if hasattr(args, "grafanaport") else 6040
        self.loki_port = args.lokiport if hasattr(args, "lokiport") else 6010
        self.statistics_port = args.statsport if hasattr(args, "statsport") else 8080
        self.simulation = args.simulation
        self.config_dir = args.config
        self.log_dir = args.logs
        self.cert_dir = args.certs
        self.env_path = args.env
        self.waf = args.waf if hasattr(args, "waf") else False
        self.debug = args.debug if hasattr(args, "debug") else False
        self.matrix = args.matrix if hasattr(args, "matrix") else None
        self.root_path = args.root_path if hasattr(args, "root_path") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Network configuration (nodes deployment in a network)
        self.network_subnet = args.network_subnet if hasattr(args, "network_subnet") else None
        self.network_gateway = args.network_gateway if hasattr(args, "network_gateway") else None

        self.config = Config(entity="controller")
        self.topologymanager = None
        self.n_nodes = 0
        self.mender = None if self.simulation else Mender()
        self.use_blockchain = args.use_blockchain if hasattr(args, "use_blockchain") else False

    def start(self):
        banner = """
                        ███╗   ██╗███████╗██████╗ ██╗   ██╗██╗      █████╗ 
                        ████╗  ██║██╔════╝██╔══██╗██║   ██║██║     ██╔══██╗
                        ██╔██╗ ██║█████╗  ██████╔╝██║   ██║██║     ███████║
                        ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║██║     ██╔══██║
                        ██║ ╚████║███████╗██████╔╝╚██████╔╝███████╗██║  ██║
                        ╚═╝  ╚═══╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝                 
                          A Platform for Decentralized Federated Learning
                            Created by Enrique Tomás Martínez Beltrán
                            https://github.com/enriquetomasmb/nebula
                    """
        print("\x1b[0;36m" + banner + "\x1b[0m")

        # Load the environment variables
        load_dotenv(self.env_path)

        # Check information about the environment
        check_environment()

        # Save the configuration in environment variables
        logging.info("Saving configuration in environment variables...")
        os.environ["NEBULA_ROOT"] = self.root_path
        os.environ["NEBULA_LOGS_DIR"] = self.log_dir
        os.environ["NEBULA_CONFIG_DIR"] = self.config_dir
        os.environ["NEBULA_CERTS_DIR"] = self.cert_dir
        os.environ["NEBULA_STATISTICS_PORT"] = str(self.statistics_port)

        if self.waf:
            self.run_waf()

        self.run_frontend()

        if self.mender:
            logging.info("[Mender.module] Mender module initialized")
            time.sleep(2)
            mender = Mender()
            logging.info("[Mender.module] Getting token from Mender server: {}".format(os.getenv("MENDER_SERVER")))
            mender.renew_token()
            time.sleep(2)
            logging.info("[Mender.module] Getting devices from {} with group Cluster_Thun".format(os.getenv("MENDER_SERVER")))
            time.sleep(2)
            devices = mender.get_devices_by_group("Cluster_Thun")
            logging.info("[Mender.module] Getting a pool of devices: 5 devices")
            # devices = devices[:5]
            for i in self.config.participants:
                logging.info("[Mender.module] Device {} | IP: {}".format(i["device_args"]["idx"], i["network_args"]["ip"]))
                logging.info("[Mender.module] \tCreating artifacts...")
                logging.info("[Mender.module] \tSending NEBULA Core...")
                # mender.deploy_artifact_device("my-update-2.0.mender", i['device_args']['idx'])
                logging.info("[Mender.module] \tSending configuration...")
                time.sleep(5)
            sys.exit(0)

        logging.info("NEBULA Frontend is running at port {}".format(self.frontend_port))
        if self.waf:
            logging.info("NEBULA WAF is running at port {}".format(self.waf_port))
            logging.info("Grafana Dashboard is running at port {}".format(self.grafana_port))

        logging.info("Press Ctrl+C for exit from NEBULA (global exit)")
        while True:
            time.sleep(1)

    def run_waf(self):
        docker_compose_template = textwrap.dedent(
            """
            services:
            {}
        """
        )

        waf_template = textwrap.dedent(
            """
            nebula-waf:
                container_name: nebula-waf
                image: nebula-waf
                build: 
                    context: .
                    dockerfile: Dockerfile-waf
                restart: unless-stopped
                volumes:
                    - {log_path}/waf/nginx:/var/log/nginx
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                ports:
                    - {waf_port}:80
                networks:
                    nebula-net-base:
                        ipv4_address: {ip}
        """
        )

        grafana_template = textwrap.dedent(
            """
            grafana:
                container_name: nebula-waf-grafana
                image: nebula-waf-grafana
                build:
                    context: .
                    dockerfile: Dockerfile-grafana
                restart: unless-stopped
                environment:
                    - GF_SECURITY_ADMIN_PASSWORD=admin
                    - GF_USERS_ALLOW_SIGN_UP=false
                    - GF_SERVER_HTTP_PORT=3000
                    - GF_SERVER_PROTOCOL=http
                    - GF_SERVER_DOMAIN=localhost:{grafana_port}
                    - GF_SERVER_ROOT_URL=http://localhost:{grafana_port}/grafana/
                    - GF_SERVER_SERVE_FROM_SUB_PATH=true
                    - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/dashboard.json
                    - GF_METRICS_MAX_LIMIT_TSDB=0
                ports:
                    - {grafana_port}:3000
                ipc: host
                privileged: true
                networks:
                    nebula-net-base:
                        ipv4_address: {ip}
        """
        )

        loki_template = textwrap.dedent(
            """
            loki:
                container_name: nebula-waf-loki
                image: nebula-waf-loki
                build:
                    context: .
                    dockerfile: Dockerfile-loki
                restart: unless-stopped
                volumes:
                    - ./loki-config.yml:/mnt/config/loki-config.yml
                ports:
                    - {loki_port}:3100
                user: "0:0"
                command: 
                    - '-config.file=/mnt/config/loki-config.yml'
                networks:
                    nebula-net-base:
                        ipv4_address: {ip}
        """
        )

        promtail_template = textwrap.dedent(
            """
            promtail:
                container_name: nebula-waf-promtail
                image: nebula-waf-promtail
                build:
                    context: .
                    dockerfile: Dockerfile-promtail
                restart: unless-stopped
                volumes:
                    - {log_path}/waf/nginx:/var/log/nginx
                    - ./promtail-config.yml:/etc/promtail/config.yml
                command: 
                    - '-config.file=/etc/promtail/config.yml'
                networks:
                    nebula-net-base:
                        ipv4_address: {ip}
        """
        )

        waf_template = textwrap.indent(waf_template, " " * 4)
        grafana_template = textwrap.indent(grafana_template, " " * 4)
        loki_template = textwrap.indent(loki_template, " " * 4)
        promtail_template = textwrap.indent(promtail_template, " " * 4)

        network_template = textwrap.dedent(
            """
            networks:
                nebula-net-base:
                    name: nebula-net-base
                    driver: bridge
                    ipam:
                        config:
                            - subnet: {}
                              gateway: {}
        """
        )

        # Generate the Docker Compose file dynamically
        services = ""
        services += waf_template.format(path=self.root_path, log_path=os.environ["NEBULA_LOGS_DIR"], waf_port=self.waf_port, gw="192.168.10.1", ip="192.168.10.200")

        services += grafana_template.format(log_path=os.environ["NEBULA_LOGS_DIR"], grafana_port=self.grafana_port, loki_port=self.loki_port, ip="192.168.10.201")

        services += loki_template.format(loki_port=self.loki_port, ip="192.168.10.202")

        services += promtail_template.format(log_path=os.environ["NEBULA_LOGS_DIR"], ip="192.168.10.203")

        docker_compose_file = docker_compose_template.format(services)
        docker_compose_file += network_template.format("192.168.10.0/24", "192.168.10.1")

        # Write the Docker Compose file in waf directory
        with open(
            f"{os.path.join(os.environ['NEBULA_ROOT'], 'nebula', 'addons', 'waf', 'docker-compose.yml')}",
            "w",
        ) as f:
            f.write(docker_compose_file)

        # Start the Docker Compose file, catch error if any
        try:
            subprocess.check_call(
                [
                    "docker",
                    "compose",
                    "-f",
                    f"{os.path.join(os.environ['NEBULA_ROOT'], 'nebula', 'addons', 'waf', 'docker-compose.yml')}",
                    "up",
                    "--build",
                    "-d",
                ]
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Docker Compose failed to start, please check if Docker Compose is installed (https://docs.docker.com/compose/install/) and Docker Engine is running.")

    def run_frontend(self):
        if sys.platform == "win32":
            if not os.path.exists("//./pipe/docker_Engine"):
                raise Exception("Docker is not running, please check if Docker is running and Docker Compose is installed.")
        else:
            if not os.path.exists("/var/run/docker.sock"):
                raise Exception("/var/run/docker.sock not found, please check if Docker is running and Docker Compose is installed.")

        docker_compose_template = textwrap.dedent(
            """
            services:
            {}
        """
        )

        frontend_template = textwrap.dedent(
            """
            nebula-frontend:
                container_name: nebula-frontend
                image: nebula-frontend
                build:
                    context: .
                restart: unless-stopped
                volumes:
                    - {path}:/nebula
                    - /var/run/docker.sock:/var/run/docker.sock
                    - ./config/nebula:/etc/nginx/sites-available/default
                environment:
                    - NEBULA_DEV=True
                    - NEBULA_DEBUG={debug}
                    - SERVER_LOG=/nebula/app/logs/server.log
                    - NEBULA_LOGS_DIR=/nebula/app/logs/
                    - NEBULA_CONFIG_DIR=/nebula/app/config/
                    - NEBULA_CERTS_DIR=/nebula/app/certs/
                    - NEBULA_ENV_PATH=/nebula/app/.env
                    - NEBULA_ROOT_HOST={path}
                    - NEBULA_DEFAULT_USER=admin
                    - NEBULA_DEFAULT_PASSWORD=admin
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                ports:
                    - {frontend_port}:80
                    - {statistics_port}:8080
                networks:
                    nebula-net-base:
                        ipv4_address: {ip}
        """
        )
        frontend_template = textwrap.indent(frontend_template, " " * 4)

        network_template = textwrap.dedent(
            """
            networks:
                nebula-net-base:
                    name: nebula-net-base
                    driver: bridge
                    ipam:
                        config:
                            - subnet: {}
                              gateway: {}
        """
        )

        network_template_external = textwrap.dedent(
            """
            networks:
                nebula-net-base:
                    external: true
        """
        )

        # Generate the Docker Compose file dynamically
        services = ""
        services += frontend_template.format(debug=self.debug, path=self.root_path, gw="192.168.10.1", ip="192.168.10.100", frontend_port=self.frontend_port, statistics_port=self.statistics_port)
        docker_compose_file = docker_compose_template.format(services)

        if self.waf:
            # If WAF is enabled, we need to use the same network
            docker_compose_file += network_template_external
        else:
            docker_compose_file += network_template.format("192.168.10.0/24", "192.168.10.1")
        # Write the Docker Compose file in config directory
        with open(
            f"{os.path.join(os.environ['NEBULA_ROOT'], 'nebula', 'frontend', 'docker-compose.yml')}",
            "w",
        ) as f:
            f.write(docker_compose_file)

        # Start the Docker Compose file, catch error if any
        try:
            subprocess.check_call(
                [
                    "docker",
                    "compose",
                    "-f",
                    f"{os.path.join(os.environ['NEBULA_ROOT'], 'nebula', 'frontend', 'docker-compose.yml')}",
                    "up",
                    "--build",
                    "-d",
                ]
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Docker Compose failed to start, please check if Docker Compose is installed (https://docs.docker.com/compose/install/) and Docker Engine is running.")

        except Exception as e:
            raise Exception("Error while starting the frontend: {}".format(e))

    @staticmethod
    def stop_frontend():
        if sys.platform == "win32":
            try:
                # kill all the docker containers which contain the word "nebula"
                commands = [
                    """docker kill $(docker ps -q --filter ancestor=nebula-frontend) | Out-Null""",
                    """docker rm $(docker ps -a -q --filter ancestor=nebula-frontend) | Out-Null""",
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
                    """docker kill $(docker ps -q --filter ancestor=nebula-frontend) > /dev/null 2>&1""",
                    """docker rm $(docker ps -a -q --filter ancestor=nebula-frontend) > /dev/null 2>&1""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(command)
                    # logging.info(f"Linux Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))

    @staticmethod
    def stop_network():
        if sys.platform == "win32":
            try:
                # kill all the docker containers which contain the word "nebula"
                commands = ["""docker network rm $(docker network ls | Where-Object { ($_ -split '\s+')[1] -like 'nebula-net-base' } | ForEach-Object { ($_ -split '\s+')[0] }) | Out-Null"""]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(f'powershell.exe -Command "{command}"')
                    # logging.info(f"Windows Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))
        else:
            try:
                commands = ["""docker network rm $(docker network ls | grep nebula-net-base | awk '{print $1}') > /dev/null 2>&1"""]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(command)
                    # logging.info(f"Linux Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))

    @staticmethod
    def stop_blockchain():
        try:
            subprocess.Popen("docker ps -a --filter 'label=com.docker.compose.project=blockchain' --format '{{.ID}}' | xargs -n 1 docker rm --force --volumes  >/dev/null 2>&1", shell=True)
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
    def stop_waf():
        if sys.platform == "win32":
            try:
                # kill all the docker containers which contain the word "nebula"
                commands = [
                    """docker compose -p waf down | Out-Null""",
                    """docker compose -p waf rm | Out-Null""",
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
                    """docker compose -p waf down > /dev/null 2>&1""",
                    """docker compose -p waf rm > /dev/null 2>&1""",
                ]

                for command in commands:
                    time.sleep(1)
                    exit_code = os.system(command)
                    # logging.info(f"Linux Command '{command}' executed with exit code: {exit_code}")

            except Exception as e:
                raise Exception("Error while killing docker containers: {}".format(e))

    @staticmethod
    def stop():
        logging.info("Closing NEBULA (exiting from components)... Please wait")
        Controller.stop_participants()
        Controller.stop_blockchain()
        Controller.stop_frontend()
        Controller.stop_waf()
        Controller.stop_network()
        sys.exit(0)

    @staticmethod
    def stop_nodes():
        logging.info("Closing NEBULA nodes... Please wait")
        Controller.stop_participants()
        Controller.stop_blockchain()

    def load_configurations_and_start_nodes(self, additional_participants=None, schema_additional_participants=None):
        if not self.scenario_name:
            self.scenario_name = f'nebula_{self.federation}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
        # Once the scenario_name is defined, we can update the config_dir
        self.config_dir = os.path.join(self.config_dir, self.scenario_name)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, self.scenario_name), exist_ok=True)
        os.makedirs(self.cert_dir, exist_ok=True)
        self.start_date_scenario = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
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

        self.topologymanager = self.create_topology(matrix=self.matrix) if self.matrix else self.create_topology()

        # Update participants configuration
        is_start_node = False
        config_participants = []
        for i in range(self.n_nodes):
            with open(f"{self.config_dir}/participant_" + str(i) + ".json") as f:
                participant_config = json.load(f)
            participant_config["scenario_args"]["federation"] = self.federation
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

        if self.simulation:
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
        elif self.topology == "fully":
            # Create a fully connected network
            topologymanager = TopologyManager(
                scenario_name=self.scenario_name,
                n_nodes=self.n_nodes,
                b_symmetric=True,
                undirected_neighbor_num=self.n_nodes - 1,
            )
            topologymanager.generate_topology()
        elif self.topology == "ring":
            # Create a partially connected network (ring-structured network)
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_ring_topology(increase_convergence=True)
        elif self.topology == "random":
            # Create network topology using topology manager (random)
            topologymanager = TopologyManager(
                scenario_name=self.scenario_name,
                n_nodes=self.n_nodes,
                b_symmetric=True,
                undirected_neighbor_num=3,
            )
            topologymanager.generate_topology()
        elif self.topology == "star" and self.federation == "CFL":
            # Create a centralized network
            topologymanager = TopologyManager(scenario_name=self.scenario_name, n_nodes=self.n_nodes, b_symmetric=True)
            topologymanager.generate_server_topology()
        else:
            raise ValueError("Unknown topology type: {}".format(self.topology))

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
        BlockchainDeployer(config_dir=f"{self.config_dir}/blockchain", input_dir="/nebula/nebula/blockchain")
        try:
            logging.info("Blockchain is being deployed")
            subprocess.check_call(["docker", "compose", "-f", f"{self.config_dir}/blockchain/blockchain-docker-compose.yml", "up", "--remove-orphans", "--force-recreate", "-d", "--build"])
        except subprocess.CalledProcessError as e:
            logging.error("Docker Compose failed to start Blockchain, please check if Docker Compose is installed (https://docs.docker.com/compose/install/) and Docker Engine is running.")
            raise e

    def start_nodes_docker(self):
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
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 /nebula/nebula/node.py {}
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
                extra_hosts:
                    - "host.docker.internal:host-gateway"
                ipc: host
                privileged: true
                command:
                    - /bin/bash
                    - -c
                    - |
                        ifconfig && echo '{} host.docker.internal' >> /etc/hosts && python3.11 /nebula/nebula/node.py {}
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
            logging.info("Starting node {} with configuration {}".format(idx, path))
            logging.info("Node {} is listening on ip {}".format(idx, node["network_args"]["ip"]))
            # Add one service for each participant
            if node["device_args"]["accelerator"] == "gpu":
                logging.info("Node {} is using GPU".format(idx))
                services += participant_gpu_template.format(
                    idx,
                    self.root_path,
                    self.network_gateway,
                    path,
                    node["network_args"]["ip"],
                    "proxy:" if self.simulation and self.use_blockchain else "",
                )
            else:
                logging.info("Node {} is using CPU".format(idx))
                services += participant_template.format(
                    idx,
                    self.root_path,
                    self.network_gateway,
                    path,
                    node["network_args"]["ip"],
                    "proxy:" if self.simulation and self.use_blockchain else "",
                )
        docker_compose_file = docker_compose_template.format(services)
        docker_compose_file += network_template.format(self.network_subnet, self.network_gateway, "proxy:" if self.simulation and self.use_blockchain else "", "name: chainnet" if self.simulation and self.use_blockchain else "", "external: true" if self.simulation and self.use_blockchain else "")
        # Write the Docker Compose file in config directory
        with open(f"{self.config_dir}/docker-compose.yml", "w") as f:
            f.write(docker_compose_file)

        # Change log and config directory in dockers to /nebula/app, and change controller endpoint
        for node in self.config.participants:
            # Print the configuration of the node
            node["tracking_args"]["log_dir"] = "/nebula/app/logs"
            node["tracking_args"]["config_dir"] = f"/nebula/app/config/{self.scenario_name}"
            node["scenario_args"]["controller"] = "nebula-frontend"
            node["security_args"]["certfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_cert.pem"
            node["security_args"]["keyfile"] = f"/nebula/app/certs/participant_{node['device_args']['idx']}_key.pem"
            node["security_args"]["cafile"] = f"/nebula/app/certs/ca_cert.pem"

            # Write the config file in config directory
            with open(f"{self.config_dir}/participant_{node['device_args']['idx']}.json", "w") as f:
                json.dump(node, f, indent=4)
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

    # @classmethod
    # def remove_files_by_run_hash(cls, hash_str):
    #     for root, dirs, files in os.walk(f"{os.environ['NEBULA_LOGS_DIR']}/.aim", topdown=False):
    #         for name in files:
    #             if hash_str in name:
    #                 full_path = os.path.join(root, name)
    #                 print(f"Removing file: {full_path}")
    #                 try:
    #                     os.remove(full_path)
    #                 except Exception as e:
    #                     print(f"Error removing file: {full_path}")
    #                     print(e)

    #         for name in dirs:
    #             if hash_str in name:
    #                 full_dir_path = os.path.join(root, name)
    #                 print(f"Removing directory: {full_dir_path}")
    #                 try:
    #                     shutil.rmtree(full_dir_path)
    #                 except Exception as e:
    #                     print(f"Error removing directory: {full_dir_path}")
    #                     print(e)
