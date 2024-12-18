import asyncio
import importlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time

import docker
import psutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from nebula.addons.env import check_environment
from nebula.config.config import Config
from nebula.config.mender import Mender
from nebula.scenarios import ScenarioManagement
from nebula.tests import main as deploy_tests
from nebula.utils import DockerUtils, SocketUtils


# Setup controller logger
class TermEscapeCodeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%", validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        escape_re = re.compile(r"\x1b\[[0-9;]*m")
        record.msg = re.sub(escape_re, "", str(record.msg))
        return super().format(record)


# Initialize FastAPI app outside the Controller class
app = FastAPI()


# Define endpoints outside the Controller class
@app.get("/")
async def read_root():
    return {"message": "Welcome to the NEBULA Controller API"}


@app.get("/status")
async def get_status():
    return {"status": "NEBULA Controller API is running"}


@app.get("/resources")
async def get_resources():
    devices = 0
    gpu_memory_percent = []

    # Obtain available RAM
    memory_info = await asyncio.to_thread(psutil.virtual_memory)

    if importlib.util.find_spec("pynvml") is not None:
        try:
            import pynvml

            await asyncio.to_thread(pynvml.nvmlInit)
            devices = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)

            # Obtain GPU info
            for i in range(devices):
                handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                memory_info_gpu = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                memory_used_percent = (memory_info_gpu.used / memory_info_gpu.total) * 100
                gpu_memory_percent.append(memory_used_percent)

        except Exception:  # noqa: S110
            pass

    return {
        # "cpu_percent": psutil.cpu_percent(),
        "gpus": devices,
        "memory_percent": memory_info.percent,
        "gpu_memory_percent": gpu_memory_percent,
    }


@app.get("/least_memory_gpu")
async def get_least_memory_gpu():
    gpu_with_least_memory_index = None

    if importlib.util.find_spec("pynvml") is not None:
        try:
            import pynvml

            await asyncio.to_thread(pynvml.nvmlInit)
            devices = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)

            # Obtain GPU info
            for i in range(devices):
                handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                memory_info = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                memory_used_percent = (memory_info.used / memory_info.total) * 100

                # Obtain GPU with less memory available
                if memory_used_percent > max_memory_used_percent:
                    max_memory_used_percent = memory_used_percent
                    gpu_with_least_memory_index = i

        except Exception:  # noqa: S110
            pass

    return {
        "gpu_with_least_memory_index": gpu_with_least_memory_index,
    }


@app.get("/available_gpus/")
async def get_available_gpu():
    available_gpus = []

    if importlib.util.find_spec("pynvml") is not None:
        try:
            import pynvml

            await asyncio.to_thread(pynvml.nvmlInit)
            devices = await asyncio.to_thread(pynvml.nvmlDeviceGetCount)

            # Obtain GPU info
            for i in range(devices):
                handle = await asyncio.to_thread(pynvml.nvmlDeviceGetHandleByIndex, i)
                memory_info = await asyncio.to_thread(pynvml.nvmlDeviceGetMemoryInfo, handle)
                memory_used_percent = (memory_info.used / memory_info.total) * 100

                # Obtain available GPUs
                if memory_used_percent < 5:
                    available_gpus.append(i)

            return {
                "available_gpus": available_gpus,
            }
        except Exception:  # noqa: S110
            pass


class NebulaEventHandler(PatternMatchingEventHandler):
    """
    NebulaEventHandler handles file system events for .sh scripts.

    This class monitors the creation, modification, and deletion of .sh scripts
    in a specified directory.
    """

    patterns = ["*.sh", "*.ps1"]

    def __init__(self):
        super(NebulaEventHandler, self).__init__()
        self.last_processed = {}
        self.timeout_ns = 5 * 1e9
        self.processing_files = set()
        self.lock = threading.Lock()

    def _should_process_event(self, src_path: str) -> bool:
        current_time_ns = time.time_ns()
        logging.info(f"Current time (ns): {current_time_ns}")
        with self.lock:
            if src_path in self.last_processed:
                logging.info(f"Last processed time for {src_path}: {self.last_processed[src_path]}")
                last_time = self.last_processed[src_path]
                if current_time_ns - last_time < self.timeout_ns:
                    return False
            self.last_processed[src_path] = current_time_ns
        return True

    def _is_being_processed(self, src_path: str) -> bool:
        with self.lock:
            if src_path in self.processing_files:
                logging.info(f"Skipping {src_path} as it is already being processed.")
                return True
            self.processing_files.add(src_path)
        return False

    def _processing_done(self, src_path: str):
        with self.lock:
            if src_path in self.processing_files:
                self.processing_files.remove(src_path)

    def verify_nodes_ports(self, src_path):
        parent_dir = os.path.dirname(src_path)
        base_dir = os.path.basename(parent_dir)
        scenario_path = os.path.join(os.path.dirname(parent_dir), base_dir)

        try:
            port_mapping = {}
            new_port_start = 50001
            for filename in os.listdir(scenario_path):
                if filename.endswith(".json") and filename.startswith("participant"):
                    file_path = os.path.join(scenario_path, filename)

                    with open(file_path) as json_file:
                        node = json.load(json_file)

                    current_port = node["network_args"]["port"]
                    port_mapping[current_port] = SocketUtils.find_free_port(start_port=new_port_start)
                    new_port_start = port_mapping[current_port] + 1

            for filename in os.listdir(scenario_path):
                if filename.endswith(".json") and filename.startswith("participant"):
                    file_path = os.path.join(scenario_path, filename)

                    with open(file_path) as json_file:
                        node = json.load(json_file)

                    current_port = node["network_args"]["port"]
                    node["network_args"]["port"] = port_mapping[current_port]
                    neighbors = node["network_args"]["neighbors"]

                    for old_port, new_port in port_mapping.items():
                        neighbors = neighbors.replace(f":{old_port}", f":{new_port}")

                    node["network_args"]["neighbors"] = neighbors

                    with open(file_path, "w") as f:
                        json.dump(node, f, indent=4)

        except Exception as e:
            print(f"Error processing JSON files: {e}")

    def on_created(self, event):
        """
        Handles the event when a file is created.
        """
        if event.is_directory:
            return
        src_path = event.src_path
        if not self._should_process_event(src_path):
            return
        if self._is_being_processed(src_path):
            return
        logging.info("File created: %s" % src_path)
        try:
            self.verify_nodes_ports(src_path)
            self.run_script(src_path)
        finally:
            self._processing_done(src_path)

    def on_deleted(self, event):
        """
        Handles the event when a file is deleted.
        """
        if event.is_directory:
            return
        src_path = event.src_path
        if not self._should_process_event(src_path):
            return
        if self._is_being_processed(src_path):
            return
        logging.info("File deleted: %s" % src_path)
        directory_script = os.path.dirname(src_path)
        pids_file = os.path.join(directory_script, "current_scenario_pids.txt")
        logging.info(f"Killing processes from {pids_file}")
        try:
            self.kill_script_processes(pids_file)
            os.remove(pids_file)
        except FileNotFoundError:
            logging.warning(f"{pids_file} not found.")
        except Exception as e:
            logging.exception(f"Error while killing processes: {e}")
        finally:
            self._processing_done(src_path)

    def run_script(self, script):
        try:
            logging.info(f"Running script: {script}")
            if script.endswith(".sh"):
                result = subprocess.run(["bash", script], capture_output=True, text=True)
                logging.info(f"Script output:\n{result.stdout}")
                if result.stderr:
                    logging.error(f"Script error:\n{result.stderr}")
            elif script.endswith(".ps1"):
                subprocess.Popen(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                )
            else:
                logging.error("Unsupported script format.")
                return
        except Exception as e:
            logging.exception(f"Error while running script: {e}")

    def kill_script_processes(self, pids_file):
        try:
            with open(pids_file) as f:
                pids = f.readlines()
                for pid in pids:
                    try:
                        pid = int(pid.strip())
                        if psutil.pid_exists(pid):
                            process = psutil.Process(pid)
                            children = process.children(recursive=True)
                            logging.info(f"Forcibly killing process {pid} and {len(children)} child processes...")
                            for child in children:
                                try:
                                    logging.info(f"Forcibly killing child process {child.pid}")
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    logging.warning(f"Child process {child.pid} already terminated.")
                                except Exception as e:
                                    logging.exception(f"Error while forcibly killing child process {child.pid}: {e}")
                            try:
                                logging.info(f"Forcibly killing main process {pid}")
                                process.kill()
                            except psutil.NoSuchProcess:
                                logging.warning(f"Process {pid} already terminated.")
                            except Exception as e:
                                logging.exception(f"Error while forcibly killing main process {pid}: {e}")
                        else:
                            logging.warning(f"PID {pid} does not exist.")
                    except ValueError:
                        logging.exception(f"Invalid PID value in file: {pid}")
                    except Exception as e:
                        logging.exception(f"Error while forcibly killing process {pid}: {e}")
        except FileNotFoundError:
            logging.exception(f"PID file not found: {pids_file}")
        except Exception as e:
            logging.exception(f"Error while reading PIDs from file: {e}")


class Controller:
    def __init__(self, args):
        self.scenario_name = args.scenario_name if hasattr(args, "scenario_name") else None
        self.start_date_scenario = None
        self.federation = args.federation if hasattr(args, "federation") else None
        self.topology = args.topology if hasattr(args, "topology") else None
        self.controller_port = args.controllerport if hasattr(args, "controllerport") else 5000
        self.waf_port = args.wafport if hasattr(args, "wafport") else 6000
        self.frontend_port = args.webport if hasattr(args, "webport") else 6060
        self.grafana_port = args.grafanaport if hasattr(args, "grafanaport") else 6040
        self.loki_port = args.lokiport if hasattr(args, "lokiport") else 6010
        self.statistics_port = args.statsport if hasattr(args, "statsport") else 8080
        self.simulation = args.simulation
        self.config_dir = args.config
        self.db_dir = args.databases if hasattr(args, "databases") else "/opt/nebula"
        self.test = args.test if hasattr(args, "test") else False
        self.log_dir = args.logs
        self.cert_dir = args.certs
        self.env_path = args.env
        self.production = args.production if hasattr(args, "production") else False
        self.advanced_analytics = args.advanced_analytics if hasattr(args, "advanced_analytics") else False
        self.matrix = args.matrix if hasattr(args, "matrix") else None
        self.root_path = (
            args.root_path
            if hasattr(args, "root_path")
            else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.host_platform = "windows" if sys.platform == "win32" else "unix"

        # Network configuration (nodes deployment in a network)
        self.network_subnet = args.network_subnet if hasattr(args, "network_subnet") else None
        self.network_gateway = args.network_gateway if hasattr(args, "network_gateway") else None

        # Configure logger
        self.configure_logger()

        # Check ports available
        if not SocketUtils.is_port_open(self.controller_port):
            self.controller_port = SocketUtils.find_free_port()

        if not SocketUtils.is_port_open(self.frontend_port):
            self.frontend_port = SocketUtils.find_free_port(self.controller_port + 1)

        if not SocketUtils.is_port_open(self.statistics_port):
            self.statistics_port = SocketUtils.find_free_port(self.frontend_port + 1)

        self.config = Config(entity="controller")
        self.topologymanager = None
        self.n_nodes = 0
        self.mender = None if self.simulation else Mender()
        self.use_blockchain = args.use_blockchain if hasattr(args, "use_blockchain") else False
        self.gpu_available = False

        # Reference the global app instance
        self.app = app

    def configure_logger(self):
        log_console_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(TermEscapeCodeFormatter(log_console_format))
        console_handler_file = logging.FileHandler(os.path.join(self.log_dir, "controller.log"), mode="a")
        console_handler_file.setLevel(logging.INFO)
        console_handler_file.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[
                console_handler,
                console_handler_file,
            ],
        )
        uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]
        for logger_name in uvicorn_loggers:
            logger = logging.getLogger(logger_name)
            logger.handlers = []  # Remove existing handlers
            logger.propagate = False  # Prevent duplicate logs
            handler = logging.FileHandler(os.path.join(self.log_dir, "controller.log"), mode="a")
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
            logger.addHandler(handler)

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
                              https://github.com/CyberDataLab/nebula
                    """
        print("\x1b[0;36m" + banner + "\x1b[0m")

        # Load the environment variables
        load_dotenv(self.env_path)

        # Save controller pid
        with open(os.path.join(os.path.dirname(__file__), "controller.pid"), "w") as f:
            f.write(str(os.getpid()))

        # Check information about the environment
        check_environment()

        # Save the configuration in environment variables
        logging.info("Saving configuration in environment variables...")
        os.environ["NEBULA_ROOT"] = self.root_path
        os.environ["NEBULA_LOGS_DIR"] = self.log_dir
        os.environ["NEBULA_CONFIG_DIR"] = self.config_dir
        os.environ["NEBULA_CERTS_DIR"] = self.cert_dir
        os.environ["NEBULA_STATISTICS_PORT"] = str(self.statistics_port)
        os.environ["NEBULA_ROOT_HOST"] = self.root_path
        os.environ["NEBULA_HOST_PLATFORM"] = self.host_platform

        # Start the FastAPI app in a daemon thread
        app_thread = threading.Thread(target=self.run_controller_api, daemon=True)
        app_thread.start()
        logging.info(f"NEBULA Controller is running at port {self.controller_port}")

        if self.production:
            self.run_waf()
            logging.info(f"NEBULA WAF is running at port {self.waf_port}")
            logging.info(f"Grafana Dashboard is running at port {self.grafana_port}")

        if self.test:
            self.run_test()
        else:
            self.run_frontend()
            logging.info(f"NEBULA Frontend is running at http://localhost:{self.frontend_port}")
            logging.info(f"NEBULA Databases created in {self.db_dir}")

        # Watchdog for running additional scripts in the host machine (i.e. during the execution of a federation)
        event_handler = NebulaEventHandler()
        observer = Observer()
        observer.schedule(event_handler, path=self.config_dir, recursive=True)
        observer.start()

        if self.mender:
            logging.info("[Mender.module] Mender module initialized")
            time.sleep(2)
            mender = Mender()
            logging.info("[Mender.module] Getting token from Mender server: {}".format(os.getenv("MENDER_SERVER")))
            mender.renew_token()
            time.sleep(2)
            logging.info(
                "[Mender.module] Getting devices from {} with group Cluster_Thun".format(os.getenv("MENDER_SERVER"))
            )
            time.sleep(2)
            devices = mender.get_devices_by_group("Cluster_Thun")
            logging.info("[Mender.module] Getting a pool of devices: 5 devices")
            # devices = devices[:5]
            for i in self.config.participants:
                logging.info(
                    "[Mender.module] Device {} | IP: {}".format(i["device_args"]["idx"], i["network_args"]["ip"])
                )
                logging.info("[Mender.module] \tCreating artifacts...")
                logging.info("[Mender.module] \tSending NEBULA Core...")
                # mender.deploy_artifact_device("my-update-2.0.mender", i['device_args']['idx'])
                logging.info("[Mender.module] \tSending configuration...")
                time.sleep(5)
            sys.exit(0)

        logging.info("Press Ctrl+C for exit from NEBULA (global exit)")

        # Adjust signal handling inside the start method
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Closing NEBULA (exiting from components)... Please wait")
            observer.stop()
            self.stop()

        observer.join()

    def signal_handler(self, sig, frame):
        # Handle termination signals
        logging.info("Received termination signal, shutting down...")
        self.stop()
        sys.exit(0)

    def run_controller_api(self):
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.controller_port,
            log_config=None,  # Prevent Uvicorn from configuring logging
        )

    def run_waf(self):
        network_name = f"{os.environ['USER']}_nebula-net-base"
        base = DockerUtils.create_docker_network(network_name)

        client = docker.from_env()

        volumes_waf = ["/var/log/nginx"]

        ports_waf = [80]

        host_config_waf = client.api.create_host_config(
            binds=[f"{os.environ['NEBULA_LOGS_DIR']}/waf/nginx:/var/log/nginx"],
            privileged=True,
            port_bindings={80: self.waf_port},
        )

        networking_config_waf = client.api.create_networking_config({
            f"{network_name}": client.api.create_endpoint_config(ipv4_address=f"{base}.200")
        })

        container_id_waf = client.api.create_container(
            image="nebula-waf",
            name=f"{os.environ['USER']}_nebula-waf",
            detach=True,
            volumes=volumes_waf,
            host_config=host_config_waf,
            networking_config=networking_config_waf,
            ports=ports_waf,
        )

        client.api.start(container_id_waf)

        environment = {
            "GF_SECURITY_ADMIN_PASSWORD": "admin",
            "GF_USERS_ALLOW_SIGN_UP": "false",
            "GF_SERVER_HTTP_PORT": "3000",
            "GF_SERVER_PROTOCOL": "http",
            "GF_SERVER_DOMAIN": f"localhost:{self.grafana_port}",
            "GF_SERVER_ROOT_URL": f"http://localhost:{self.grafana_port}/grafana/",
            "GF_SERVER_SERVE_FROM_SUB_PATH": "true",
            "GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH": "/var/lib/grafana/dashboards/dashboard.json",
            "GF_METRICS_MAX_LIMIT_TSDB": "0",
        }

        ports = [3000]

        host_config = client.api.create_host_config(
            port_bindings={3000: self.grafana_port},
        )

        networking_config = client.api.create_networking_config({
            f"{network_name}": client.api.create_endpoint_config(ipv4_address=f"{base}.201")
        })

        container_id = client.api.create_container(
            image="nebula-waf-grafana",
            name=f"{os.environ['USER']}_nebula-waf-grafana",
            detach=True,
            environment=environment,
            host_config=host_config,
            networking_config=networking_config,
            ports=ports,
        )

        client.api.start(container_id)

        command = ["-config.file=/mnt/config/loki-config.yml"]

        ports_loki = [3100]

        host_config_loki = client.api.create_host_config(
            port_bindings={3100: self.loki_port},
        )

        networking_config_loki = client.api.create_networking_config({
            f"{network_name}": client.api.create_endpoint_config(ipv4_address=f"{base}.202")
        })

        container_id_loki = client.api.create_container(
            image="nebula-waf-loki",
            name=f"{os.environ['USER']}_nebula-waf-loki",
            detach=True,
            command=command,
            host_config=host_config_loki,
            networking_config=networking_config_loki,
            ports=ports_loki,
        )

        client.api.start(container_id_loki)

        volumes_promtail = ["/var/log/nginx"]

        host_config_promtail = client.api.create_host_config(
            binds=[
                f"{os.environ['NEBULA_LOGS_DIR']}/waf/nginx:/var/log/nginx",
            ],
        )

        networking_config_promtail = client.api.create_networking_config({
            f"{network_name}": client.api.create_endpoint_config(ipv4_address=f"{base}.203")
        })

        container_id_promtail = client.api.create_container(
            image="nebula-waf-promtail",
            name=f"{os.environ['USER']}_nebula-waf-promtail",
            detach=True,
            volumes=volumes_promtail,
            host_config=host_config_promtail,
            networking_config=networking_config_promtail,
        )

        client.api.start(container_id_promtail)

    def run_frontend(self):
        if sys.platform == "win32":
            if not os.path.exists("//./pipe/docker_Engine"):
                raise Exception(
                    "Docker is not running, please check if Docker is running and Docker Compose is installed."
                )
        else:
            if not os.path.exists("/var/run/docker.sock"):
                raise Exception(
                    "/var/run/docker.sock not found, please check if Docker is running and Docker Compose is installed."
                )

        try:
            subprocess.check_call(["nvidia-smi"])
            self.gpu_available = True
        except Exception:
            logging.info("No GPU available for the frontend, nodes will be deploy in CPU mode")

        network_name = f"{os.environ['USER']}_nebula-net-base"

        # Create the Docker network
        base = DockerUtils.create_docker_network(network_name)

        client = docker.from_env()

        environment = {
            "NEBULA_CONTROLLER_NAME": os.environ["USER"],
            "NEBULA_PRODUCTION": self.production,
            "NEBULA_GPU_AVAILABLE": self.gpu_available,
            "NEBULA_ADVANCED_ANALYTICS": self.advanced_analytics,
            "NEBULA_FRONTEND_LOG": "/nebula/app/logs/frontend.log",
            "NEBULA_LOGS_DIR": "/nebula/app/logs/",
            "NEBULA_CONFIG_DIR": "/nebula/app/config/",
            "NEBULA_CERTS_DIR": "/nebula/app/certs/",
            "NEBULA_ENV_PATH": "/nebula/app/.env",
            "NEBULA_ROOT_HOST": self.root_path,
            "NEBULA_HOST_PLATFORM": self.host_platform,
            "NEBULA_DEFAULT_USER": "admin",
            "NEBULA_DEFAULT_PASSWORD": "admin",
            "NEBULA_FRONTEND_PORT": self.frontend_port,
            "NEBULA_CONTROLLER_PORT": self.controller_port,
            "NEBULA_CONTROLLER_HOST": "host.docker.internal",
        }

        volumes = ["/nebula", "/var/run/docker.sock", "/etc/nginx/sites-available/default"]

        ports = [80, 8080]

        host_config = client.api.create_host_config(
            binds=[
                f"{self.root_path}:/nebula",
                "/var/run/docker.sock:/var/run/docker.sock",
                f"{self.root_path}/nebula/frontend/config/nebula:/etc/nginx/sites-available/default",
                f"{self.db_dir}/databases:/nebula/nebula/frontend/databases",
            ],
            extra_hosts={"host.docker.internal": "host-gateway"},
            port_bindings={80: self.frontend_port, 8080: self.statistics_port},
        )

        networking_config = client.api.create_networking_config({
            f"{network_name}": client.api.create_endpoint_config(ipv4_address=f"{base}.100")
        })

        container_id = client.api.create_container(
            image="nebula-frontend",
            name=f"{os.environ['USER']}_nebula-frontend",
            detach=True,
            environment=environment,
            volumes=volumes,
            host_config=host_config,
            networking_config=networking_config,
            ports=ports,
        )

        client.api.start(container_id)

    def run_test(self):
        deploy_tests.start()

    @staticmethod
    def stop_waf():
        DockerUtils.remove_containers_by_prefix(f"{os.environ['USER']}_nebula-waf")

    @staticmethod
    def stop():
        logging.info("Closing NEBULA (exiting from components)... Please wait")
        DockerUtils.remove_containers_by_prefix(f"{os.environ['USER']}_")
        ScenarioManagement.stop_blockchain()
        ScenarioManagement.stop_participants()
        Controller.stop_waf()
        DockerUtils.remove_docker_networks_by_prefix(f"{os.environ['USER']}_")
        controller_pid_file = os.path.join(os.path.dirname(__file__), "controller.pid")
        try:
            with open(controller_pid_file) as f:
                pid = int(f.read())
                os.kill(pid, signal.SIGKILL)
                os.remove(controller_pid_file)
        except Exception as e:
            logging.exception(f"Error while killing controller process: {e}")
        sys.exit(0)
