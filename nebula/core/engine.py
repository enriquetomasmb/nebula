import asyncio
import logging
import os
import docker
from nebula.addons.functions import print_msg_box
from nebula.addons.attacks.attacks import create_attack
from nebula.addons.reporter import Reporter
from nebula.core.aggregation.aggregator import create_aggregator, create_malicious_aggregator, create_target_aggregator
from nebula.core.eventmanager import EventManager, event_handler
from nebula.core.network.communications import CommunicationsManager
from nebula.core.pb import nebula_pb2
from nebula.core.utils.locker import Locker
from lightning.pytorch.loggers import CSVLogger
from nebula.core.utils.nebulalogger_tensorboard import NebulaTensorBoardLogger

try:
    from nebula.core.utils.nebulalogger import NebulaLogger
except:
    pass

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("aim").setLevel(logging.ERROR)
logging.getLogger("plotly").setLevel(logging.ERROR)

from nebula.config.config import Config
from nebula.core.training.lightning import Lightning
from nebula.core.utils.helper import cosine_metric
import sys
import pdb


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    pdb.set_trace()
    pdb.post_mortem(exc_traceback)


def signal_handler(sig, frame):
    print("Signal handler called with signal", sig)
    print("Exiting gracefully")
    sys.exit(0)


def print_banner():
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
    logging.info(f"\n{banner}\n")


class Engine:
    def __init__(
        self,
        model,
        dataset,
        config=Config,
        trainer=Lightning,
        security=False,
        model_poisoning=False,
        poisoned_ratio=0,
        noise_type="gaussian",
    ):
        self.config = config
        self.idx = config.participant["device_args"]["idx"]
        self.experiment_name = config.participant["scenario_args"]["name"]
        self.ip = config.participant["network_args"]["ip"]
        self.port = config.participant["network_args"]["port"]
        self.addr = config.participant["network_args"]["addr"]
        self.role = config.participant["device_args"]["role"]
        self.name = config.participant["device_args"]["name"]
        self.docker_id = config.participant["device_args"]["docker_id"]
        self.client = docker.from_env()

        print_banner()

        print_msg_box(msg=f"Name {self.name}\nRole: {self.role}", indent=2, title="Node information")

        self._trainer = None
        self._aggregator = None
        self.round = None
        self.total_rounds = None
        self.federation_nodes = set()
        self.initialized = False
        self.log_dir = os.path.join(config.participant["tracking_args"]["log_dir"], self.experiment_name)

        self.security = security
        self.model_poisoning = model_poisoning
        self.poisoned_ratio = poisoned_ratio
        self.noise_type = noise_type

        if self.config.participant["tracking_args"]["local_tracking"] == "csv":
            nebulalogger = CSVLogger(f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}")
        elif self.config.participant["tracking_args"]["local_tracking"] == "basic":
            nebulalogger = NebulaTensorBoardLogger(self.config.participant["scenario_args"]["start_time"], f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}", log_graph=True)
        elif self.config.participant["tracking_args"]["local_tracking"] == "advanced":
            nebulalogger = NebulaLogger(
                config=self.config,
                engine=self,
                scenario_start_time=self.config.participant["scenario_args"]["start_time"],
                repo=f"{self.config.participant['tracking_args']['log_dir']}",
                experiment=self.experiment_name,
                run_name=f"participant_{self.idx}",
                train_metric_prefix="train_",
                test_metric_prefix="test_",
                val_metric_prefix="val_",
                log_system_params=False,
            )
            # nebulalogger_aim = NebulaLogger(config=self.config, engine=self, scenario_start_time=self.config.participant["scenario_args"]["start_time"], repo=f"aim://nebula-frontend:8085",
            #                                     experiment=self.experiment_name, run_name=f"participant_{self.idx}",
            #                                     train_metric_prefix='train_', test_metric_prefix='test_', val_metric_prefix='val_', log_system_params=False)
            self.config.participant["tracking_args"]["run_hash"] = nebulalogger.experiment.hash
        else:
            nebulalogger = None
        self._trainer = trainer(model, dataset, config=self.config, logger=nebulalogger)
        self._aggregator = create_aggregator(config=self.config, engine=self)

        self._secure_neighbors = []
        self._is_malicious = True if self.config.participant["adversarial_args"]["attacks"] != "No Attack" else False

        msg = f"Trainer: {self._trainer.__class__.__name__}"
        msg += f"\nDataset: {self.config.participant['data_args']['dataset']}"
        msg += f"\nIID: {self.config.participant['data_args']['iid']}"
        msg += f"\nModel: {model.__class__.__name__}"
        msg += f"\nAggregation algorithm: {self._aggregator.__class__.__name__}"
        msg += f"\nNode behavior: {'malicious' if self._is_malicious else 'benign'}"
        print_msg_box(msg=msg, indent=2, title="Scenario information")
        print_msg_box(msg=f"Logging type: {nebulalogger.__class__.__name__}", indent=2, title="Logging information")

        self.with_reputation = self.config.participant["defense_args"]["with_reputation"]
        self.is_dynamic_topology = self.config.participant["defense_args"]["is_dynamic_topology"]
        self.is_dynamic_aggregation = self.config.participant["defense_args"]["is_dynamic_aggregation"]
        self.target_aggregation = create_target_aggregator(config=self.config, engine=self) if self.is_dynamic_aggregation else None
        msg = f"Reputation system: {self.with_reputation}\nDynamic topology: {self.is_dynamic_topology}\nDynamic aggregation: {self.is_dynamic_aggregation}"
        msg += f"\nTarget aggregation: {self.target_aggregation.__class__.__name__}" if self.is_dynamic_aggregation else ""
        print_msg_box(msg=msg, indent=2, title="Defense information")

        self.learning_cycle_lock = Locker(name="learning_cycle_lock", async_lock=True)
        self.federation_ready_lock = Locker(name="federation_ready_lock", async_lock=True)
        self.round_lock = Locker(name="round_lock", async_lock=True)

        self.config.reload_config_file()

        self._cm = CommunicationsManager(engine=self)

        self._reporter = Reporter(config=self.config, trainer=self.trainer, cm=self.cm)

        self._event_manager = EventManager(
            default_callbacks=[
                self._discovery_discover_callback,
                self._control_alive_callback,
                self._connection_connect_callback,
                self._connection_disconnect_callback,
                self._start_federation_callback,
                self._federation_models_included_callback,
            ]
        )

        # Register additional callbacks
        self._event_manager.register_event((nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.REPUTATION), self._reputation_callback)
        # ... add more callbacks here

    @property
    def cm(self):
        return self._cm

    @property
    def reporter(self):
        return self._reporter

    @property
    def event_manager(self):
        return self._event_manager

    @property
    def aggregator(self):
        return self._aggregator

    def get_aggregator_type(self):
        return type(self.aggregator)

    @property
    def trainer(self):
        return self._trainer

    def get_addr(self):
        return self.addr

    def get_config(self):
        return self.config

    def get_federation_nodes(self):
        return self.federation_nodes

    def get_initialization_status(self):
        return self.initialized

    def set_initialization_status(self, status):
        self.initialized = status

    def get_round(self):
        return self.round

    def get_federation_ready_lock(self):
        return self.federation_ready_lock

    def get_round_lock(self):
        return self.round_lock

    @event_handler(nebula_pb2.DiscoveryMessage, nebula_pb2.DiscoveryMessage.Action.DISCOVER)
    async def _discovery_discover_callback(self, source, message):
        logging.info(f"🔍  handle_discovery_message | Trigger | Received discovery message from {source} (network propagation)")
        current_connections = await self.cm.get_addrs_current_connections(myself=True)
        if source not in current_connections:
            logging.info(f"🔍  handle_discovery_message | Trigger | Connecting to {source} indirectly")
            await self.cm.connect(source, direct=False)
        async with self.cm.get_connections_lock():
            if source in self.cm.connections:
                # Update the latitude and longitude of the node (if already connected)
                if message.latitude is not None and -90 <= message.latitude <= 90 and message.longitude is not None and -180 <= message.longitude <= 180:
                    self.cm.connections[source].update_geolocation(message.latitude, message.longitude)
                else:
                    logging.warning(f"🔍  Invalid geolocation received from {source}: latitude={message.latitude}, longitude={message.longitude}")

    @event_handler(nebula_pb2.ControlMessage, nebula_pb2.ControlMessage.Action.ALIVE)
    async def _control_alive_callback(self, source, message):
        logging.info(f"🔧  handle_control_message | Trigger | Received alive message from {source}")
        current_connections = await self.cm.get_addrs_current_connections(myself=True)
        if source in current_connections:
            try:
                await self.cm.health.alive(source)
            except Exception as e:
                logging.error(f"Error updating alive status in connection: {e}")
        else:
            logging.error(f"❗️  Connection {source} not found in connections...")

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.CONNECT)
    async def _connection_connect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received connection message from {source}")
        current_connections = await self.cm.get_addrs_current_connections(myself=True)
        if source not in current_connections:
            logging.info(f"🔗  handle_connection_message | Trigger | Connecting to {source}")
            await self.cm.connect(source, direct=True)

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.DISCONNECT)
    async def _connection_disconnect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received disconnection message from {source}")
        await self.cm.disconnect(source, mutual_disconnection=False)

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.FEDERATION_START)
    async def _start_federation_callback(self, source, message):
        logging.info(f"📝  handle_federation_message | Trigger | Received start federation message from {source}")
        await self.create_trainer_module()

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.REPUTATION)
    async def _reputation_callback(self, source, message):
        malicious_nodes = message.arguments  # List of malicious nodes
        if self.with_reputation:
            if len(malicious_nodes) > 0 and not self._is_malicious:
                if self.is_dynamic_topology:
                    await self._disrupt_connection_using_reputation(malicious_nodes)
                if self.is_dynamic_aggregation and self.aggregator != self.target_aggregation:
                    await self._dynamic_aggregator(self.aggregator.get_nodes_pending_models_to_aggregate(), malicious_nodes)

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.FEDERATION_MODELS_INCLUDED)
    async def _federation_models_included_callback(self, source, message):
        logging.info(f"📝  handle_federation_message | Trigger | Received aggregation finished message from {source}")
        try:
            await self.cm.get_connections_lock().acquire_async()
            if self.round is not None and source in self.cm.connections:
                try:
                    if message is not None and len(message.arguments) > 0:
                        self.cm.connections[source].update_round(int(message.arguments[0])) if message.round in [self.round - 1, self.round] else None
                except Exception as e:
                    logging.error(f"Error updating round in connection: {e}")
            else:
                logging.error(f"Connection not found for {source}")
        except Exception as e:
            logging.error(f"Error updating round in connection: {e}")
        finally:
            await self.cm.get_connections_lock().release_async()

    async def create_trainer_module(self):
        asyncio.create_task(self._start_learning())
        logging.info(f"Started trainer module...")

    async def start_communications(self):
        logging.info(f"Neighbors: {self.config.participant['network_args']['neighbors']}")
        logging.info(f"💤  Cold start time: {self.config.participant['misc_args']['grace_time_connection']} seconds before connecting to the network")
        await asyncio.sleep(self.config.participant["misc_args"]["grace_time_connection"])
        await self.cm.start()
        if self.config.participant["scenario_args"]["controller"] == "nebula-frontend":
            await self.cm.register()
            await self.cm.wait_for_controller()
        initial_neighbors = self.config.participant["network_args"]["neighbors"].split()
        for i in initial_neighbors:
            addr = f"{i.split(':')[0]}:{i.split(':')[1]}"
            await self.cm.connect(addr, direct=True)
            await asyncio.sleep(1)
        while not self.cm.verify_connections(initial_neighbors):
            await asyncio.sleep(1)
        current_connections = await self.cm.get_addrs_current_connections()
        logging.info(f"Connections verified: {current_connections}")
        await self._reporter.start()
        await self.cm.deploy_additional_services()
        await asyncio.sleep(self.config.participant["misc_args"]["grace_time_connection"] // 2)

    async def deploy_federation(self):
        await self.federation_ready_lock.acquire_async()
        if self.config.participant["device_args"]["start"]:
            logging.info(f"💤  Waiting for {self.config.participant['misc_args']['grace_time_start_federation']} seconds to start the federation")
            await asyncio.sleep(self.config.participant["misc_args"]["grace_time_start_federation"])
            if self.round is None:
                logging.info(f"Sending FEDERATION_START to neighbors...")
                message = self.cm.mm.generate_federation_message(nebula_pb2.FederationMessage.Action.FEDERATION_START)
                await self.cm.send_message_to_neighbors(message)
                await self.get_federation_ready_lock().release_async()
                await self.create_trainer_module()
            else:
                logging.info(f"Federation already started")

        else:
            logging.info(f"💤  Waiting until receiving the start signal from the start node")

    async def _start_learning(self):
        await self.learning_cycle_lock.acquire_async()
        try:
            if self.round is None:
                self.total_rounds = self.config.participant["scenario_args"]["rounds"]
                epochs = self.config.participant["training_args"]["epochs"]
                await self.get_round_lock().acquire_async()
                self.round = 0
                await self.get_round_lock().release_async()
                await self.learning_cycle_lock.release_async()
                print_msg_box(msg=f"Starting Federated Learning process...", indent=2, title="Start of the experiment")
                direct_connections = await self.cm.get_addrs_current_connections(only_direct=True)
                undirected_connections = await self.cm.get_addrs_current_connections(only_undirected=True)
                logging.info(f"Initial DIRECT connections: {direct_connections} | Initial UNDIRECT participants: {undirected_connections}")
                logging.info(f"💤  Waiting initialization of the federation...")
                # Lock to wait for the federation to be ready (only affects the first round, when the learning starts)
                # Only applies to non-start nodes --> start node does not wait for the federation to be ready
                await self.get_federation_ready_lock().acquire_async()
                if self.config.participant["device_args"]["start"]:
                    logging.info(f"Propagate initial model updates.")
                    await self.cm.propagator.propagate_continuously("initialization")
                    await self.get_federation_ready_lock().release_async()

                self.trainer.set_epochs(epochs)
                self.trainer.create_trainer()

                await self._learning_cycle()
            else:
                await self.learning_cycle_lock.release_async()
        finally:
            if self.learning_cycle_lock.locked_async():
                await self.learning_cycle_lock.release_async()

    async def _disrupt_connection_using_reputation(self, malicious_nodes):
        malicious_nodes = list(set(malicious_nodes) & set(self.get_current_connections()))
        logging.info(f"Disrupting connection with malicious nodes at round {self.round}")
        logging.info(f"Removing {malicious_nodes} from {self.get_current_connections()}")
        logging.info(f"Current connections before aggregation at round {self.round}: {self.get_current_connections()}")
        for malicious_node in malicious_nodes:
            if (self.get_name() != malicious_node) and (malicious_node not in self._secure_neighbors):
                await self.cm.disconnect(malicious_node)
        logging.info(f"Current connections after aggregation at round {self.round}: {self.get_current_connections()}")

        await self._connect_with_benign(malicious_nodes)

    async def _connect_with_benign(self, malicious_nodes):
        lower_threshold = 1
        higher_threshold = len(self.federation_nodes) - 1
        if higher_threshold < lower_threshold:
            higher_threshold = lower_threshold

        benign_nodes = [i for i in self.federation_nodes if i not in malicious_nodes]
        logging.info(f"_reputation_callback benign_nodes at round {self.round}: {benign_nodes}")
        if len(self.get_current_connections()) <= lower_threshold:
            for node in benign_nodes:
                if len(self.get_current_connections()) <= higher_threshold and self.get_name() != node:
                    connected = await self.cm.connect(node)
                    if connected:
                        logging.info(f"Connect new connection with at round {self.round}: {connected}")

    async def _dynamic_aggregator(self, aggregated_models_weights, malicious_nodes):
        logging.info(f"malicious detected at round {self.round}, change aggergation protocol!")
        if self.aggregator != self.target_aggregation:
            logging.info(f"Current aggregator is: {self.aggregator}")
            self.aggregator = self.target_aggregation
            await self.aggregator.update_federation_nodes(self.federation_nodes)

            for subnodes in aggregated_models_weights.keys():
                sublist = subnodes.split()
                (submodel, weights) = aggregated_models_weights[subnodes]
                for node in sublist:
                    if node not in malicious_nodes:
                        await self.aggregator.include_model_in_buffer(submodel, weights, source=self.get_name(), round=self.round)
            logging.info(f"Current aggregator is: {self.aggregator}")

    async def _waiting_model_updates(self):
        logging.info(f"💤  Waiting convergence in round {self.round}.")
        params = await self.aggregator.get_aggregation()
        if params is not None:
            logging.info(f"_waiting_model_updates | Aggregation done for round {self.round}, including parameters in local model.")
            self.trainer.set_model_parameters(params)
        else:
            logging.error(f"Aggregation finished with no parameters")

    async def _learning_cycle(self):
        while self.round is not None and self.round < self.total_rounds:
            print_msg_box(msg=f"Round {self.round} of {self.total_rounds} started.", indent=2, title="Round information")
            self.trainer.on_round_start()
            self.federation_nodes = await self.cm.get_addrs_current_connections(only_direct=True, myself=True)
            logging.info(f"Federation nodes: {self.federation_nodes}")
            direct_connections = await self.cm.get_addrs_current_connections(only_direct=True)
            undirected_connections = await self.cm.get_addrs_current_connections(only_undirected=True)
            logging.info(f"Direct connections: {direct_connections} | Undirected connections: {undirected_connections}")
            logging.info(f"[Role {self.role}] Starting learning cycle...")
            await self.aggregator.update_federation_nodes(self.federation_nodes)
            await self._extended_learning_cycle()

            await self.get_round_lock().acquire_async()
            print_msg_box(msg=f"Round {self.round} of {self.total_rounds} finished.", indent=2, title="Round information")
            self.aggregator.reset()
            self.trainer.on_round_end()
            self.round = self.round + 1
            self.config.participant["federation_args"]["round"] = self.round  # Set current round in config (send to the controller)
            await self.get_round_lock().release_async()

        # End of the learning cycle
        self.trainer.on_learning_cycle_end()
        logging.info(f"[Testing] Starting final testing...")
        self.trainer.test()
        logging.info(f"[Testing] Finishing final testing...")
        self.round = None
        self.total_rounds = None
        await self.get_federation_ready_lock().acquire_async()
        print_msg_box(msg=f"Federated Learning process has been completed.", indent=2, title="End of the experiment")
        # Enable loggin info
        logging.getLogger().disabled = True
        # Report
        if self.config.participant["scenario_args"]["controller"] == "nebula-frontend":
            await self.reporter.report_scenario_finished()
        # Kill itself
        try:
            self.client.containers.get(self.docker_id).stop()
            print(f"Docker container with ID {self.docker_id} stopped successfully.")
        except Exception as e:
            print(f"Error stopping Docker container with ID {self.docker_id}: {e}")

    async def _extended_learning_cycle(self):
        """
        This method is called in each round of the learning cycle. It is used to extend the learning cycle with additional
        functionalities. The method is called in the _learning_cycle method.
        """
        pass

    def reputation_calculation(self, aggregated_models_weights):
        cossim_threshold = 0.5
        loss_threshold = 0.5

        current_models = {}
        for subnodes in aggregated_models_weights.keys():
            sublist = subnodes.split()
            submodel = aggregated_models_weights[subnodes][0]
            for node in sublist:
                current_models[node] = submodel

        malicious_nodes = []
        reputation_score = {}
        local_model = self.trainer.get_model_parameters()
        untrusted_nodes = list(current_models.keys())
        logging.info(f"reputation_calculation untrusted_nodes at round {self.round}: {untrusted_nodes}")

        for untrusted_node in untrusted_nodes:
            logging.info(f"reputation_calculation untrusted_node at round {self.round}: {untrusted_node}")
            logging.info(f"reputation_calculation self.get_name() at round {self.round}: {self.get_name()}")
            if untrusted_node != self.get_name():
                untrusted_model = current_models[untrusted_node]
                cossim = cosine_metric(local_model, untrusted_model, similarity=True)
                logging.info(f"reputation_calculation cossim at round {self.round}: {untrusted_node}: {cossim}")
                self.trainer._logger.log_data({f"Reputation/cossim_{untrusted_node}": cossim}, step=self.round)

                avg_loss = self.trainer.validate_neighbour_model(untrusted_model)
                logging.info(f"reputation_calculation avg_loss at round {self.round} {untrusted_node}: {avg_loss}")
                self.trainer._logger.log_data({f"Reputation/avg_loss_{untrusted_node}": avg_loss}, step=self.round)
                reputation_score[untrusted_node] = (cossim, avg_loss)

                if cossim < cossim_threshold or avg_loss > loss_threshold:
                    malicious_nodes.append(untrusted_node)
                else:
                    self._secure_neighbors.append(untrusted_node)

        return malicious_nodes, reputation_score

    async def send_reputation(self, malicious_nodes):
        logging.info(f"Sending REPUTATION to the rest of the topology: {malicious_nodes}")
        message = self.cm.mm.generate_federation_message(nebula_pb2.FederationMessage.Action.REPUTATION, malicious_nodes)
        await self.cm.send_message_to_neighbors(message)


class MaliciousNode(Engine):

    def __init__(self, model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type="gaussian"):
        super().__init__(model, dataset, config, trainer, security, model_poisoning, poisoned_ratio, noise_type)
        self.attack = create_attack(config.participant["adversarial_args"]["attacks"])
        self.fit_time = 0.0
        self.extra_time = 0.0

        self.round_start_attack = 3
        self.round_stop_attack = 6

        self.aggregator_bening = self._aggregator

    async def _extended_learning_cycle(self):
        if self.round in range(self.round_start_attack, self.round_stop_attack):
            logging.info(f"Changing aggregation function maliciously...")
            self._aggregator = create_malicious_aggregator(self._aggregator, self.attack)
        elif self.round == self.round_stop_attack:
            logging.info(f"Changing aggregation function benignly...")
            self._aggregator = self.aggregator_bening

        await AggregatorNode._extended_learning_cycle(self)


class AggregatorNode(Engine):
    def __init__(self, model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type="gaussian"):
        super().__init__(model, dataset, config, trainer, security, model_poisoning, poisoned_ratio, noise_type)

    async def _extended_learning_cycle(self):
        # Define the functionality of the aggregator node
        logging.info(f"[Testing] Starting...")
        self.trainer.test()
        logging.info(f"[Testing] Finishing...")

        logging.info(f"[Training] Starting...")
        self.trainer.train()
        logging.info(f"[Training] Finishing...")

        await self.aggregator.include_model_in_buffer(self.trainer.get_model_parameters(), self.trainer.get_model_weight(), source=self.addr, round=self.round)

        await self.cm.propagator.propagate_continuously("stable")
        await self._waiting_model_updates()


class ServerNode(Engine):
    def __init__(self, model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type="gaussian"):
        super().__init__(model, dataset, config, trainer, security, model_poisoning, poisoned_ratio, noise_type)

    async def _extended_learning_cycle(self):
        # Define the functionality of the server node
        logging.info(f"[Testing] Starting...")
        self.trainer.test()
        logging.info(f"[Testing] Finishing...")

        # In the first round, the server node doest take into account the initial model parameters for the aggregation
        await self.aggregator.include_model_in_buffer(self.trainer.get_model_parameters(), self.trainer.BYPASS_MODEL_WEIGHT, source=self.addr, round=self.round)
        await self._waiting_model_updates()
        await self.cm.propagator.propagate_continuously("stable")


class TrainerNode(Engine):
    def __init__(self, model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type="gaussian"):
        super().__init__(model, dataset, config, trainer, security, model_poisoning, poisoned_ratio, noise_type)

    async def _extended_learning_cycle(self):
        # Define the functionality of the trainer node
        logging.info(f"Waiting global update | Assign _waiting_global_update = True")
        self.aggregator.set_waiting_global_update()

        logging.info(f"[Testing] Starting...")
        self.trainer.test()
        logging.info(f"[Testing] Finishing...")

        logging.info(f"[Training] Starting...")
        self.trainer.train()
        logging.info(f"[Training] Finishing...")

        await self.aggregator.include_model_in_buffer(self.trainer.get_model_parameters(), self.trainer.get_model_weight(), source=self.addr, round=self.round, local=True)

        await self.cm.propagator.propagate_continuously("stable")
        await self._waiting_model_updates()


class IdleNode(Engine):
    def __init__(self, model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type="gaussian"):
        super().__init__(model, dataset, config, trainer, security, model_poisoning, poisoned_ratio, noise_type)

    async def _extended_learning_cycle(self):
        # Define the functionality of the idle node
        logging.info(f"Waiting global update | Assign _waiting_global_update = True")
        self.aggregator.set_waiting_global_update()
        await self._waiting_model_updates()
