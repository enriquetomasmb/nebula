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
from nebula.core.utils.nebulalogger_tensorboard import NebulaTensorBoardLogger
from nebula.core.utils.nebulalogger import NebulaLogger
from nebula.core.utils.locker import Locker
from nebula.core.neighbormanagement.nodemanager import NodeManager

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("aim").setLevel(logging.ERROR)
logging.getLogger("plotly").setLevel(logging.ERROR)

import threading

from lightning.pytorch.loggers import CSVLogger

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
            nebulalogger = NebulaLogger(config=self.config, engine=self, scenario_start_time=self.config.participant["scenario_args"]["start_time"], repo=f"{self.config.participant['tracking_args']['log_dir']}",
                                                experiment=self.experiment_name, run_name=f"participant_{self.idx}",
                                                train_metric_prefix='train_', test_metric_prefix='test_', val_metric_prefix='val_', log_system_params=False)
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

        self.learning_cycle_lock = Locker(name="learning_cycle_lock")
        self.federation_ready_lock = Locker(name="federation_ready_lock")
        self.federation_ready_lock.acquire()
        self.round_lock = Locker(name="round_lock")

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
        
        self._event_manager.register_event((nebula_pb2.DiscoverMessage, nebula_pb2.DiscoverMessage.Action.DISCOVER_JOIN), self._discover_discover_join_callback)
        self._event_manager.register_event((nebula_pb2.DiscoverMessage, nebula_pb2.DiscoverMessage.Action.DISCOVER_NODE), self._discover_discover_node_callback)
        
        self._event_manager.register_event((nebula_pb2.OfferMessage, nebula_pb2.OfferMessage.Action.OFFER_METRIC), self._offer_offer_metric_callback)
        self._event_manager.register_event((nebula_pb2.OfferMessage, nebula_pb2.OfferMessage.Action.OFFER_MODEL), self._offer_offer_model_callback)
        
        self._event_manager.register_event((nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.LATE_CONNECT), self._connection_late_connect_callback)
        self._event_manager.register_event((nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.RESTRUCTURE), self._connection_late_connect_callback)
        
        self._event_manager.register_event((nebula_pb2.LinkMessage, nebula_pb2.LinkMessage.Action.CONNECT_TO), self._link_connect_to_callback)
        self._event_manager.register_event((nebula_pb2.LinkMessage, nebula_pb2.LinkMessage.Action.DISCONNECT_FROM), self._link_disconnect_from_callback)
        # ...

        # Thread for the trainer service, it is created when the learning starts
        self.trainer_service = None
        
        if self.config.participant["mobility_args"]["mobility"]:
            topology = self.config.participant["mobility_args"]["mobility_type"]
            model_handler = self.config.participant["mobility_args"]["model_handler"]
            self._node_manager = NodeManager(topology, model_handler, engine=self)
            if self.config.participant["mobility_args"]["late_creation"]:
                self._init_late_node()
            

    @property
    def cm(self):
        return self._cm

    @property
    def nm(self):
        return self._node_manager
    
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
        if source not in self.cm.get_addrs_current_connections(myself=True):
            logging.info(f"🔍  handle_discovery_message | Trigger | Connecting to {source} indirectly")
            await self.cm.connect(source, direct=False)
        with self.cm.get_connections_lock():
            if source in self.cm.connections:
                # Update the latitude and longitude of the node (if already connected)
                if message.latitude is not None and -90 <= message.latitude <= 90 and message.longitude is not None and -180 <= message.longitude <= 180:
                    self.cm.connections[source].update_geolocation(message.latitude, message.longitude)
                else:
                    logging.warning(f"🔍  Invalid geolocation received from {source}: latitude={message.latitude}, longitude={message.longitude}")

    @event_handler(nebula_pb2.ControlMessage, nebula_pb2.ControlMessage.Action.ALIVE)
    async def _control_alive_callback(self, source, message):
        logging.info(f"🔧  handle_control_message | Trigger | Received alive message from {source}")
        if source in self.cm.get_addrs_current_connections(myself=True):
            try:
                await self.cm.health.alive(source)
            except Exception as e:
                logging.error(f"Error updating alive status in connection: {e}")
        else:
            logging.error(f"❗️  Connection {source} not found in connections...")

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.CONNECT)
    async def _connection_connect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received connection message from {source}")
        if source not in self.cm.get_addrs_current_connections(myself=True):
            logging.info(f"🔗  handle_connection_message | Trigger | Connecting to {source}")
            await self.cm.connect(source, direct=True)
            self.nm.update_neighbors(source)

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.DISCONNECT)
    async def _connection_disconnect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received disconnection message from {source}")
        await self.cm.disconnect(source, mutual_disconnection=False)
        self.nm.update_neighbors(source, remove=True)

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.FEDERATION_START)
    async def _start_federation_callback(self, source, message):
        logging.info(f"📝  handle_federation_message | Trigger | Received start federation message from {source}")
        self.create_trainer_service()

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.REPUTATION)
    async def _reputation_callback(self, source, message):
        malicious_nodes = message.arguments  # List of malicious nodes
        if self.with_reputation:
            if len(malicious_nodes) > 0 and not self._is_malicious:
                if self.is_dynamic_topology:
                    self._disrupt_connection_using_reputation(malicious_nodes)
                if self.is_dynamic_aggregation and self.aggregator != self.target_aggregation:
                    await self._dynamic_aggregator(self.aggregator.get_nodes_pending_models_to_aggregate(), malicious_nodes)

    @event_handler(nebula_pb2.FederationMessage, nebula_pb2.FederationMessage.Action.FEDERATION_MODELS_INCLUDED)
    def _federation_models_included_callback(self, source, message):
        logging.info(f"📝  handle_federation_message | Trigger | Received aggregation finished message from {source}")
        try:
            self.cm.get_connections_lock().acquire()
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
            self.cm.get_connections_lock().release()

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.LATE_CONNECT)
    async def _connection_late_connect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received late_connect message from {source}")   
        if self.nm.accept_connection(source, joining=True):
            self.nm.add_weight_modifier(source) 
            ct_actions , df_actions = self.nm.get_actions()
            
            # connect to            
            for addr in ct_actions.split():
                cnt_msg = self.cm.mm.generate_link_message(nebula_pb2.LinkMessage.Action.CONNECTO_TO, addr)
                await self.cm.send_message(source, cnt_msg)
            
            # disconnect from
            for addr in df_actions.split():
                df_msg = self.cm.mm.generate_link_message(nebula_pb2.LinkMessage.Action.DISCONNECT_FROM, addr)
                await self.cm.send_message(source, df_msg)

            await self.cm.connect(source, direct=True)
            self.nm.update_neighbors(source)

    @event_handler(nebula_pb2.ConnectionMessage, nebula_pb2.ConnectionMessage.Action.RESTRUCTURE)
    async def _connection_disconnect_callback(self, source, message):
        logging.info(f"🔗  handle_connection_message | Trigger | Received restructure message from {source}")
        if self.nm.accept_connection(source, joining=False):
            logging.info(f"🔗  handle_connection_message | Trigger | restructure connection accepted from {source}")
            ct_actions , df_actions = self.nm.get_actions()
                        
            for addr in ct_actions.split():
                cnt_msg = self.cm.mm.generate_link_message(nebula_pb2.LinkMessage.Action.CONNECTO_TO, addr)
                await self.cm.send_message(source, cnt_msg)
            
            for addr in df_actions.split():
                df_msg = self.cm.mm.generate_link_message(nebula_pb2.LinkMessage.Action.DISCONNECT_FROM, addr)
                await self.cm.send_message(source, df_msg)      
        else:
            logging.info(f"🔗  handle_connection_message | Trigger | restructure connection denied from {source}")
            await self.cm.disconnect(source, mutual_disconnection=False)
            self.nm.update_neighbors(source, remove=True) 

    @event_handler(nebula_pb2.DiscoverMessage, nebula_pb2.DiscoverMessage.Action.DISCOVER_JOIN)
    async def _discover_discover_join_callback(self, source, message):
        logging.info(f"🔍  handle_discover_message | Trigger | Received discover_join message from {source} ")
        
        self.nm.meet_node(source)
        # if no neighbors means i'm new
        if len(self.get_federation_nodes()) > 0:
            model, rounds, round = self.cm.propagator.get_model_information(source, "stable") if self.get_round() > 0 else self.cm.propagator.get_model_information(source, "initialization")
            epochs = self.config.participant["training_args"]["epochs"]
            msg = self.cm.mm.generate_offer_message(
                nebula_pb2.OfferMessage.Action.OFFER_MODEL, 
                len(self.get_federation_nodes()), 
                self.trainer.get_loss(),
                model,
                rounds,
                round,
                epochs
            )
            await self.cm.send_message(source, msg)

    @event_handler(nebula_pb2.DiscoverMessage, nebula_pb2.DiscoverMessage.Action.DISCOVER_NODE)
    async def _discover_discover_node_callback(self, source, message):
        logging.info(f"🔍  handle_discover_message | Trigger | Received discover_node message from {source} ")
        self.nm.meet_node(source)
        msg = self.cm.mm.generate_offer_message(nebula_pb2.OfferMessage.Action.OFFER_METRIC, len(self.get_federation_nodes()), self.trainer.get_loss())
        await self.cm.send_message(source, msg)
      
    @event_handler(nebula_pb2.OfferMessage, nebula_pb2.OfferMessage.Action.OFFER_MODEL)
    async def _offer_offer_model_callback(self, source, message):
        logging.info(f"🔍  handle_offer_message | Trigger | Received offer_model message from {source}")
        if not self.nm.get_restructure_process_lock().locked():
            decoded_model = self.trainer.deserialize_model(message.parameters)
            self.nm.accept_model(source, decoded_model, message.rounds, message.round, message.epochs, message.n_neighbors, message.loss)
            self.nm.add_candidate(source, message.n_neighbors, message.loss)
            self.nm.meet_node(source)
        
    @event_handler(nebula_pb2.OfferMessage, nebula_pb2.OfferMessage.Action.OFFER_METRIC)
    async def _offer_offer_metric_callback(self, source, message):
        logging.info(f"🔍  handle_offer_message | Trigger | Received offer_metric message from {source}")
        if not self.nm.get_restructure_process_lock().locked():
            n_neighbors, loss, _, _, _, _ = message.arguments
            self.nm.add_candidate(source, n_neighbors, loss)
            self.nm.meet_node(source)

    @event_handler(nebula_pb2.LinkMessage, nebula_pb2.LinkMessage.Action.CONNECTO_TO)
    async def _link_connect_to_callback(self, source, message):
        logging.info(f"🔗  handle_link_message | Trigger | Received connecto_to message from {source}")
        addrs = message.arguments
        for addr in addrs:
            await self.cm.connect(addr, direct=True)
            self.nm.update_neighbors(addr)
            self.nm.meet_node(source)
            
    @event_handler(nebula_pb2.LinkMessage, nebula_pb2.LinkMessage.Action.DISCONNECT_FROM)
    async def _link_disconnect_from_callback(self, source, message):
        logging.info(f"🔗  handle_link_message | Trigger | Received disconnect_from message from {source}")
        addrs = message.arguments
        for addr in addrs:
            await self.cm.disconnect(source, mutual_disconnection=False)
            self.nm.update_neighbors(addr, remove=True)

    def create_trainer_service(self, round=0):
        if self.trainer_service is None:
            self.trainer_service = threading.Thread(
                target=self._start_learning,
                args=(round,),
                daemon=True,
                name="trainer_service_thread-" + self.addr,
            )
            self.trainer_service.start()
            logging.info(f"Started trainer service thread...")

    def get_trainer_service(self):
        return self.trainer_service

    async def start_communications(self):
        logging.info(f"Neighbors: {self.config.participant['network_args']['neighbors']}")
        logging.info(f"💤  Cold start time: {self.config.participant['misc_args']['grace_time_connection']} seconds before connecting to the network")
        await asyncio.sleep(self.config.participant["misc_args"]["grace_time_connection"])
        await self.cm.start()
        await self.cm.register()
        await self.cm.wait_for_controller()
        initial_neighbors = self.config.participant["network_args"]["neighbors"].split()
        for i in initial_neighbors:
            addr = f"{i.split(':')[0]}:{i.split(':')[1]}"
            await self.cm.connect(addr, direct=True)
            await asyncio.sleep(1)
        while not self.cm.verify_connections(initial_neighbors):
            await asyncio.sleep(1)
        logging.info(f"Connections verified: {self.cm.get_addrs_current_connections()}")
        self._reporter.start()
        await self.cm.deploy_additional_services()
        await asyncio.sleep(self.config.participant["misc_args"]["grace_time_connection"] // 2)

    async def deploy_federation(self):
        if self.config.participant["device_args"]["start"]:
            logging.info(f"💤  Waiting for {self.config.participant['misc_args']['grace_time_start_federation']} seconds to start the federation")
            await asyncio.sleep(self.config.participant["misc_args"]["grace_time_start_federation"])
            if self.round is None:
                logging.info(f"Sending FEDERATION_START to neighbors...")
                message = self.cm.mm.generate_federation_message(nebula_pb2.FederationMessage.Action.FEDERATION_START)
                await self.cm.send_message_to_neighbors(message)
                self.get_federation_ready_lock().release()
                self.create_trainer_service()
            else:
                logging.info(f"Federation already started")

        else:
            logging.info(f"💤  Waiting until receiving the start signal from the start node")

    def _start_learning(self, round=0):
        self.learning_cycle_lock.acquire()
        try:
            if self.round is None:
                self.total_rounds = self.config.participant["scenario_args"]["rounds"]
                epochs = self.config.participant["training_args"]["epochs"]
                self.get_round_lock().acquire()
                self.round = round
                self.get_round_lock().release()
                self.learning_cycle_lock.release()
                print_msg_box(msg=f"Starting Federated Learning process...", indent=2, title="Start of the experiment")
                logging.info(f"Initial DIRECT connections: {self.cm.get_addrs_current_connections(only_direct=True)} | Initial UNDIRECT participants: {self.cm.get_addrs_current_connections(only_undirected=True)}")
                logging.info(f"💤  Waiting initialization of the federation...")
                # Lock to wait for the federation to be ready (only affects the first round, when the learning starts)
                # Only applies to non-start nodes --> start node does not wait for the federation to be ready
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                self.get_federation_ready_lock().acquire()
                if self.config.participant["device_args"]["start"]:
                    logging.info(f"Propagate initial model updates.")
                    loop.run_until_complete(self.cm.propagator.propagate_continuously("initialization"))
                    self.get_federation_ready_lock().release()

                self.trainer.set_epochs(epochs)
                self.trainer.create_trainer()

                loop.run_until_complete(self._learning_cycle())
            else:
                self.learning_cycle_lock.release()
        finally:
            loop.close()

    def _disrupt_connection_using_reputation(self, malicious_nodes):
        malicious_nodes = list(set(malicious_nodes) & set(self.get_current_connections()))
        logging.info(f"Disrupting connection with malicious nodes at round {self.round}")
        logging.info(f"Removing {malicious_nodes} from {self.get_current_connections()}")
        logging.info(f"Current connections before aggregation at round {self.round}: {self.get_current_connections()}")
        for malicious_node in malicious_nodes:
            if (self.get_name() != malicious_node) and (malicious_node not in self._secure_neighbors):
                self.cm.disconnect(malicious_node)
        logging.info(f"Current connections after aggregation at round {self.round}: {self.get_current_connections()}")

        self._connect_with_benign(malicious_nodes)

    def _connect_with_benign(self, malicious_nodes):
        lower_threshold = 1
        higher_threshold = len(self.federation_nodes) - 1
        if higher_threshold < lower_threshold:
            higher_threshold = lower_threshold

        benign_nodes = [i for i in self.federation_nodes if i not in malicious_nodes]
        logging.info(f"_reputation_callback benign_nodes at round {self.round}: {benign_nodes}")
        if len(self.get_current_connections()) <= lower_threshold:
            for node in benign_nodes:
                if len(self.get_current_connections()) <= higher_threshold and self.get_name() != node:
                    connected = self.cm.connect(node)
                    if connected:
                        logging.info(f"Connect new connection with at round {self.round}: {connected}")

    async def _dynamic_aggregator(self, aggregated_models_weights, malicious_nodes):
        logging.info(f"malicious detected at round {self.round}, change aggergation protocol!")
        if self.aggregator != self.target_aggregation:
            logging.info(f"Current aggregator is: {self.aggregator}")
            self.aggregator = self.target_aggregation
            self.aggregator.update_federation_nodes(self.federation_nodes)

            for subnodes in aggregated_models_weights.keys():
                sublist = subnodes.split()
                (submodel, weights) = aggregated_models_weights[subnodes]
                for node in sublist:
                    if node not in malicious_nodes:
                        await self.aggregator.include_model_in_buffer(submodel, weights, source=self.get_name(), round=self.round)
            logging.info(f"Current aggregator is: {self.aggregator}")

    async def _waiting_model_updates(self):
        logging.info(f"💤  Waiting convergence in round {self.round}.")
        params = self.aggregator.get_aggregation()
        if params is not None:
            logging.info(f"_waiting_model_updates | Aggregation done for round {self.round}, including parameters in local model.")
            self.trainer.set_model_parameters(params)
        else:
            logging.error(f"Aggregation finished with no parameters")

    async def _learning_cycle(self):
        while self.round is not None and self.round < self.total_rounds:
            print_msg_box(msg=f"Round {self.round} of {self.total_rounds} started.", indent=2, title="Round information")
            self.trainer.on_round_start()
            self.federation_nodes = self.cm.get_addrs_current_connections(only_direct=True, myself=True)
            logging.info(f"Federation nodes: {self.federation_nodes}")
            logging.info(f"Direct connections: {self.cm.get_addrs_current_connections(only_direct=True)} | Undirected connections: {self.cm.get_addrs_current_connections(only_undirected=True)}")
            logging.info(f"[Role {self.role}] Starting learning cycle...")
            self.aggregator.update_federation_nodes(self.federation_nodes)
            await self._extended_learning_cycle()

            self.get_round_lock().acquire()
            print_msg_box(msg=f"Round {self.round} of {self.total_rounds} finished.", indent=2, title="Round information")
            self.aggregator.reset()
            self.trainer.on_round_end()
            self.round = self.round + 1
            self.config.participant["federation_args"]["round"] = self.round  # Set current round in config (send to the controller)
            self.get_round_lock().release()

        # End of the learning cycle
        self.trainer.on_learning_cycle_end()
        logging.info(f"[Testing] Starting final testing...")
        self.trainer.test()
        logging.info(f"[Testing] Finishing final testing...")
        self.round = None
        self.total_rounds = None
        self.get_federation_ready_lock().acquire()
        print_msg_box(msg=f"Federated Learning process has been completed.", indent=2, title="End of the experiment")
        # Report 
        self.reporter.report_scenario_finished()
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


    def _init_late_node(self):
        """
            Method to initialize a late connected node, creating its trainer and setting up the learning process

            First step broadcasting discover message, after that we select candidates and connect to them.
            The information to create the trainer is recieved from nodes that are already on federation and answared the discover message.
                -model:     params
                -rounds:    total rounds
                -round:     current round of the learning process
                -epochs:    epochs
        """
        logging.info("🌐  Initializing late creation node life from Engine")
        model, rounds, round, epochs = self.nm.start_late_connection_process()
           
        self.config.participant["scenario_args"]["rounds"] = rounds
        self.config.participant["training_args"]["epochs"] = epochs
        
        self.round = round
        
        #self._trainer = trainer(model, self.dataset, config=self.config, logger=nebulalogger)
        self.trainer.set_model_parameters(model, initialize=True)
        
        self.set_initialization_status(True)
        self.get_federation_ready_lock().release()
        self._create_trainer_service(round=round)
        self.cm.start_external_connection_service()
        
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
