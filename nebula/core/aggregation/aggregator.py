from abc import ABC, abstractmethod
from functools import partial
import logging
from nebula.core.utils.locker import Locker
from nebula.core.pb import nebula_pb2


class AggregatorException(Exception):
    pass


def create_aggregator(config, engine):
    from nebula.core.aggregation.fedavg import FedAvg
    from nebula.core.aggregation.krum import Krum
    from nebula.core.aggregation.median import Median
    from nebula.core.aggregation.trimmedmean import TrimmedMean
    from nebula.core.aggregation.blockchainReputation import BlockchainReputation

    ALGORITHM_MAP = {
        "FedAvg": FedAvg,
        "Krum": Krum,
        "Median": Median,
        "TrimmedMean": TrimmedMean,
        "BlockchainReputation": BlockchainReputation,
    }
    algorithm = config.participant["aggregator_args"]["algorithm"]
    aggregator = ALGORITHM_MAP.get(algorithm)
    if aggregator:
        return aggregator(config=config, engine=engine)
    else:
        raise AggregatorException(f"Aggregation algorithm {algorithm} not found.")


def create_target_aggregator(config, engine):
    from nebula.core.aggregation.fedavg import FedAvg
    from nebula.core.aggregation.krum import Krum
    from nebula.core.aggregation.median import Median
    from nebula.core.aggregation.trimmedmean import TrimmedMean

    ALGORITHM_MAP = {
        "FedAvg": FedAvg,
        "Krum": Krum,
        "Median": Median,
        "TrimmedMean": TrimmedMean,
    }
    algorithm = config.participant["defense_args"]["target_aggregation"]
    aggregator = ALGORITHM_MAP.get(algorithm)
    if aggregator:
        return aggregator(config=config, engine=engine)
    else:
        raise AggregatorException(f"Aggregation algorithm {algorithm} not found.")


class Aggregator(ABC):
    def __init__(self, config=None, engine=None):
        self.config = config
        self.engine = engine
        self._addr = config.participant["network_args"]["addr"]
        logging.info(f"[{self.__class__.__name__}] Starting Aggregator")
        self._federation_nodes = set()
        self._waiting_global_update = False
        self._pending_models_to_aggregate = {}
        self._add_model_lock = Locker(name="add_model_lock")
        self._aggregation_done_lock = Locker(name="aggregation_done_lock")

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    @property
    def cm(self):
        return self.engine.cm

    @abstractmethod
    def run_aggregation(self, models):
        if len(models) == 0:
            logging.error("Trying to aggregate models when there are no models")
            return None

    def update_federation_nodes(self, federation_nodes):
        if not self._aggregation_done_lock.locked():
            self._federation_nodes = federation_nodes
            self._pending_models_to_aggregate.clear()
            self._aggregation_done_lock.acquire(timeout=self.config.participant["aggregator_args"]["aggregation_timeout"])
        else:
            raise Exception("It is not possible to set nodes to aggregate when the aggregation is running.")

    def set_waiting_global_update(self):
        self._waiting_global_update = True

    def reset(self):
        self._add_model_lock.acquire()
        self._federation_nodes.clear()
        self._pending_models_to_aggregate.clear()
        try:
            self._aggregation_done_lock.release()
        except:
            pass
        self._add_model_lock.release()

    def get_nodes_pending_models_to_aggregate(self):
        return {node for key in self._pending_models_to_aggregate.keys() for node in key.split()}

    def _handle_global_update(self, model, source):
        logging.info(f"🔄  _handle_global_update | source={source}")
        logging.info(f"🔄  _handle_global_update | Received a model from {source}. Overwriting __models with the aggregated model.")
        self._pending_models_to_aggregate.clear()
        self._pending_models_to_aggregate = {source: (model, 1)}
        self._waiting_global_update = False
        self._add_model_lock.release()
        self._aggregation_done_lock.release()

    def _add_pending_model(self, model, weight, source):
        if len(self._federation_nodes) <= len(self.get_nodes_pending_models_to_aggregate()):
            logging.info(f"🔄  _add_pending_model | Ignoring model...")
            self._add_model_lock.release()
            return None

        if source not in self._federation_nodes:
            logging.info(f"🔄  _add_pending_model | Can't add a model from ({source}), which is not in the federation.")
            self._add_model_lock.release()
            return None

        elif source not in self.get_nodes_pending_models_to_aggregate():
            logging.info(f"🔄  _add_pending_model | Node is not in the aggregation buffer --> Include model in the aggregation buffer.")
            self._pending_models_to_aggregate.update({source: (model, weight)})

        logging.info(f"🔄  _add_pending_model | Model added in aggregation buffer ({str(len(self.get_nodes_pending_models_to_aggregate()))}/{str(len(self._federation_nodes))}) | Pending nodes: {self._federation_nodes - self.get_nodes_pending_models_to_aggregate()}")
        if len(self.get_nodes_pending_models_to_aggregate()) >= len(self._federation_nodes):
            logging.info(f"🔄  _add_pending_model | All models were added in the aggregation buffer. Run aggregation...")
            self._aggregation_done_lock.release()
        self._add_model_lock.release()
        return self.get_nodes_pending_models_to_aggregate()

    async def include_model_in_buffer(self, model, weight, source=None, round=None, local=False):
        self._add_model_lock.acquire()
        logging.info(f"🔄  include_model_in_buffer | source={source} | round={round} | weight={weight} |--| __models={self._pending_models_to_aggregate.keys()} | federation_nodes={self._federation_nodes} | pending_models_to_aggregate={self.get_nodes_pending_models_to_aggregate()}")
        if model is None:
            logging.info(f"🔄  include_model_in_buffer | Ignoring model bad formed...")
            self._add_model_lock.release()
            return

        if self._waiting_global_update and not local:
            self._handle_global_update(model, source)
            return

        self._add_pending_model(model, weight, source)
        
        # TODO: Adapt the message format
        if len(self.get_nodes_pending_models_to_aggregate()) >= len(self._federation_nodes):
            logging.info(f"🔄  include_model_in_buffer | Broadcasting ALL received models done for round {self.engine.get_round()}")
            message = self.cm.mm.generate_federation_message(nebula_pb2.FederationMessage.Action.FEDERATION_AGGREGATION_FINISHED, [self.engine.get_round()])
            await self.cm.send_message_to_neighbors(message)
            
        return

    def get_aggregation(self):
        if self._aggregation_done_lock.acquire(timeout=self.config.participant["aggregator_args"]["aggregation_timeout"]):
            try:
                self._aggregation_done_lock.release()
            except:
                pass
        else:
            logging.error(f"🔄  get_aggregation | Timeout reached for aggregation")

        if self._waiting_global_update and len(self._pending_models_to_aggregate) == 1:
            logging.info(f"🔄  get_aggregation | Received an global model. Overwriting my model with the aggregated model.")
            return next(iter(self._pending_models_to_aggregate.values()))[0]

        unique_nodes_involved = set(node for key in self._pending_models_to_aggregate for node in key.split())

        if len(unique_nodes_involved) != len(self._federation_nodes):
            missing_nodes = self._federation_nodes - unique_nodes_involved
            logging.info(f"🔄  get_aggregation | Aggregation incomplete, missing models from: {missing_nodes}")
        else:
            logging.info(f"🔄  get_aggregation | All models accounted for, proceeding with aggregation.")

        return self.run_aggregation(self._pending_models_to_aggregate)

    def print_model_size(self, model):
        total_params = 0
        total_memory = 0

        for _, param in model.items():
            num_params = param.numel()
            total_params += num_params

            memory_usage = param.element_size() * num_params
            total_memory += memory_usage

        total_memory_in_mb = total_memory / (1024**2)
        logging.info(f"print_model_size | Model size: {total_memory_in_mb} MB")


def create_malicious_aggregator(aggregator, attack):
    # It creates a partial function aggregate that wraps the aggregate method of the original aggregator.
    run_aggregation = partial(aggregator.run_aggregation)  # None is the self (not used)

    # This function will replace the original aggregate method of the aggregator.
    def malicious_aggregate(self, models):
        accum = run_aggregation(models)
        logging.info(f"malicious_aggregate | original aggregation result={accum}")
        if models is not None:
            accum = attack(accum)
            logging.info(f"malicious_aggregate | attack aggregation result={accum}")
        return accum

    aggregator.run_aggregation = partial(malicious_aggregate, aggregator)
    return aggregator
