import asyncio
import logging

from collections import deque
from abc import ABC, abstractmethod
from nebula.addons.functions import print_msg_box

from typing import TYPE_CHECKING
from typing import List, Tuple, Any, Optional

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager
    from nebula.config.config import Config
    from nebula.core.aggregation.aggregator import Aggregator
    from nebula.core.engine import Engine
    from nebula.core.training.lightning import Lightning


class PropagationStrategy(ABC):
    @abstractmethod
    def is_node_eligible(self, node: str) -> bool:
        pass

    @abstractmethod
    def prepare_model_payload(self, node: str) -> Optional[Tuple[Any, List[str], float]]:
        pass


class InitialModelPropagation(PropagationStrategy):
    def __init__(self, aggregator: "Aggregator", trainer: "Lightning", engine: "Engine"):
        self.aggregator = aggregator
        self.trainer = trainer
        self.engine = engine

    def get_round(self):
        return self.engine.get_round()

    def is_node_eligible(self, node: str) -> bool:
        return node not in self.engine.cm.get_ready_connections()

    def prepare_model_payload(self, node: str) -> Optional[Tuple[Any, List[str], float]]:
        return self.trainer.get_model_parameters(), self.trainer.DEFAULT_MODEL_WEIGHT


class StableModelPropagation(PropagationStrategy):
    def __init__(self, aggregator: "Aggregator", trainer: "Lightning", engine: "Engine"):
        self.aggregator = aggregator
        self.trainer = trainer
        self.engine = engine
        self.addr = self.engine.get_addr()

    def get_round(self):
        return self.engine.get_round()

    def is_node_eligible(self, node: str) -> bool:
        return (node not in self.aggregator.get_nodes_pending_models_to_aggregate()) or (self.engine.cm.connections[node].get_federated_round() < self.get_round())

    def prepare_model_payload(self, node: str) -> Optional[Tuple[Any, List[str], float]]:
        return self.trainer.get_model_parameters(), self.trainer.get_model_weight()


class Propagator:
    def __init__(self, cm: "CommunicationsManager"):
        self.engine: Engine = cm.engine
        self.config: Config = cm.get_config()
        self.addr = cm.get_addr()
        self.cm: "CommunicationsManager" = cm
        self.aggregator: "Aggregator" = self.engine.aggregator
        self.trainer: "Lightning" = self.engine._trainer

        self.status_history = deque(maxlen=self.config.participant["propagator_args"]["history_size"])

        self.interval = self.config.participant["propagator_args"]["propagate_interval"]
        self.model_interval = self.config.participant["propagator_args"]["propagate_model_interval"]
        self.early_stop = self.config.participant["propagator_args"]["propagation_early_stop"]
        self.stable_rounds_count = 0

        # Propagation strategies (adapt to the specific use case)
        self.strategies = {"initialization": InitialModelPropagation(self.aggregator, self.trainer, self.engine), "stable": StableModelPropagation(self.aggregator, self.trainer, self.engine)}

    def start(self):
        print_msg_box(msg=f"Starting propagator functionality...\nModel propagation through the network", indent=2, title="Propagator")

    def get_round(self):
        return self.engine.get_round()

    def update_and_check_neighbors(self, strategy, eligible_neighbors):
        # Update the status of eligible neighbors
        current_status = [n for n in eligible_neighbors]

        # Check if the deque is full and the new status is different from the last one
        if self.status_history and current_status != self.status_history[-1]:
            logging.info(f"Status History deque is full and the new status is different from the last one: {list(self.status_history)}")
            self.status_history.append(current_status)
            return True

        # Add the current status to the deque
        logging.info(f"Adding current status to the deque: {current_status}")
        self.status_history.append(current_status)

        # If the deque is full and all elements are the same, stop propagation
        if len(self.status_history) == self.status_history.maxlen and all(s == self.status_history[0] for s in self.status_history):
            logging.info(f"Propagator exited for {self.status_history.maxlen} equal rounds: {list(self.status_history)}")
            return False

        return True

    def reset_status_history(self):
        self.status_history.clear()

    async def propagate(self, strategy_id: str):
        if strategy_id not in self.strategies:
            logging.info(f"Strategy {strategy_id} not found.")
            return False
        if self.get_round() is None:
            logging.info("Propagation halted: round is not set.")
            return False

        strategy = self.strategies[strategy_id]
        logging.info(f"Starting model propagation with strategy: {strategy_id}")

        eligible_neighbors = [neighbor_addr for neighbor_addr in self.cm.get_addrs_current_connections(only_direct=True) if strategy.is_node_eligible(neighbor_addr)]
        logging.info(f"Eligible neighbors for model propagation: {eligible_neighbors}")
        if not eligible_neighbors:
            logging.info("Propagation complete: No eligible neighbors.")
            return False

        logging.info(f"Checking repeated statuses during propagation")
        if not self.update_and_check_neighbors(strategy, eligible_neighbors):
            logging.info("Exiting propagation due to repeated statuses.")
            return False

        for neighbor_addr in eligible_neighbors:
            serialized_model, weight = strategy.prepare_model_payload(neighbor_addr)
            if serialized_model:
                serialized_model = serialized_model if isinstance(serialized_model, bytes) else self.trainer.serialize_model(serialized_model)
                if strategy_id == "initialization":
                    await self.cm.send_model(neighbor_addr, -1, serialized_model, weight)
                else:
                    await self.cm.send_model(neighbor_addr, self.get_round(), serialized_model, weight)
            await asyncio.sleep(self.model_interval)

        await asyncio.sleep(self.interval)
        return True

    async def propagate_continuously(self, strategy_id: str):
        self.reset_status_history()
        while True:
            propagated = await self.propagate(strategy_id)
            if not propagated:
                logging.info("Exiting continuous propagation...")
                return
