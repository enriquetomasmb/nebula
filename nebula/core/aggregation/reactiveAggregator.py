import logging
from nebula.core.aggregation.aggregator import Aggregator

class ReactiveAggregator(Aggregator):
    def __init__(self, config = None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        logging.info(f"[ReactiveAggregator] Initializing Aggregation")
        super().run_aggregation(models)

        if self.engine._is_malicious:
            from nebula.core.aggregation.dynamicAggregator import DynamicAggregator
            logging.info(f"[ReactiveAggregator] Malicious Node - Using Dynamic Aggregator")
            dynamic_aggregator = DynamicAggregator(config=self.config)
            return dynamic_aggregator.run_aggregation(models, tensorboard_log = False)
        else:
            default_aggregator = self.config.participant["aggregator_args"]["reactive_aggregator_default"]
            logging.info(f"[ReactiveAggregator] Normal Node - Using Aggregator {default_aggregator}")
            from nebula.core.aggregation.fedavg import FedAvg
            from nebula.core.aggregation.krum import Krum
            from nebula.core.aggregation.median import Median
            from nebula.core.aggregation.trimmedmean import TrimmedMean
            from nebula.core.aggregation.bulyan import Bulyan
            from nebula.core.aggregation.blockchainReputation import BlockchainReputation
            from nebula.core.aggregation.dynamicAggregator import DynamicAggregator
            ALGORITHM_MAP = {
                "FedAvg": FedAvg,
                "Krum": Krum,
                "Median": Median,
                "TrimmedMean": TrimmedMean,
                "Bulyan": Bulyan,
                "BlockchainReputation": BlockchainReputation,
                "DynamicAggregator": DynamicAggregator,
            }
            if default_aggregator not in ALGORITHM_MAP:
                logging.error(f"[ReactiveAggregator] Invalid default aggregator {default_aggregator}, falling back to FedAvg")
                default_aggregator = "FedAvg"
            default_aggregator_cls = ALGORITHM_MAP[default_aggregator]
            default_aggregator = default_aggregator_cls(config=self.config)
            return default_aggregator.run_aggregation(models)
