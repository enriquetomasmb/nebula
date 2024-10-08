import logging
import random
import torch
from nebula.core.aggregation.aggregator import Aggregator


class DynamicAggregator(Aggregator):
    def __init__(self, config = None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models, tensorboard_log=True):
        logging.info(f"[DynamicAggregator] Initializing Aggregation")
        from nebula.core.aggregation.fedavg import FedAvg
        from nebula.core.aggregation.krum import Krum
        from nebula.core.aggregation.median import Median
        from nebula.core.aggregation.trimmedmean import TrimmedMean
        from nebula.core.aggregation.bulyan import Bulyan
        super().run_aggregation(models)
        available_aggregators = [FedAvg, Krum, Median, TrimmedMean, Bulyan]
        # to check if the random seed is working
        import time
        random.seed(int(time.time()) % 2 ** 32)
        chosen_aggregator_cls = random.choice(available_aggregators)
        logging.info(f"[DynamicAggregator] Chosen Aggregator: {chosen_aggregator_cls}")
        if tensorboard_log:
            self.engine.nebulalogger.log_text(tag="[DynamicAggregator] Chosen Aggregator", text=chosen_aggregator_cls.__name__, step=self.engine.round)
        chosen_aggregator = chosen_aggregator_cls(config=self.config)
        return chosen_aggregator.run_aggregation(models)
