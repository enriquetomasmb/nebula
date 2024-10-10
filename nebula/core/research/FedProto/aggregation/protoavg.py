import gc
import torch

from nebula.core.aggregation.aggregator import Aggregator


class ProtoAvg(Aggregator):
    """
    Prototype Aggregation (FedProto) [Yue Tan et al., 2022]
    Paper: https://arxiv.org/abs/2105.00243
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, prototypes):
        super().run_aggregation(prototypes)

        prototypes = list(prototypes.values())

        # Total Samples
        total_samples = sum(w for _, w in prototypes)

        # Create a Zero Prototype
        accum = {label: torch.zeros_like(proto_info) for label, proto_info in prototypes[-1][0].items()}

        # Add weighted models
        for prototype, weight in prototypes:
            for label in prototype:
                if label not in accum:
                    accum[label] = torch.zeros_like(prototype[label])
                accum[label] += prototype[label] * weight
        # Normalize Accum
        for label in accum:
            accum[label] /= total_samples

        del prototypes, total_samples
        gc.collect()

        return accum
