"""
This module implements the ProtoFedAvg aggregator for federated learning, combining both prototype model aggregation
and traditional FedAvg.
"""

import torch

from nebula.core.aggregation.aggregator import Aggregator


def _agregate_prototypes(prototypes):
    """
    Weighted average of the prototype.
    Args:
        prototypes: Dictionary with the prototypes (node: prototypes, num_samples)
    """
    if len(prototypes) == 0:
        return None

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

    return accum


class ProtoFedAvg(Aggregator):
    """
    Mix Prototype Model Aggregation. Prototype Model Aggregation and Normal FedAvg are implemented

    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):
        super().run_aggregation(models)

        # Get a dictionary with the prototypes with (node: prototypes, num_samples)
        proto_flag = False
        prototypes = {}
        for node, model_info in models.items():
            if "protos" in model_info[0]:
                proto_flag = True
                prototypes[node] = model_info[0]["protos"], model_info[1]
                del model_info[0]["protos"]

        models = list(models.values())

        # Total Samples
        total_samples = sum(w for _, w in models)

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1][0].items()}

        # Add weighted models
        for model, weight in models:
            for layer in accum:
                # Convert to accum type (float16 or float32)
                model[layer] = model[layer].to(accum[layer].dtype)
                accum[layer] += model[layer] * weight

        # Normalize Accum
        for layer in accum:
            # Convert to float32
            accum[layer] = accum[layer].float()
            accum[layer] /= total_samples

        if proto_flag:
            agregated_protos = _agregate_prototypes(prototypes)
            accum["protos"] = agregated_protos

        return accum
