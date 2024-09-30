from nebula.core.optimizations.adaptative_weighted.weighting import Weighting

"""
AdaptiveWeighting.py

This module implements the AdaptiveWeighting class based on the principles outlined in the thesis
"Creation of a Framework for Decentralized Federated Knowledge Distillation" by Philip Giryes.

The class utilizes adaptive weighting techniques to dynamically adjust parameters for improved performance
in federated learning environments with non-IID data.

Functions:
- clamp(value, min, max): Restricts a value within a given range.
- rclamp(value, min, max): Variant that restricts the value to an upper limit.
- AdaptiveWeighting: Main class for adaptive weighting in federated learning models.

Author: Miguel Fernández Llamas
Date: 05/09/2024
"""


def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


class AdaptiveWeighting(Weighting):
    def __init__(self, min_weight=0.1, max_weight=5.0, scale=1.0):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.scale = scale

    def get_alpha(self, ce_protos_value):
        epsilon = 1e-8  # Para evitar división por cero
        alpha = self.scale / (ce_protos_value + epsilon)
        # Clampeamos el valor para que esté entre min_weight y max_weight
        alpha = clamp(alpha, self.min_weight, self.max_weight)
        return alpha

    def get_beta(self, ce_protos_value, kl_divergence_value):
        epsilon = 1e-8
        beta = self.scale / (ce_protos_value + kl_divergence_value + epsilon)
        beta = clamp(beta, self.min_weight, self.max_weight)
        return beta

    def get_lambda(self, ce_protos_value, kl_divergence_value, mse_value):
        epsilon = 1e-8
        lambda_weight = self.scale / (ce_protos_value + kl_divergence_value + mse_value + epsilon)
        lambda_weight = clamp(lambda_weight, self.min_weight, self.max_weight)
        return lambda_weight
