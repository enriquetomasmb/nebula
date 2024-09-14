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


def rclamp(value, min_val, max_val):
    if value >= min_val:
        return max_val
    return value


class AdaptiveWeighting(Weighting):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def get_alpha(self, ce_value):
        return 1 / rclamp(ce_value, self.min_val, self.max_val)

    def get_beta(self, ce_value, kl_divergence_value):
        return 1 / rclamp(ce_value + kl_divergence_value, self.min_val, self.max_val)

    def get_lambda(self, ce_value, kl_divergence_value, mse_value):
        return 1 / rclamp(ce_value + kl_divergence_value + mse_value, self.min_val, self.max_val)
