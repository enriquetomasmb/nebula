"""
This module defines the GlobalPrototypeDistillationLoss class, used for computing distillation loss based
on global and local prototypes within a federated learning framework, specifically tailored for heterogeneous data distributions.

The concept is derived from the 'Global Prototype Distillation for Heterogeneous Federated Learning' as presented
in the paper by Wu et al. (2024). The approach uses global class prototypes as distilled knowledge to guide local
training, aiming to improve the alignment of local models with the global model, thereby enhancing overall performance.

Reference:
Wu, S., Chen, J., Nie, X., Wang, Y., Zhou, X., Lu, L., Peng, W., Nie, Y., & Menhaj, W. (2024).
Global prototype distillation for heterogeneous federated learning. Scientific Reports, 14, 12057.
https://doi.org/10.1038/s41598-024-62908-0
"""

import torch
from torch import nn
import torch.nn.functional as F


class GlobalPrototypeDistillationLoss(nn.Module):
    """
    Global Prototype Distillation Loss for Federated Learning
    This class computes the distillation loss using global and local prototypes.
    """

    def __init__(self, temperature=2):
        super().__init__()
        self.temperature = temperature

    def forward(self, global_protos, local_features, labels):
        """
        Compute the distillation loss.

        Parameters:
        - global_protos: Dictionary containing global prototypes for each class.
        - local_features: Tensor containing local features to be distilled.
        - labels: Tensor containing the labels corresponding to the local features.

        Returns:
        - loss_gpd: The computed global prototype distillation loss.
        """
        local_softmax = F.softmax(local_features / self.temperature, dim=1)
        loss_gpd = 0
        for i, feature in enumerate(local_features):
            class_idx = labels[i].item()
            if class_idx in global_protos and global_protos[class_idx] is not None:
                global_proto = global_protos[class_idx].to(feature.device)
                global_softmax = F.softmax(global_proto / self.temperature, dim=0)
                loss_gpd += torch.sum(global_softmax * torch.log(local_softmax[i]))

        del local_softmax

        return -loss_gpd
