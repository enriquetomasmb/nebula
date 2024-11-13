"""
This module provides a function for adding noise to a machine learning model's parameters, simulating
data poisoning attacks. The main function allows for the injection of various types of noise into
the model parameters, effectively altering them to test the model's robustness against malicious
manipulations.

Function:
- modelpoison: Modifies the parameters of a model by injecting noise according to a specified ratio
  and type of noise (e.g., Gaussian, salt, salt-and-pepper).
"""

from collections import OrderedDict

import torch
from skimage.util import random_noise


def modelpoison(model: OrderedDict, poisoned_ratio, noise_type="gaussian"):
    """
    Adds random noise to the parameters of a model for the purpose of data poisoning.

    This function modifies the model's parameters by injecting noise according to the specified
    noise type and ratio. Various types of noise can be applied, including salt noise, Gaussian
    noise, and salt-and-pepper noise.

    Args:
        model (OrderedDict): The model's parameters organized as an `OrderedDict`. Each key corresponds
                             to a layer, and each value is a tensor representing the parameters of that layer.
        poisoned_ratio (float): The proportion of noise to apply, expressed as a fraction (0 <= poisoned_ratio <= 1).
        noise_type (str, optional): The type of noise to apply to the model parameters. Supported types are:
                                    - "salt": Applies salt noise, replacing random elements with 1.
                                    - "gaussian": Applies Gaussian-distributed additive noise.
                                    - "s&p": Applies salt-and-pepper noise, replacing random elements with either 1 or low_val.
                                    Default is "gaussian".

    Returns:
        OrderedDict: A new `OrderedDict` containing the model parameters with noise added.

    Raises:
        ValueError: If `poisoned_ratio` is not between 0 and 1, or if `noise_type` is unsupported.

    Notes:
        - If a layer's tensor is a single point (0-dimensional), it will be reshaped for processing.
        - Unsupported noise types will result in an error message, and the original tensor will be retained.
    """
    poisoned_model = OrderedDict()
    if not isinstance(noise_type, str):
        noise_type = noise_type[0]

    for layer in model:
        bt = model[layer]
        t = bt.detach().clone()
        single_point = False
        if len(t.shape) == 0:
            t = t.view(-1)
            single_point = True
        # print(t)
        if noise_type == "salt":
        #Salt Noise:
        # Effect: Randomly sets some of the model's parameters to the maximum value (e.g., 1).
        # Purpose: Simulates sharp, sudden disturbances in the model's parameters. It can be used to test how well the model can handle extreme outliers or unexpected changes in parameter values.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        elif noise_type == "gaussian":
        # Gaussian Noise:
        # Effect: Adds random values drawn from a Gaussian distribution to the model's parameters.
        # Purpose: Simulates random variations in the model's parameters, similar to sensor noise or measurement errors. It can help in testing the model's robustness to small, continuous perturbations and can also act as a form of regularization to prevent overfitting.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
        elif noise_type == "s&p":
        # Salt-and-Pepper (S&P) Noise:
        # Effect: Randomly sets some of the model's parameters to either the minimum value (e.g., 0) or the maximum value (e.g., 1).
        # Purpose: Simulates a combination of sharp disturbances (both high and low) in the model's parameters. It can be used to test the model's robustness to a mix of extreme outliers and can help in understanding how the model handles a wide range of parameter perturbations.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        else:
            print("ERROR: poison attack type not supported.")
            poisoned = t
        if single_point:
            poisoned = poisoned[0]
        poisoned_model[layer] = poisoned

    return poisoned_model
