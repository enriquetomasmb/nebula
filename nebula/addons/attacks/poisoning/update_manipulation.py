import logging

import torch


def update_manipulation_LIE(parameters, z):
    """
    Attack from Paper: "A Little Is Enough: Circumventing Defenses For Distributed Learning" by Moran Baruch et al.
    :param parameters: Honest parameters (calculated by client)
    :param z: by how many standard deviations the parameter gets skewed
    :return: Malicious parameters
    """
    logging.info("[Attack update_manipulation_LIE] running attack on model parameters")
    malicious_parameters = {}
    for key, value in parameters.items():
        if key.endswith("bias"):
            malicious_parameters[key] = value
        else:
            new_weights_list = []
            for weights in value:
                new_weights = []
                avg = torch.mean(weights, dim=0)
                std = torch.std(weights, dim=0)
                for _ in weights:
                    # new_weights.append(avg + z * std)
                    new_weights.append(1)
                new_weights_list.append(new_weights)
            malicious_parameters[key] = torch.tensor(new_weights_list)
    logging.info("[Attack update_manipulation_LIE] finished")

    return malicious_parameters