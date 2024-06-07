from collections import OrderedDict

import torch
from skimage.util import random_noise


def modelpoison(model: OrderedDict, poisoned_ratio, noise_type="gaussian"):
    """
    Function to add random noise of various types to the model parameter.
    """
    poisoned_model = OrderedDict()
    if type(noise_type) != type("salt"):
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
            # Replaces random pixels with 1.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        elif noise_type == "gaussian":
            # Gaussian-distributed additive noise.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
        elif noise_type == "s&p":
            # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
            poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
        else:
            print("ERROR: poison attack type not supported.")
            poisoned = t
        if single_point:
            poisoned = poisoned[0]
        poisoned_model[layer] = poisoned

    return poisoned_model
