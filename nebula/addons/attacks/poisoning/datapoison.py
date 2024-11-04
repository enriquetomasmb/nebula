import copy
import random

import numpy as np
import torch
from skimage.util import random_noise


def datapoison(
    dataset,
    indices,
    poisoned_persent,
    poisoned_ratio,
    targeted=False,
    target_label=3,
    noise_type="salt",
):
    """
    Function to add random noise of various types to the dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    train_data = new_dataset.data
    targets = new_dataset.targets
    num_indices = len(indices)
    if type(noise_type) != str:
        noise_type = noise_type[0]

    if targeted == False:
        num_poisoned = int(poisoned_persent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_poisoned > num_indices:
            return new_dataset
        poisoned_indice = random.sample(indices, num_poisoned)

        for i in poisoned_indice:
            t = train_data[i]
            if noise_type == "salt":
                # Replaces random pixels with 1.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
            elif noise_type == "gaussian":
                # Gaussian-distributed additive noise.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
            elif noise_type == "s&p":
                # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
            elif noise_type == "nlp_rawdata":
                # for NLP data, change the word vector to 0 with p=poisoned_ratio
                poisoned = poison_to_nlp_rawdata(t, poisoned_ratio)
            else:
                print("ERROR: poison attack type not supported.")
                poisoned = t
            train_data[i] = poisoned
    else:
        for i in indices:
            if int(targets[i]) == int(target_label):
                t = train_data[i]
                poisoned = add_x_to_image(t)
                train_data[i] = poisoned
    new_dataset.data = train_data
    return new_dataset


def add_x_to_image(img):
    """
    Add a 10*10 pixels X at the top-left of an image
    """
    for i in range(0, 10):
        for j in range(0, 10):
            if i + j <= 9 or i == j:
                img[i][j] = 255
    return torch.tensor(img)


def poison_to_nlp_rawdata(text_data, poisoned_ratio):
    """
    for NLP data, change the word vector to 0 with p=poisoned_ratio
    """
    non_zero_vector_indice = [i for i in range(0, len(text_data)) if text_data[i][0] != 0]
    non_zero_vector_len = len(non_zero_vector_indice)

    num_poisoned_token = int(poisoned_ratio * non_zero_vector_len)
    if num_poisoned_token == 0:
        return text_data
    if num_poisoned_token > non_zero_vector_len:
        return text_data

    poisoned_token_indice = random.sample(non_zero_vector_indice, num_poisoned_token)
    zero_vector = torch.Tensor(np.zeros(len(text_data[0][0])))
    for i in poisoned_token_indice:
        text_data[i] = zero_vector
    return text_data
