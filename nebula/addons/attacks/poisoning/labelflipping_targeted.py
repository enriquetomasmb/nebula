import copy
from typing import Union

import torch
import random
from torch.utils.data import Dataset
import logging


def labelflipping_targeted_specific(dataset, indices, label_og: Union[list, int], label_goal: int):
    """
    This attack changes the label(s) given in label_og to the label given in label_goal
    :param dataset: Dataset to flip the labels of
    :param indices: Indices of subsets where the attack will be applied
    :param label_og: The original label(s) / class ID(s) which will be changed to label_goal
    :param label_goal: The label / class ID to which label_og will be changed
    :return:
    """
    new_dataset = copy.copy(dataset)
    try:
        targets = new_dataset.targets.detach().clone()
    except AttributeError:
        targets = new_dataset.targets
    logging.info("[LabelFlipping Attack] Changing labels from {} to {}".format(label_og, label_goal))

    for i in indices:
        try:
            t = targets[i].numpy()
        except AttributeError:
            t = targets[i]
        if (t in label_og) or (str(t) in label_og):
            targets[i] = label_goal
    new_dataset.targets = targets
    return new_dataset


def labelflipping_targeted_unspecific(dataset, indices, label_og: Union[list, int]):
    """
    This attack changes the label given in label_og to some other label (selected randomly)
    :param dataset: Dataset to flip the labels of
    :param indices: Indices of subsets where the attack will be applied
    :param label_og: The label(s) / class ID(s) which will be changed to some other label (selected randomly)
    :return:
    """
    new_dataset = copy.copy(dataset)
    targets = new_dataset.targets.detach().clone()
    class_list = new_dataset.class_to_idx.values()
    logging.info("[LabelFlipping Attack] Changing labels from {} randomly.".format(label_og))

    for i in indices:
        t = targets[i].numpy()
        if (t in label_og) or (str(t) in label_og):
            targets[i] = torch.tensor(
                random.sample(sorted([x for x in class_list if x != t]), 1)
            )

    new_dataset.targets = targets
    return new_dataset
