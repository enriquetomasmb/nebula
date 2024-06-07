import copy
import random

import torch


def labelFlipping(dataset, indices, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7):
    """
    select flipping_persent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_persent: The ratio of labels want to change, float like.
    """
    new_dataset = copy.deepcopy(dataset)
    targets = new_dataset.targets.detach().clone()
    num_indices = len(indices)
    # classes = new_dataset.classes
    # class_to_idx = new_dataset.class_to_idx
    # class_list = [class_to_idx[i] for i in classes]
    class_list = set(targets.tolist())
    if targeted == False:
        num_flipped = int(poisoned_persent * num_indices)
        if num_indices == 0:
            return new_dataset
        if num_flipped > num_indices:
            return new_dataset
        flipped_indice = random.sample(indices, num_flipped)

        for i in flipped_indice:
            t = targets[i]
            flipped = torch.tensor(random.sample(class_list, 1)[0])
            while t == flipped:
                flipped = torch.tensor(random.sample(class_list, 1)[0])
            targets[i] = flipped
    else:
        for i in indices:
            if int(targets[i]) == int(target_label):
                targets[i] = torch.tensor(target_changed_label)
    new_dataset.targets = targets
    return new_dataset
