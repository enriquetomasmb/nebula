import copy
import random
import torch


def labelflipping_untargeted(dataset, indices, flipping_persent):
    """
    select flipping_persent of labels, and change them to random values.
    Args:
        dataset: the dataset of training data, torch.util.data.dataset like.
        indices: Indices of subsets, list like.
        flipping_persent: The ratio of labels want to change, float like.
    """
    new_dataset = copy.copy(dataset)

    if type(new_dataset.targets) == list:
        new_dataset.targets = torch.tensor(new_dataset.targets)
    targets = new_dataset.targets.detach().clone()
    num_indices = len(indices)
    classes = new_dataset.classes
    class_to_idx = new_dataset.class_to_idx
    class_list = [class_to_idx[i] for i in classes]

    num_flipped = int(flipping_persent * num_indices)
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
    new_dataset.targets = targets

    return new_dataset
