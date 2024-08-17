import copy
import torch


def labelflipping_fang(dataset):
    """
    Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
    Minghong Fang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong
    https://arxiv.org/abs/1911.11815

    Label flipping attack. This is a data poisoning attack that does not require knowledge
    of the training data distribution. On each compromised worker device, this attack flips
    the label of each training instance. Specifically, we flip a label l as L − l − 1,
    where L is the number of classes in the classification problem and l = 0,1,··· ,L−1.

    dataset: the dataset of training data, torch.util.data.dataset like.
    :return:
    """
    new_dataset = copy.copy(dataset)
    targets = new_dataset.targets.detach().clone()
    class_list = new_dataset.class_to_idx.values()

    for i in range(len(targets.tolist())):
        t = targets[i].numpy()
        targets[i] = torch.tensor(len(class_list) - t - 1)

    new_dataset.targets = targets
    return new_dataset
