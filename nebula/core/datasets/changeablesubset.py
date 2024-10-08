import copy
import logging

from torch.utils.data import Subset

from nebula.addons.attacks.poisoning import *


class ChangeableSubset(Subset):
    def __init__(self, dataset, indices, label_flipping=False, label_flipping_config=None, data_poisoning=False, poisoned_persent=0, poisoned_ratio=0, targeted=False, target_label=0, target_changed_label=0, noise_type="salt"):
        super().__init__(dataset, indices)
        # TODO: Improve the efficiency of the following code
        new_dataset = copy.copy(dataset)
        self.dataset = new_dataset
        self.indices = indices
        self.label_flipping = label_flipping
        self.label_flipping_config = label_flipping_config
        self.data_poisoning = data_poisoning
        self.poisoned_persent = poisoned_persent
        self.poisoned_ratio = poisoned_ratio
        self.targeted = targeted if isinstance(targeted, list) else [targeted]
        self.target_label = target_label
        self.target_changed_label = target_changed_label
        self.noise_type = noise_type

        if self.label_flipping:
            logging.info("[Labelflipping] Received attack: {}".format(self.label_flipping_config["attack"]))
            if self.label_flipping_config["attack"] == "label_flipping_targeted_specific":
                self.dataset = labelflipping_targeted_specific(
                    self.dataset,
                    self.indices,
                    self.label_flipping_config["label_og"],
                    self.label_flipping_config["label_goal"]
                )
            elif self.label_flipping_config["attack"] == "label_flipping_targeted_unspecific":
                self.dataset = labelflipping_targeted_unspecific(
                    self.dataset,
                    self.indices,
                    self.label_flipping_config["label_og"]
                )
            elif self.label_flipping_config["attack"] == "label_flipping_untargeted":
                self.dataset = labelflipping_untargeted(
                    self.dataset,
                    self.indices,
                    self.label_flipping_config["sample_percent"]
                )
            elif self.label_flipping_config["attack"] == "label_flipping_fang":
                self.dataset = labelflipping_fang(self.dataset)
            logging.info("[Labelflipping] Dataset manipulated (attack: {})".format(self.label_flipping_config["attack"]))

        if self.data_poisoning:
            self.dataset = datapoison(self.dataset, self.indices, self.poisoned_persent, self.poisoned_ratio, self.targeted, self.target_label, self.noise_type)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
