from abc import ABC

import torch

from nebula.core.models.nebulamodel import NebulaModel


class StudentNebulaModel(NebulaModel, ABC):

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        teacher_model=None,
        T=2,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

        self.teacher_model = teacher_model
        self.T = T
        self.model_updated_flag1 = False
        self.model_updated_flag2 = False

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict to handle missing teacher model keys gracefully.
        """
        self.model_updated_flag1 = True

        # Get the current state_dict of the full model to prepare for comparison.
        own_state = self.state_dict()
        missing_keys = []

        for name, param in state_dict.items():
            if name in own_state:
                # Try to load the parameter if it exists in the current model state.
                if isinstance(param, torch.nn.Parameter):
                    # The parameters are invariant; we need the data.
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    raise RuntimeError(
                        "While copying the parameter named {}, whose dimensions in the saved model are {} and whose dimensions in the current model are {}, an error occurred: {}".format(
                            name, param.size(), own_state[name].size(), e
                        )
                    ) from e
            elif strict:
                # If the mode is strict, warn that this parameter was not found.
                missing_keys.append(name)

        if strict:
            # Check if there are any missing keys.
            missing_keys = set(own_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                message = "Error loading state_dict, missing keys:{} and unexpected keys:{}".format(missing_keys, unexpected_keys)
                raise KeyError(message)

        return

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        original_state = super().state_dict(destination, prefix, keep_vars)
        # Filter out teacher model parameters
        filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
        return filtered_state
