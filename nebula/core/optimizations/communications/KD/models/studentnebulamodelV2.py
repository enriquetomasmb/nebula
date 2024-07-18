from abc import ABC

import torch

from nebula.core.models.nebulamodel import NebulaModel

class StudentNebulaModelV2(NebulaModel, ABC):

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
            beta=1,
            decreasing_beta=False,
            limit_beta=0.1,
            send_logic=None

    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

        self.limit_beta = limit_beta
        self.decreasing_beta = decreasing_beta
        self.beta = beta
        self.teacher_model = teacher_model
        self.T = T
        if send_logic is not None:
            self.send_logic_method = send_logic
        else:
            self.send_logic_method = None
        self.send_logic_counter = 0
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
                        'While copying the parameter named {}, whose dimensions in the saved model are {} and whose dimensions in the current model are {}, an error occurred: {}'.format(
                            name, param.size(), own_state[name].size(), e))
            elif strict:
                # If the mode is strict, warn that this parameter was not found.
                missing_keys.append(name)

        if strict:
            # Check if there are any missing keys.
            missing_keys = set(own_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                message = "Error loading state_dict, missing keys:{} and unexpected keys:{}".format(missing_keys,
                                                                                                    unexpected_keys)
                raise KeyError(message)


        return


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        si send_logic() == 0: solo envía el modelo estudiante
        si send_logic() == 1: solo envía las capas fully connected
        """
        if self.send_logic() == 0:
            original_state = super().state_dict(destination, prefix, keep_vars)
            # Filter out teacher model parameters
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
        elif self.send_logic() == 1:
            original_state = super().state_dict(destination, prefix, keep_vars)
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
            filtered_state = {k: v for k, v in filtered_state.items() if k.startswith('l2.')}

        return filtered_state


    def send_logic(self):
        """
        Send logic.
        """
        if self.send_logic_method is None:
            return 0

        if self.send_logic_method == 'model':
            return 0
        elif self.send_logic_method == 'mixed_2rounds':
            if self.send_logic_counter % 2 == 0:
                return 0
            else:
                return 1


    def send_logic_step(self):
        """
        Send logic step.
        """
        self.send_logic_counter += 1
        if self.decreasing_beta:
            self.beta = self.beta / 2
            if self.beta < self.limit_beta:
                self.beta = 0

        if self.send_logic_method is None:
            return 'model'
        if self.send_logic_method == 'mixed_2rounds':

            if self.send_logic_counter % 2 == 0:
                return 'linear_layers'
            else:
                return 'model'
        else:
            return 'unknown'