from abc import ABC

import torch
from nebula.core.optimizations.communications.KD.models.studentnebulamodel import StudentNebulaModel
import logging

class ProtoStudentNebulaModel(StudentNebulaModel, ABC):

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
            mutual_distilation=False,
            send_logic=None

    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, teacher_model, T)

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        if send_logic is not None:
            self.send_logic_method = send_logic
        else:
            self.send_logic_method = None
        self.send_logic_counter = 0
        self.mutual_distilation = mutual_distilation
        self.global_protos = dict()
        self.agg_protos_label = dict()

    def get_protos(self):
        """
        Get the protos for the model.
        """

        if len(self.agg_protos_label) == 0:
            return {k: v.cpu() for k, v in self.global_protos.items()}

        proto = dict()
        for label, proto_info in self.agg_protos_label.items():

            if proto_info['count'] > 1:
                proto[label] = (proto_info['sum'] / proto_info['count']).to('cpu')
            else:
                proto[label] = proto_info['sum'].to('cpu')

        #logging.info(f"[ProtoFashionMNISTModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        """
        Set the protos for the model.
        """
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
        if self.teacher_model is not None:
            self.teacher_model.set_protos(protos)

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict to handle missing teacher model keys gracefully.
        """
        self.model_updated_flag1 = True

        # Obten el state_dict actual del modelo completo para preparar una comparación.
        own_state = self.state_dict()
        missing_keys = []

        for name, param in state_dict.items():
            if name in own_state:
                if name == 'protos':
                    self.set_protos(param)
                    continue
                # Intenta cargar el parámetro si existe en el estado actual del modelo.
                if isinstance(param, torch.nn.Parameter):
                    # Los parámetros son invariantes; necesitamos los datos.
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception as e:
                    raise RuntimeError(
                        'While copying the parameter named {}, whose dimensions in the saved model are {} and whose dimensions in the current model are {}, an error occurred: {}'.format(
                            name, param.size(), own_state[name].size(), e))
            elif strict:
                # Si el modo es estricto, avisa que este parámetro no fue encontrado.
                missing_keys.append(name)

        if strict:
            # Revisa si hay parámetros faltantes o inesperados.
            missing_keys = set(own_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                message = "Error loading state_dict, missing keys:{} and unexpected keys:{}".format(missing_keys,
                                                                                                    unexpected_keys)
                raise KeyError(message)


        return


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        original_state = super().state_dict(destination, prefix, keep_vars)
        # Filter out teacher model parameters
        if self.send_logic() == 0:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
        elif self.send_logic() == 1:
            filtered_state = dict()
            filtered_state['protos'] = self.get_protos()
        elif self.send_logic() == 2:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
            filtered_state['protos'] = self.get_protos()
        else:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
            filtered_state['protos'] = self.get_protos()

        return filtered_state

    def send_logic(self):
        """
        Send logic for ProtoFedAvg.
        if send_logic() == 0: only send the student model
        if send_logic() == 1: only send the protos
        if send_logic() == 2: send both the student model and the protos
        """
        if self.send_logic_method is None:
            return 0

        if self.send_logic_method == 'only_protos':
            return 1
        elif self.send_logic_method == 'only_model':
            return 0
        elif self.send_logic_method == 'both':
            return 2
        elif self.send_logic_method == 'mixed_2rounds':
            if self.send_logic_counter % 2 == 0:
                return 2
            else:
                return 1


    def send_logic_step(self):
        """
        Send logic for ProtoFedAvg.
        if send_logic() == 0: only send the student model
        if send_logic() == 1: only send the protos
        """
        if self.send_logic_method is None:
            return 'model'

        if self.send_logic_method == 'mixed_2rounds':
            self.send_logic_counter += 1
            if self.send_logic_counter % 2 == 0:
                return 'protos'
            else:
                return 'model'
        else:
            return 'unknown'