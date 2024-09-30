from abc import ABC

import torch

from nebula.core.optimizations.adaptative_weighted.decreasingweighting import DeacreasingWeighting
from nebula.core.optimizations.adaptative_weighted.weighting import Weighting
from nebula.core.optimizations.communications.KD.models.studentnebulamodel import StudentNebulaModel
from nebula.core.optimizations.adaptative_weighted.adaptativeweighting import AdaptiveWeighting
from nebula.core.optimizations.communications.KD_prototypes.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


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
        alpha_kd=0.5,
        beta_feat=0.3,
        lambda_proto=0.2,
        knowledge_distilation="KD",
        send_logic="both",
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            teacher_model,
            T,
        )

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        if weighting == "adaptative":
            self.weighting = AdaptiveWeighting(min_val=1, max_val=10)
        elif weighting == "decreasing":
            self.weighting = DeacreasingWeighting(alpha_value=alpha_kd, beta_value=beta_feat, lambda_value=lambda_proto, limit=0.1)
        else:
            self.weighting = Weighting(alpha_value=alpha_kd, beta_value=beta_feat, lambda_value=lambda_proto)

        if send_logic is not None:
            self.send_logic_method = send_logic
        else:
            self.send_logic_method = None
        self.model_updated_flag = False
        self.send_logic_counter = 0
        self.knowledge_distilation = knowledge_distilation
        self.global_protos = dict()
        self.agg_protos_label = dict()
        self.criterion_gpd = GlobalPrototypeDistillationLoss(temperature=T)

    def get_protos(self):
        """
        Get the protos for the model.
        """

        if len(self.agg_protos_label) == 0:
            return {k: v.cpu() for k, v in self.global_protos.items()}

        proto = dict()
        for label, proto_info in self.agg_protos_label.items():

            if proto_info["count"] > 1:
                proto[label] = (proto_info["sum"] / proto_info["count"]).to("cpu")
            else:
                proto[label] = proto_info["sum"].to("cpu")

        # logging.info(f"[ProtoFashionMNISTModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        """
        Set the protos for the model.
        """
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
        if self.teacher_model is not None:
            self.teacher_model.set_protos(protos)

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images, softmax=False)

        protos_copy = protos.clone()
        with torch.no_grad():
            teacher_logits, teacher_protos = self.teacher_model.forward_train(images, softmax=False)

        # Compute loss 1
        loss_ce = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_ce
        else:
            """
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            """
            # Compute the loss for the prototypes
            loss_protos = self.criterion_gpd(self.global_protos, protos, labels)

        # Compute loss knowledge distillation
        loss_kd = self.criterion_kd(logits, teacher_logits)

        # Compute loss knowledge distillation for the prototypes
        loss_protos_teacher = self.criterion_mse(protos_copy, teacher_protos)

        # Combine the losses
        loss_ce_protos = loss_ce + 0.05 * loss_protos

        loss = loss_ce_protos + self.weighting.get_alpha(loss_ce_protos) * loss_kd + self.weighting.get_beta(loss_ce_protos, loss_kd) * loss_protos_teacher

        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Update the prototypes
            self.model_updated_flag = True
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        elif phase == "Validation":
            if self.model_updated_flag:
                self.model_updated_flag = False
                self.send_logic_step()

        return loss

    def load_state_dict(self, state_dict, strict=False):
        """
        Overrides the default load_state_dict to handle missing teacher model keys gracefully.
        """
        # Obten el state_dict actual del modelo completo para preparar una comparación.
        own_state = self.state_dict()
        missing_keys = []

        for name, param in state_dict.items():
            if name in own_state:
                if name == "protos":
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
                        "While copying the parameter named {}, whose dimensions in the saved model are {} and whose dimensions in the current model are {}, an error occurred: {}".format(
                            name, param.size(), own_state[name].size(), e
                        )
                    ) from e

            if name == "protos":
                self.set_protos(param)
                continue

            if strict:
                # Si el modo es estricto, avisa que este parámetro no fue encontrado.
                missing_keys.append(name)

        if strict:
            # Revisa si hay parámetros faltantes o inesperados.
            missing_keys = set(own_state.keys()) - set(state_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(own_state.keys())
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                message = "Error loading state_dict, missing keys:{} and unexpected keys:{}".format(missing_keys, unexpected_keys)
                raise KeyError(message)

        return

    def state_dict(self, destination=None, prefix="", keep_vars=False):

        original_state = super().state_dict(destination, prefix, keep_vars)
        # Filter out teacher model parameters
        if self.send_logic() == 0:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
        elif self.send_logic() == 1:
            filtered_state = dict()
            filtered_state["protos"] = self.get_protos()
        elif self.send_logic() == 2:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
            filtered_state["protos"] = self.get_protos()
        else:
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
            filtered_state["protos"] = self.get_protos()

        return filtered_state

    def send_logic(self):
        """
        Send logic for ProtoFedAvg.
        if send_logic() == 0: only send the student model
        if send_logic() == 1: only send the protos
        if send_logic() == 2: send both the student model and the protos

        """
        if self.send_logic_method is None:
            return 2

        if self.send_logic_method == "only_protos":
            return 1
        if self.send_logic_method == "only_model":
            return 0
        if self.send_logic_method == "both":
            return 2
        if self.send_logic_method == "mixed_2rounds":
            if self.send_logic_counter % 2 == 0:
                return 2
            return 1
        return 0

    def send_logic_step(self):
        """
        Send logic for ProtoFedAvg.
        if send_logic() == 0: only send the student model
        if send_logic() == 1: only send the protos
        """
        self.send_logic_counter += 1
        if hasattr(self.weighting, "decreasing_step"):
            self.weighting.decreasing_step()

        if self.send_logic_method is None:
            return "model"
        if self.send_logic_method == "mixed_2rounds":

            if self.send_logic_counter % 2 == 0:
                return "linear_layers"
            return "model"
        return "unknown"
