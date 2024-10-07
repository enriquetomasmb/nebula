import torch
import torch.multiprocessing
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.models.cifar10.TeacherResnet18 import MDTeacherCIFAR10ModelResNet18, TeacherCIFAR10ModelResNet18
from nebula.core.optimizations.communications.KD.models.studentnebulamodelV2 import StudentNebulaModelV2
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.resnet8 import ResNet8


class StudentCIFAR10ModelResNet8(StudentNebulaModelV2):

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        teacher_model=None,
        T=2,
        beta_kd=1,
        decreasing_beta=False,
        limit_beta=0.1,
        mutual_distilation="KD",
        teacher_beta=100,
        send_logic=None,
    ):

        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                teacher_model = MDTeacherCIFAR10ModelResNet18(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = TeacherCIFAR10ModelResNet18()

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            teacher_model,
            T,
            beta_kd,
            decreasing_beta,
            limit_beta,
            send_logic,
        )
        self.limit_beta = limit_beta
        self.decreasing_beta = decreasing_beta
        self.beta_kd = beta_kd
        self.teacher_model = teacher_model
        self.T = T
        if send_logic is not None:
            self.send_logic_method = send_logic
        else:
            self.send_logic_method = None
        self.send_logic_counter = 0
        self.model_updated_flag1 = False
        self.model_updated_flag2 = False
        self.criterion_div = DistillKL(self.T)
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()

        self.resnet = ResNet8()
        self.resnet.fc_dense = nn.Linear(self.resnet.fc.in_features, self.embedding_dim)
        self.resnet.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        conv1 = x

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        conv2 = x
        x = self.resnet.layer2(x)
        conv3 = x
        x = self.resnet.layer3(x)
        conv4 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)
        logits = self.resnet.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4],
                )
            return logits, dense, [conv1, conv2, conv3, conv4]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, phase):
        if phase == "Train":
            self.model_updated_flag2 = True
        images, labels = batch
        student_logits = self(images)
        loss_ce = self.criterion_cls(student_logits, labels)
        # If the beta is greater than the limit, apply knowledge distillation
        if self.beta_kd > self.limit_beta and self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)
            # Compute the KD loss
            loss_kd = self.criterion_div(student_logits, teacher_logits)
            # Combine the losses
            loss = loss_ce + self.beta_kd * loss_kd
        else:
            loss = loss_ce

        self.process_metrics(phase, student_logits, labels, loss)
        return loss

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
            elif strict:
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
        """
        si send_logic() == 0: solo envía el modelo estudiante
        si send_logic() == 1: solo envía las capas fully connected
        """
        if self.send_logic() == 0:
            original_state = super().state_dict(destination, prefix, keep_vars)
            # Filter out teacher model parameters
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
        elif self.send_logic() == 1:
            original_state = super().state_dict(destination, prefix, keep_vars)
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith("teacher_model.")}
            filtered_state = {k: v for k, v in filtered_state.items() if k.startswith("fc.")}

        return filtered_state

    def send_logic(self):
        """
        Send logic.
        """
        if self.send_logic_method is None:
            return 0

        if self.send_logic_method == "model":
            return 0

        if self.send_logic_method == "mixed_2rounds":
            if self.send_logic_counter % 2 == 0:
                return 0
            return 1

        return 0

    def send_logic_step(self):
        """
        Send logic step.
        """
        self.send_logic_counter += 1
        if self.decreasing_beta:
            self.beta_kd = self.beta_kd / 2
            if self.beta_kd < self.limit_beta:
                self.beta_kd = 0

        if self.send_logic_method is None:
            return "model"
        if self.send_logic_method == "mixed_2rounds":

            if self.send_logic_counter % 2 == 0:
                return "linear_layers"
            return "model"
        return "unknown"
