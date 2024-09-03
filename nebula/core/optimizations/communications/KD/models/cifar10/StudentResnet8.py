import torch
import torch.multiprocessing
from torch import nn
import torch.nn.functional as F
from nebula.core.optimizations.communications.KD.models.studentnebulamodelV2 import StudentNebulaModelV2
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


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
        beta=1,
        decreasing_beta=False,
        limit_beta=0.1,
        send_logic=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
        )
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
        self.criterion_div = DistillKL(self.T)
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-8
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x = self.avgpool(x2)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        if is_feat:
            return logits, [x1, x2]

        return logits

    def step(self, batch, batch_idx, phase):
        if phase == "Train":
            self.model_updated_flag2 = True
        images, labels = batch
        student_logits = self(images)
        loss_ce = self.criterion_cls(student_logits, labels)
        # If the beta is greater than the limit, apply knowledge distillation
        if self.beta > self.limit_beta and self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)
            # Compute the KD loss
            loss_kd = self.criterion_div(student_logits, teacher_logits)
            # Combine the losses
            loss = loss_ce + self.beta * loss_kd
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
            self.beta = self.beta / 2
            if self.beta < self.limit_beta:
                self.beta = 0

        if self.send_logic_method is None:
            return "model"
        if self.send_logic_method == "mixed_2rounds":

            if self.send_logic_counter % 2 == 0:
                return "linear_layers"
            return "model"
        return "unknown"
