import torch
import torch.multiprocessing
from nebula.core.optimizations.communications.KD.models.cifar10.resnet import CIFAR10ModelResNet8
from nebula.core.optimizations.communications.KD.models.cifar10.TeacherResnet import MDTeacherCIFAR10ModelResNet14, TeacherCIFAR10ModelResNet14
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL

torch.multiprocessing.set_sharing_strategy("file_system")

__all__ = ["resnet"]


class StudentModelResNet8(CIFAR10ModelResNet8):

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        depth=8,
        num_filters=[16, 16, 32, 64],
        block_name="BasicBlock",
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
            depth,
            num_filters,
            block_name,
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


class StudentCIFAR10ModelResNet8(StudentModelResNet8):
    """
    LightningModule for CIFAR10.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        depth=8,
        num_filters=[16, 16, 32, 64],
        block_name="BasicBlock",
        teacher_model=None,
        T=2,
        beta=1,
        decreasing_beta=False,
        limit_beta=0.1,
        mutual_distilation="KD",
        teacher_beta=100,
        send_logic=None,
    ):
        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                teacher_model = MDTeacherCIFAR10ModelResNet14(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = TeacherCIFAR10ModelResNet14()

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            depth,
            num_filters,
            block_name,
            teacher_model,
            T,
            beta,
            decreasing_beta,
            limit_beta,
            send_logic,
        )
        self.teacher_model = teacher_model
        self.T = T
        self.mutual_distilation = mutual_distilation
        self.beta = beta
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)

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
