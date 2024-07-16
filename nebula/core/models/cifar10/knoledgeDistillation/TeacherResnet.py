import torch
import torch.multiprocessing
from nebula.core.knoledgeDistillation.SemCKD import SelfA, SemCKDLoss
torch.multiprocessing.set_sharing_strategy("file_system")

import torch.nn as nn
from nebula.core.models.cifar10.knoledgeDistillation.resnet import CIFAR10ModelResNet8
from nebula.core.knoledgeDistillation.AT import Attention
import copy

class TeacherCIFAR10ModelResNet14(CIFAR10ModelResNet8):
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
            depth=14,
            num_filters=[16, 16, 32, 64],
            block_name='BasicBlock',

    ):


        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, depth, num_filters, block_name)
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """ Configure the optimizer for training. """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        logits = self(images)
        loss = self.criterion_cls(logits, labels)
        self.process_metrics(phase, logits, labels, loss)
        return loss


class MDTeacherCIFAR10ModelResNet14(CIFAR10ModelResNet8):
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
            depth=14,
            num_filters=[16, 16, 32, 64],
            block_name='BasicBlock',
            p=2,
            beta=1000

    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, depth, num_filters,
                         block_name)
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.p = p
        self.beta = beta
        self.criterion_kd = Attention(self.p)

        self.student_model = None

    def configure_optimizers(self):
        """ Configure the optimizer for training. """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def set_student_model(self, student_model):
        """
        For the mutual distillation, the student model is set in the teacher model.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, 'teacher_model'):
            del self.student_model.teacher_model

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        feat_t, logit_t = self(images, is_feat=True)
        loss_cls = self.criterion_cls(logit_t, labels)
        # If the student model is set, the mutual distillation is applied
        if self.student_model is not None:
            with torch.no_grad():
                feat_s, logit_s = self.student_model(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            # Compute the attention loss
            loss_group = self.criterion_kd(g_t, g_s)
            loss_kd = sum(loss_group)
        else:
            loss_kd = 0 * loss_cls

        loss = loss_cls + self.beta * loss_kd
        self.process_metrics(phase, logit_t, labels, loss)
        return loss

class SemMDTeacherCIFAR10ModelResNet14(CIFAR10ModelResNet8):
    """
    LightningModule for CIFAR10.
    Still in DEVELOPMENT
    The problem is the initialization of the SelfA layer, because it needs an example of the input (maybe the example_input_array is enough), and
    the parameters of the SelfA layer have to be included in the optimizer, so they are updated too.

    """

    def __init__(
            self,
            input_channels=3,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            depth=14,
            num_filters=[16, 16, 32, 64],
            block_name='BasicBlock',
            p=2,
            beta=1000,
            student_model=None

    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, depth, num_filters,
                         block_name)
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.p = p
        self.beta = beta
        self.criterion_kd = SemCKDLoss() # Cambiar para SemCKD
        self.self_attention = None
        self.student_model = student_model
        if self.student_model is not None:
            if hasattr(self.student_model, 'teacher_model'):
                del self.student_model.teacher_model
            self.init_self_attention(self.example_input_array)

    def configure_optimizers(self):
        """ """
        parameters = nn.ModuleList([])
        if self.self_attention is not None:
            parameters.append(self.self_attention)

        parameters.append(self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        return optimizer

    def set_student_model(self, student_model):
        """
        Para evitar el problema de la dependencia cíclica, se crea una copia del modelo estudiante,
        se le quita el atributo teacher_model.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, 'teacher_model'):
            del self.student_model.teacher_model

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        feat_t, logit_t = self(images, is_feat=True)
        loss_cls = self.criterion_cls(logit_t, labels)

        if self.student_model is not None:
            with torch.no_grad():
                feat_s, logit_s = self.student_model(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            s_value, f_target, weight = self.self_attention(g_s, g_t)
            loss_kd = self.criterion_kd(s_value, f_target, weight)
        else:
            loss_kd = 0 * loss_cls

        loss = loss_cls + self.beta * loss_kd
        self.process_metrics(phase, logit_t, labels, loss)
        return loss

    def init_self_attention(self, input_array):
        batch_size = 32
        feat_t, _ = self(input_array, is_feat=True)
        feat_s, _ = self.student_model(input_array, is_feat=True)
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        self.self_attention = SelfA(len(feat_s)-2, len(feat_t)-2, batch_size, s_n, t_n)
