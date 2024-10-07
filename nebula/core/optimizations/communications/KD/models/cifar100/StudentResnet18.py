import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from nebula.core.optimizations.communications.KD.models.studentnebulamodelV2 import StudentNebulaModelV2
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD.models.cifar100.TeacherResnet32 import TeacherCIFAR100ModelResNet32, MDTeacherCIFAR100ModelResNet32


class StudentCIFAR100ModelResNet18(StudentNebulaModelV2):
    """
    LightningModule for CIFAR100.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=100,
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
                teacher_model = MDTeacherCIFAR100ModelResNet32(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = TeacherCIFAR100ModelResNet32()

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

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.beta_kd = beta_kd
        self.criterion_nll = nn.NLLLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)
        self.resnet = resnet18(num_classes=num_classes)

    def forward(self, x, softmax=True, is_feat=False):
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
        x = self.resnet.layer4(x)
        conv5 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.resnet.fc(x)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, [conv1, conv2, conv3, conv4, conv5]

        if softmax:
            return F.log_softmax(logits, dim=1)
        return logits

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):
        if phase == "Train":
            self.model_updated_flag2 = True
        images, labels = batch
        student_logits = self(images, softmax=False)
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
