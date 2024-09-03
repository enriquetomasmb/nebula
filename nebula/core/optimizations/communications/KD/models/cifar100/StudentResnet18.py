import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.models.studentnebulamodelV2 import StudentNebulaModelV2
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD.models.cifar100.TeacherResnet32 import TeacherCIFAR100ModelResNet32, MDTeacherCIFAR100ModelResNet32


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

        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-18
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(512 * BasicBlock.expansion, 512)
        self.fc = nn.Linear(512, num_classes)

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

    def forward(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x
        print(conv1.shape)
        x = self.layer1(x)  # 32x32
        conv2 = x
        print(conv2.shape)
        x = self.layer2(x)  # 16x16
        conv3 = x
        print(conv3.shape)
        x = self.layer3(x)  # 8x8
        conv4 = x
        print(conv4.shape)
        x = self.layer4(x)
        conv5 = x
        print(conv5.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

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
