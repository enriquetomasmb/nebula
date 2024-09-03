import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherResnet18 import (
    MDProtoTeacherCIFAR10ModelResnet18,
    ProtoTeacherCIFAR10ModelResnet18,
)
from nebula.core.optimizations.communications.KD_prototypes.models.protostudentnebulamodel import ProtoStudentNebulaModel


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


class ProtoStudentCIFAR10ModelResnet8(ProtoStudentNebulaModel):
    """
    LightningModule for MNIST.
    """

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
        beta_proto=1,
        decreasing_beta=False,
        limit_beta=0.1,
        mutual_distilation="KD",
        teacher_beta=100,
        send_logic=None,
    ):
        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                teacher_model = MDProtoTeacherCIFAR10ModelResnet18(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = ProtoTeacherCIFAR10ModelResnet18()

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
            beta_proto,
            decreasing_beta,
            limit_beta,
            mutual_distilation,
            send_logic,
        )
        self.embedding_dim = 512
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.beta_proto = beta_proto
        self.beta_kd = beta_kd
        self.criterion_nll = nn.NLLLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-8
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(128 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, self.num_classes)

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

    def forward_train(self, x, softmax=True, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x = self.avgpool(x2)
        x = torch.flatten(x, 1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [x1, x2]
            return logits, dense, [x1, x2]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo."""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Obtener las características intermedias
        logits, dense, features = self.forward_train(x, is_feat=True)

        # Calcular distancias a los prototipos globales
        distances = []
        for key, proto in self.global_protos.items():
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        return distances.argmin(dim=1)

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

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images, softmax=False)

        logits_softmax = F.log_softmax(logits, dim=1)
        protos_copy = protos.clone()
        with torch.no_grad():
            teacher_logits, teacher_protos = self.teacher_model.forward_train(images, softmax=False)

        # Compute loss 1
        loss1 = self.criterion_nll(logits_softmax, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss2 = self.criterion_mse(proto_new, protos)

        # Compute loss knowledge distillation
        loss3 = self.criterion_div(logits, teacher_logits)

        # Compute loss knowledge distillation for the prototypes
        loss4 = self.criterion_mse(protos_copy, teacher_protos)

        # Combine the losses
        loss = loss1 + self.beta_proto * loss2 + self.beta_kd * (0.5 * loss3 + 0.5 * loss4)

        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Update the prototypes
            self.model_updated_flag2 = True
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss
