import torch
from torchvision.models import resnet18
import torch.nn.functional as F
from torch import nn

from nebula.core.optimizations.adaptative_weighted.adaptativeweighting import AdaptiveWeighting
from nebula.core.optimizations.adaptative_weighted.decreasingweighting import DeacreasingWeighting
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.cifar100.ProtoTeacherResnet34 import (
    ProtoTeacherCIFAR100ModelResNet34,
    MDProtoTeacherCIFAR100ModelResNet34,
)
from nebula.core.optimizations.communications.KD_prototypes.models.protostudentnebulamodel import ProtoStudentNebulaModel
from nebula.core.optimizations.communications.KD_prototypes.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


class ProtoStudentCIFAR100ModelResnet18(ProtoStudentNebulaModel):
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
        T=10,
        alpha_kd=0.5,
        beta_feat=0.3,
        lambda_proto=0.2,
        knowledge_distilation="KD",
        send_logic=None,
        weighting=None,
    ):

        if weighting == "adaptative":
            self.weighting = AdaptiveWeighting(min_weight=1, max_weight=10)
        elif weighting == "decreasing":
            self.weighting = DeacreasingWeighting(alpha_value=alpha_kd, beta_value=beta_feat, lambda_value=lambda_proto, limit=0.1, rounds=200)
        if teacher_model is None:
            if knowledge_distilation is not None and knowledge_distilation == "MD":
                teacher_model = MDProtoTeacherCIFAR100ModelResNet34(weighting=weighting)
            elif knowledge_distilation is not None and knowledge_distilation == "KD":
                teacher_model = ProtoTeacherCIFAR100ModelResNet34(weighting=weighting)

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            teacher_model,
            T,
            alpha_kd,
            beta_feat,
            lambda_proto,
            knowledge_distilation,
            send_logic,
            weighting,
        )
        self.embedding_dim = 512
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kd = DistillKL(self.T)
        self.criterion_gpd = GlobalPrototypeDistillationLoss(temperature=2)
        self.resnet = resnet18(num_classes=num_classes)
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
        x = self.resnet.layer4(x)
        conv5 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)
        logits = self.resnet.fc(dense)

        del x

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, dense, [conv1, conv2, conv3, conv4, conv5]

        del conv1, conv2, conv3, conv4, conv5

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
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)

        del x
        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        del dense
        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        # Excluir los parámetros del modelo del profesor
        student_params = [p for name, p in self.named_parameters() if not name.startswith("teacher_model.")]
        optimizer = torch.optim.Adam(
            student_params,
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer
