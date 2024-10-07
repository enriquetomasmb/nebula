import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherResnet18 import (
    MDProtoTeacherCIFAR10ModelResnet18,
    ProtoTeacherCIFAR10ModelResnet18,
)
from nebula.core.optimizations.communications.KD_prototypes.models.protostudentnebulamodel import ProtoStudentNebulaModel
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.resnet8 import ResNet8


class ProtoStudentCIFAR10ModelResnet8(ProtoStudentNebulaModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=0.01,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        teacher_model=None,
        T=2,
        alpha_kd=0.5,
        beta_feat=0.3,
        lambda_proto=0.2,
        knowledge_distilation="KD",
        send_logic=None,
        weighting=None,
    ):
        if teacher_model is None:
            if knowledge_distilation is not None and knowledge_distilation == "MD":
                teacher_model = MDProtoTeacherCIFAR10ModelResnet18(weighting=weighting)
            elif knowledge_distilation is not None and knowledge_distilation == "KD":
                teacher_model = ProtoTeacherCIFAR10ModelResnet18(weighting=weighting)

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
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer
