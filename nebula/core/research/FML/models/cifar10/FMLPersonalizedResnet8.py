import torch
import torch.multiprocessing
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.resnet8 import ResNet8
from nebula.core.research.FML.models.fmlpersonalizednebulamodel import FMLPersonalizedNebulaModel
from nebula.core.research.FML.utils.KD import FMLDistillKL


class FMLCIFAR10PersonalizedModelResNet8(FMLPersonalizedNebulaModel):

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        alpha=0.2,
    ):

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
            alpha,
        )

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_div = FMLDistillKL(self.T)
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()
        self.resnet = ResNet8()
        self.resnet.fc_dense = nn.Linear(self.resnet.fc.in_features, 512)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x, softmax=False, is_feat=False):
        """Forward pass for inference the model, if model have prototypes"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        conv1 = self.resnet.relu(x)

        x = self.resnet.maxpool(conv1)

        conv2 = self.resnet.layer1(x)
        conv3 = self.resnet.layer2(conv2)
        conv4 = self.resnet.layer3(conv3)

        x = self.resnet.avgpool(conv4)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)
        logits = self.resnet.fc(dense)

        del dense

        if is_feat:
            if softmax:
                return F.softmax(logits), [conv1, conv2, conv3, conv4]
            else:
                return logits, [conv1, conv2, conv3, conv4]
        else:
            del conv1, conv2, conv3, conv4
            if softmax:
                return F.softmax(logits)

        return logits

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer
