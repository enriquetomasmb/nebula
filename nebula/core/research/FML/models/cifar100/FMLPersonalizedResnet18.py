import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from nebula.core.research.FML.models.fmlpersonalizednebulamodel import FMLPersonalizedNebulaModel
from nebula.core.research.FML.utils.KD import FMLDistillKL


class FMLCIFAR100PersonalizedModelResNet18(FMLPersonalizedNebulaModel):
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
        T=2,
        alpha=1,
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

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = FMLDistillKL(self.T)

        self.resnet = resnet18()
        self.resnet.fc_dense = nn.Linear(self.resnet.fc.in_features, 512)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x, softmax=False, is_feat=False):
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
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, [conv1, conv2, conv3, conv4, conv5]

        del conv1, conv2, conv3, conv4, conv5

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
