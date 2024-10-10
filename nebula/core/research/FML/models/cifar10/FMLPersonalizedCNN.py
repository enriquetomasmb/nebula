import torch

from nebula.core.research.FML.models.fmlpersonalizednebulamodel import FMLPersonalizedNebulaModel
from nebula.core.research.FML.utils.KD import FMLDistillKL


class FMLCIFAR10PersonalizedModelCNN(FMLPersonalizedNebulaModel):
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
        T=2,
        beta=1,
    ):

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
            beta,
        )

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()
        self.criterion_div = FMLDistillKL(self.T)

        # Define layers of the model
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x, is_feat=False):
        """
        Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        logits = self.fc_layer(flattened)

        del flattened

        if is_feat:
            return logits, [conv1, conv2, conv3]

        del conv1, conv2, conv3
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
