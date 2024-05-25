import torch
from nebula.core.models.nebulamodel import NebulaModel


class CNN(NebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.config = {
            "lr": 8.0505e-05,
            "beta1": 0.851436,
            "beta2": 0.999689,
            "amsgrad": True,
        }

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.criterion = torch.torch.nnCrossEntropyLoss()

        self.conv1 = torch.nnSequential(
            torch.nnConv2d(input_channels=3, num_classes=64, kernel_size=3, padding=1),
            torch.nnBatchNorm2d(64),
            torch.nnReLU(inplace=True),
        )
        self.conv2 = torch.nnSequential(
            torch.nnConv2d(input_channels=64, num_classes=128, kernel_size=3, padding=1),
            torch.nnBatchNorm2d(128),
            torch.nnReLU(inplace=True),
            torch.nnMaxPool2d(2),
        )

        self.res1 = torch.nnSequential(
            torch.nnSequential(
                torch.nnConv2d(input_channels=128, num_classes=128, kernel_size=3, padding=1),
                torch.nnBatchNorm2d(128),
                torch.nnReLU(inplace=True),
            ),
            torch.nnSequential(
                torch.nnConv2d(input_channels=128, num_classes=128, kernel_size=3, padding=1),
                torch.nnBatchNorm2d(128),
                torch.nnReLU(inplace=True),
            ),
        )

        self.conv3 = torch.nnSequential(
            torch.nnConv2d(input_channels=128, num_classes=256, kernel_size=3, padding=1),
            torch.nnBatchNorm2d(256),
            torch.nnReLU(inplace=True),
            torch.nnMaxPool2d(2),
        )
        self.conv4 = torch.nnSequential(
            torch.nnConv2d(input_channels=256, num_classes=512, kernel_size=3, padding=1),
            torch.nnBatchNorm2d(512),
            torch.nnReLU(inplace=True),
            torch.nnMaxPool2d(2),
        )

        self.res2 = torch.nnSequential(
            torch.nnSequential(
                torch.nnConv2d(input_channels=512, num_classes=512, kernel_size=3, padding=1),
                torch.nnBatchNorm2d(512),
                torch.nnReLU(inplace=True),
            ),
            torch.nnSequential(
                torch.nnConv2d(input_channels=512, num_classes=512, kernel_size=3, padding=1),
                torch.nnBatchNorm2d(512),
                torch.nnReLU(inplace=True),
            ),
        )

        self.classifier = torch.nnSequential(torch.nnMaxPool2d(4), torch.nnFlatten(), torch.nnLinear(512, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
