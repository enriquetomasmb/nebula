import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.AT import Attention
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.prototeachernebulamodel import ProtoTeacherNebulaModel, MDProtoTeacherNebulaModel


class ProtoTeacherCIFAR10ModelCNN(ProtoTeacherNebulaModel):
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
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )
        self.embedding_dim = 512
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.4),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5),
        )

        self.fc_layer_dense = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )

        self.fc_layer = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        # First convolutional layer
        conv1 = self.layer1(input_layer)

        # Second convolutional layer
        conv2 = self.layer2(conv1)

        # Third convolutional layer
        conv3 = self.layer3(conv2)

        # Flatten the tensor
        flattened = conv3.view(conv3.size(0), -1)

        # Fully connected layers
        dense = self.fc_layer_dense(flattened)
        logits = self.fc_layer(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2, conv3]
            return logits, dense, [conv1, conv2, conv3]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits
        input_layer = x.view(-1, 3, 32, 32)
        conv1 = self.layer1(input_layer)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        dense = self.fc_layer_dense(flattened)

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            # send protos and dense to the same device
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


class MDProtoTeacherCIFAR10ModelCNN(MDProtoTeacherNebulaModel):
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
        p=2,
        alpha_kd=1,
        beta_feat=1000,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
            p,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )
        self.embedding_dim = 512
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_feat = Attention(self.p)
        self.criterion_kd = DistillKL(self.T)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.4),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5),
        )

        self.fc_layer_dense = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )

        self.fc_layer = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        # First convolutional layer
        conv1 = self.layer1(input_layer)

        # Second convolutional layer
        conv2 = self.layer2(conv1)

        # Third convolutional layer
        conv3 = self.layer3(conv2)

        # Flatten the tensor
        flattened = conv3.view(conv3.size(0), -1)

        # Fully connected layers
        dense = self.fc_layer_dense(flattened)
        logits = self.fc_layer(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2, conv3]
            return logits, dense, [conv1, conv2, conv3]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits
        input_layer = x.view(-1, 3, 32, 32)
        conv1 = self.layer1(input_layer)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        dense = self.fc_layer_dense(flattened)

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            # send protos and dense to the same device
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
