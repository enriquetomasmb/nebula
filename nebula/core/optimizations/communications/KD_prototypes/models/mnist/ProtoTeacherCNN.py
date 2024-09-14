import copy

import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.AT import Attention
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.prototeachernebulamodel import ProtoTeacherNebulaModel


class ProtoTeacherMNISTModelCNN(ProtoTeacherNebulaModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
        self,
        input_channels=1,
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

        self.embedding_dim = 2048
        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(5, 5),
            padding="same",
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 128, self.embedding_dim)
        self.l2 = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2]
            return logits, dense, [conv1, conv2]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))

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
        """Configure the optimizer for training."""
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

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_cls
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        # Combine the losses
        loss = loss_cls + self.weighting.get_beta(loss_cls) * loss_protos
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss


class MDProtoTeacherMNISTModelCNN(ProtoTeacherNebulaModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        alpha_kd=1,
        beta_feat=100,
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
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )

        self.embedding_dim = 2048
        self.p = 2
        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_feat = Attention(self.p)
        self.criterion_kd = DistillKL(self.T)
        self.student_model = None
        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(5, 5),
            padding="same",
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 128, self.embedding_dim)
        self.l2 = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2]
            return logits, dense, [conv1, conv2]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))

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
        """Configure the optimizer for training."""
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
        logits, protos, feat_t = self.forward_train(images, is_feat=True, softmax=False)

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_cls
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1

            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        if self.student_model is not None:
            with torch.no_grad():
                student_logits, student_protos, feat_s = self.student_model.forward_train(images, is_feat=True, softmax=False)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s
            g_t = feat_t
            # Compute the mutual distillation loss
            loss_kd = self.criterion_kd(student_logits, logits)
            loss_group = self.criterion_feat(g_t, g_s)
            loss_feat = sum(loss_group)
        else:
            loss_feat = 0 * loss_cls
            loss_kd = 0 * loss_cls

        # Combine the losses
        loss = (
            loss_cls
            + self.weighting.get_alpha(loss_cls) * loss_kd
            + self.weighting.get_beta(loss_cls, loss_kd) * loss_feat
            + self.weighting.get_lambda(loss_cls, loss_kd, loss_feat) * loss_protos
        )
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss

    def set_student_model(self, student_model):
        """
        For cyclic dependency problem, a copy of the student model is created, the teacher_model attribute is removed.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, "teacher_model"):
            del self.student_model.teacher_model
