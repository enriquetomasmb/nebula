import torch
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoTeacherCNN import MDProtoTeacherMNISTModelCNN, ProtoTeacherMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.protostudentnebulamodel import ProtoStudentNebulaModel


class ProtoStudentMNISTModelCNN(ProtoStudentNebulaModel):
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
        teacher_model=None,
        T=2,
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        knowledge_distilation="KD",
        send_logic=None,
        weighting=None,
    ):
        if teacher_model is None:
            if knowledge_distilation is not None and knowledge_distilation == "MD":
                teacher_model = MDProtoTeacherMNISTModelCNN(weighting=weighting)
            elif knowledge_distilation is not None and knowledge_distilation == "KD":
                teacher_model = ProtoTeacherMNISTModelCNN(weighting=weighting)

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
        self.embedding_dim = 2048
        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kd = DistillKL(self.T)

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding="same",
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 64, self.embedding_dim)
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
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

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
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

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
        logits, protos = self.forward_train(images, softmax=False)

        protos_copy = protos.clone()
        with torch.no_grad():
            teacher_logits, teacher_protos = self.teacher_model.forward_train(images, softmax=False)

        # Compute loss 1
        loss_ce = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_ce
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        # Compute loss knowledge distillation
        loss_kd = self.criterion_kd(logits, teacher_logits)

        # Compute loss knowledge distillation for the prototypes
        loss_protos_teacher = self.criterion_mse(protos_copy, teacher_protos)

        # Combine the losses
        loss = (
            loss_ce + self.weighting.get_alpha(loss_ce) * loss_kd + self.weighting.get_beta(loss_ce, loss_kd) * (0.8 * loss_protos + 0.2 * loss_protos_teacher)
        )

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
